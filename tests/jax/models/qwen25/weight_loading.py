# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for Qwen2.5 models.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
from tqdm import tqdm
import safetensors.numpy

# Import get_partition_specs from tensor_parallel
from tensor_parallel import get_partition_specs

# Set up logging
logger = logging.getLogger(__name__)

def load_safetensors_index(model_path: str) -> Dict[str, Any]:
    """
    Load the safetensors index file which contains weight file mapping information.
    
    Args:
        model_path: Path to the model directory containing model.safetensors.index.json
        
    Returns:
        Dictionary from the index file
    """
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        logger.error(f"Safetensors index file not found at {index_path}")
        raise FileNotFoundError(f"Safetensors index file not found at {index_path}")
    
    logger.info(f"Loading safetensors index from {index_path}")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    # Extract the weight map for easier use
    if "weight_map" in index:
        return index["weight_map"]
    
    return index

def convert_weight_name_to_flax(pt_name: str) -> str:
    """
    Convert a PyTorch weight name to the equivalent Flax parameter name.
    
    Args:
        pt_name: PyTorch parameter name (from safetensors)
        
    Returns:
        Flax parameter name
    """
    # Main parameter mapping dictionary
    QWEN_PARAMETER_MAPPING = {
        r'model\.embed_tokens\.weight': r'model/embed_tokens/embedding',
        r'model\.norm\.weight': r'model/norm/weight',
        r'lm_head\.weight': r'lm_head/kernel',
        
        # Layer parameters
        r'model\.layers\.(\d+)\.input_layernorm\.weight': r'model/layers_\1/input_layernorm/weight',
        r'model\.layers\.(\d+)\.post_attention_layernorm\.weight': r'model/layers_\1/post_attention_layernorm/weight',
        
        # Attention parameters
        r'model\.layers\.(\d+)\.self_attn\.q_proj\.weight': r'model/layers_\1/self_attn/q_proj/kernel',
        r'model\.layers\.(\d+)\.self_attn\.q_proj\.bias': r'model/layers_\1/self_attn/q_proj/bias',
        r'model\.layers\.(\d+)\.self_attn\.k_proj\.weight': r'model/layers_\1/self_attn/k_proj/kernel',
        r'model\.layers\.(\d+)\.self_attn\.k_proj\.bias': r'model/layers_\1/self_attn/k_proj/bias',
        r'model\.layers\.(\d+)\.self_attn\.v_proj\.weight': r'model/layers_\1/self_attn/v_proj/kernel',
        r'model\.layers\.(\d+)\.self_attn\.v_proj\.bias': r'model/layers_\1/self_attn/v_proj/bias',
        r'model\.layers\.(\d+)\.self_attn\.o_proj\.weight': r'model/layers_\1/self_attn/o_proj/kernel',
        r'model\.layers\.(\d+)\.self_attn\.o_proj\.bias': r'model/layers_\1/self_attn/o_proj/bias',
        
        # MLP parameters
        r'model\.layers\.(\d+)\.mlp\.gate_proj\.weight': r'model/layers_\1/mlp/gate_proj/kernel',
        r'model\.layers\.(\d+)\.mlp\.gate_proj\.bias': r'model/layers_\1/mlp/gate_proj/bias',
        r'model\.layers\.(\d+)\.mlp\.up_proj\.weight': r'model/layers_\1/mlp/up_proj/kernel',
        r'model\.layers\.(\d+)\.mlp\.up_proj\.bias': r'model/layers_\1/mlp/up_proj/bias',
        r'model\.layers\.(\d+)\.mlp\.down_proj\.weight': r'model/layers_\1/mlp/down_proj/kernel',
        r'model\.layers\.(\d+)\.mlp\.down_proj\.bias': r'model/layers_\1/mlp/down_proj/bias',
    }
    
    # Apply the mapping
    flax_name = pt_name
    for pt_pattern, flax_pattern in QWEN_PARAMETER_MAPPING.items():
        if re.match(pt_pattern, pt_name):
            flax_name = re.sub(pt_pattern, flax_pattern, pt_name)
            break
    
    if flax_name == pt_name:
        logger.warning(f"No mapping found for parameter: {pt_name}")
    else:
        logger.debug(f"Mapped parameter {pt_name} → {flax_name}")
        
    return flax_name

def load_qwen_weights(
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[jax.sharding.Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
    progress_callback = None,
) -> Dict:
    """
    Load weights from safetensors files with tensor parallelism.
    
    Args:
        model_path: Path to model safetensors files
        config: Model configuration dictionary
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        debug: Whether to print debug information
        progress_callback: Optional callback function(current, total) for progress reporting
        
    Returns:
        Dictionary of model parameters
    """
    try:
        # Set up timing and progress tracking
        start_time = time.time()
        last_progress_time = start_time
        last_file_time = start_time
        
        # Load safetensors index mapping
        param_file_map = load_safetensors_index(model_path)
        
        if debug:
            print(f"Found {len(param_file_map)} parameters in the index file")
            if len(param_file_map) < 10:
                print("Parameters:", list(param_file_map.keys()))
        
        # Get the list of all parameter names
        param_names = list(param_file_map.keys())
        total_params = len(param_names)
        
        print(f"Starting to load {total_params} parameters from {model_path}")
        print(f"This may take several minutes. Progress updates will be shown.")
        sys.stdout.flush()
        
        # Create partition specs for tensor parallelism
        if mesh is not None:
            partition_specs = get_partition_specs(config)
        else:
            partition_specs = None
        
        # Initialize the parameter dictionary - using a flat structure first
        flat_params = {}
        
        # Map of tensors to load directly from files
        file_handles = {}
        
        # Track which parameters we've loaded
        loaded_params = set()
        
        # Collect unique files - ensuring they're absolute paths
        all_files = set()
        for file_name in param_file_map.values():
            if not os.path.isabs(file_name):
                file_path = os.path.join(model_path, file_name)
            else:
                file_path = file_name
            all_files.add(file_path)
        
        print(f"Loading weights from {len(all_files)} safetensors file(s)")
        sys.stdout.flush()
        
        # Check if files exist before proceeding
        missing_files = []
        for file_path in all_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                
        if missing_files:
            print(f"ERROR: Missing {len(missing_files)} weight file(s):")
            for file_path in missing_files[:3]:
                print(f"  - {file_path}")
            if len(missing_files) > 3:
                print(f"  ... and {len(missing_files) - 3} more")
            raise FileNotFoundError(f"Missing weight files. Check the model path: {model_path}")
        
        # Load weights from each file
        for i, name in enumerate(param_names):
            # Report progress more frequently
            current_time = time.time()
            if progress_callback is not None and i % 5 == 0:
                progress_callback(i, total_params)
            
            # Also report progress in the console every 2 seconds
            if current_time - last_progress_time > 2:
                elapsed = current_time - start_time
                percent = (i / total_params) * 100
                print(f"Loading weights: {i}/{total_params} parameters ({percent:.1f}%) - {elapsed:.1f}s elapsed")
                sys.stdout.flush()
                last_progress_time = current_time
            
            # Get file name from param_file_map and convert to full path if needed
            file_name = param_file_map[name]
            if not os.path.isabs(file_name):
                file_path = os.path.join(model_path, file_name)
            else:
                file_path = file_name
            
            # Create file handle if we don't have one yet
            if file_path not in file_handles:
                try:
                    # Report on file opening
                    file_name = os.path.basename(file_path)
                    print(f"Opening file: {file_name}")
                    sys.stdout.flush()
                    last_file_time = time.time()
                    
                    from safetensors import safe_open
                    file_handles[file_path] = safe_open(file_path, framework="numpy")
                    
                    # Report how long it took to open the file
                    file_open_time = time.time() - last_file_time
                    print(f"File opened in {file_open_time:.2f}s: {file_name}")
                    sys.stdout.flush()
                    
                    if debug:
                        print(f"Opened safetensors file: {file_path}")
                except Exception as e:
                    print(f"ERROR opening safetensors file {file_path}: {e}")
                    raise
            
            # Get the tensor from the file
            try:
                tensor = file_handles[file_path].get_tensor(name)
            except Exception as e:
                print(f"ERROR loading tensor '{name}' from {file_path}: {e}")
                continue
                
            # Convert to the right dtype
            tensor = tensor.astype(np.dtype(param_dtype))
            
            # Convert the weight name to Flax format
            flax_name = convert_weight_name_to_flax(name)
            
            if debug and i % 20 == 0:
                print(f"Converted '{name}' → '{flax_name}'")
            
            # Get the right partition spec for this parameter
            if mesh is not None:
                # Create a path through the partition specs
                flat_spec = flatten_dict(partition_specs)
                
                # Try to find a matching spec
                part_spec = None
                
                for spec_key, spec_value in flat_spec.items():
                    # Convert tuple key to string path for matching
                    spec_path = "/".join(spec_key)
                    
                    # Check if this spec matches our parameter
                    if spec_path == flax_name:
                        part_spec = spec_value
                        break
                    
                    # Try with regex patterns for layer indices
                    elif ".*" in spec_path:
                        # Convert to regex pattern
                        regex_pattern = spec_path.replace(".*", r"\d+")
                        import re
                        if re.match(f"^{regex_pattern}$", flax_name):
                            part_spec = spec_value
                            break
                
                if part_spec is None:
                    # Default to not sharding if we can't find a spec
                    part_spec = P(None)
                    if debug:
                        print(f"No partition spec found for {flax_name}")
                
                # Create sharded array
                with mesh:
                    # Create a NamedSharding for this parameter
                    array_sharding = jax.sharding.NamedSharding(mesh, part_spec)
                    
                    # Create sharded array
                    sharded_tensor = jax.device_put(tensor, array_sharding)
                    
                    # Add to parameters
                    flat_params[flax_name] = sharded_tensor
            else:
                # No sharding needed
                flat_params[flax_name] = jnp.array(tensor)
            
            # Mark as loaded
            loaded_params.add(name)
            
            # Report on specific milestones to give better feedback
            if i > 0 and (i % 50 == 0 or i == total_params - 1):
                elapsed = time.time() - start_time
                percent = (i + 1) / total_params * 100
                print(f"Loaded {i+1}/{total_params} parameters ({percent:.1f}%) - {elapsed:.1f}s elapsed")
                sys.stdout.flush()
        
        # File handles are automatically closed when no longer referenced
        # Do NOT attempt to explicitly close safetensors handles as they don't have a close() method
        
        # Check if we loaded all parameters
        if len(loaded_params) < len(param_names):
            missing = set(param_names) - loaded_params
            print(f"Warning: Did not load {len(missing)} parameters")
            if debug:
                print("Missing parameters:", list(missing)[:10])
        
        # Convert the flat parameter dictionary to the nested structure expected by Flax
        # This is crucial for the model to work correctly
        try:
            params = {}
            
            # Group parameters by their collection (e.g., 'model', 'lm_head')
            for key, value in flat_params.items():
                # Split the parameter path
                parts = key.split('/')
                
                # The first part is the collection name
                collection = parts[0]
                
                # Initialize the collection if needed
                if collection not in params:
                    params[collection] = {}
                
                # Skip the collection name and join the rest of the path
                param_path = '/'.join(parts[1:])
                
                # Build a dict containing the parameter path
                current = params[collection]
                parts = param_path.split('/')
                
                # Create nested dictionaries for each path component
                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the parameter value at the leaf
                current[parts[-1]] = value
            
            print(f"Successfully created nested parameter structure with collections: {list(params.keys())}")
        except Exception as e:
            print(f"Error creating nested parameter structure: {e}")
            # Fall back to using unflatten_dict
            try:
                params = unflatten_dict(flat_params, sep='/')
                print("Used unflatten_dict to create parameter structure")
            except Exception as e2:
                print(f"Error unflattening parameters: {e2}")
                # Last resort - just use the flat structure
                params = {'params': flat_params}
                print("Using flat parameter structure as fallback")
        
        total_time = time.time() - start_time
        print(f"✅ Weight loading completed in {total_time:.2f} seconds")
        print(f"Loaded {len(flat_params)} parameters")
        sys.stdout.flush()
        
        return params
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        raise

def init_model_from_weights(
    model_class,
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[jax.sharding.Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    input_ids_shape: Tuple[int, int] = (1, 16),
):
    """
    Initialize a model and load weights.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model weights
        config: Model configuration dictionary
        mesh: JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        input_ids_shape: Shape for input ids (batch_size, seq_len)
        
    Returns:
        Tuple of (model instance, loaded parameters)
    """
    print("Initializing model...")
    
    # Create the model instance
    model = model_class(
        config=config,
        mesh=mesh,
        dtype=jnp.bfloat16,
        param_dtype=param_dtype
    )
    
    # Load the weights
    print("Loading model weights from disk...")
    params = load_qwen_weights(
        model_path=model_path,
        config=config,
        mesh=mesh,
        param_dtype=param_dtype,
        debug=True
    )
    
    print("Model weights loaded successfully")
    
    return model, params

def load_safetensors_into_params(model_params, weight_map, safetensors_dir):
    """
    Load weights from safetensors files into model parameters with filtering.
    
    Args:
        model_params: Initial model parameters structure
        weight_map: Dictionary mapping parameter names to file paths
        safetensors_dir: Directory containing safetensors files
        
    Returns:
        Updated model parameters with loaded weights
    """
    import sys
    import time
    from flax.traverse_util import flatten_dict, unflatten_dict
    
    print(f"Loading {len(weight_map)} parameters from safetensors files")
    start_time = time.time()
    
    # Track file handles to avoid repeatedly opening the same file
    file_handles = {}
    
    # Try different separator formats to find what works with this model
    for sep in ['/', '.']:
        try:
            # Convert params to mutable dict for updating
            params_dict = flatten_dict(model_params, sep=sep)
            print(f"Successfully flattened parameters with separator '{sep}'")
            break
        except Exception as e:
            print(f"Failed to flatten with separator '{sep}': {e}")
    else:
        # Default to '/' if none worked
        print("Using default separator '/' for flattening")
        params_dict = flatten_dict(model_params, sep='/')
    
    # Debug: show initial param structure
    print(f"Initial model parameters structure has {len(params_dict)} entries")
    for i, key in enumerate(sorted(list(params_dict.keys()))[:5]):
        print(f"  Sample param {i}: {key}")
    
    # Create a lookup dictionary for parameter names
    param_lookup = {}
    for key in params_dict.keys():
        # Create normalized key variations for lookup
        # 1. Original key
        param_lookup[key] = key
        
        # 2. Convert / to .
        if '/' in key:
            param_lookup[key.replace('/', '.')] = key
        
        # 3. Convert . to /
        if '.' in key:
            param_lookup[key.replace('.', '/')] = key
        
        # 4. Last component only
        last_component = key.split('/')[-1] if '/' in key else key.split('.')[-1] if '.' in key else key
        if last_component not in param_lookup:  # Don't overwrite if already exists
            param_lookup[last_component] = key
    
    # Count successful parameter updates
    success_count = 0
    failed_count = 0
    
    # Process parameters
    for i, (pt_name, file_path) in enumerate(tqdm(weight_map.items(), desc="Loading weights")):
        # Log progress periodically
        if i % 50 == 0:
            print(f"Loaded {i}/{len(weight_map)} parameters ({i/len(weight_map)*100:.1f}%)")
            sys.stdout.flush()
        
        # Get the full file path
        full_path = os.path.join(safetensors_dir, os.path.basename(file_path))
        
        # Create file handle if we don't have one yet
        if full_path not in file_handles:
            try:
                file_handles[full_path] = safetensors.numpy.safe_open(full_path, framework="numpy")
            except Exception as e:
                print(f"Error opening {full_path}: {e}")
                continue
        
        # Get the file handle
        f = file_handles[full_path]
        
        try:
            # Convert the PyTorch parameter name to Flax
            flax_name = convert_weight_name_to_flax(pt_name)
            
            # Check if we have a direct match in our lookup
            if flax_name in param_lookup:
                param_path = param_lookup[flax_name]
            else:
                # Try with separator conversion
                for sep_from, sep_to in [('/', '.'), ('.', '/')]:
                    converted_name = flax_name.replace(sep_from, sep_to)
                    if converted_name in param_lookup:
                        param_path = param_lookup[converted_name]
                        print(f"Found parameter with converted path: {flax_name} -> {converted_name}")
                        break
                else:
                    # Try last component matching
                    last_component = flax_name.split('/')[-1] if '/' in flax_name else flax_name.split('.')[-1] if '.' in flax_name else flax_name
                    if last_component in param_lookup:
                        param_path = param_lookup[last_component]
                        print(f"Matched by last component: {flax_name} -> {last_component}")
                    else:
                        # Try looking for similar paths
                        similar_paths = [k for k in params_dict.keys() if last_component in k]
                        if similar_paths:
                            print(f"Parameter {flax_name} not found, but found similar paths: {similar_paths[:3]}")
                        else:
                            print(f"Parameter {flax_name} not found in model, skipping")
                        failed_count += 1
                        continue
            
            # Load the tensor
            tensor = f.get_tensor(pt_name)
            
            # Transpose weight matrices for linear layers (Flax uses different convention)
            if 'kernel' in flax_name and len(tensor.shape) == 2:
                print(f"Transposing weight matrix for {flax_name}")
                tensor = tensor.T
            
            # Validate parameter shapes
            target_shape = params_dict[param_path].shape if hasattr(params_dict[param_path], 'shape') else None
            if target_shape is not None and tensor.shape != target_shape:
                print(f"Warning: Shape mismatch for {param_path}. Expected {target_shape}, got {tensor.shape}.")
                # Try to reshape if possible
                if np.prod(tensor.shape) == np.prod(target_shape):
                    print(f"Attempting to reshape {tensor.shape} -> {target_shape}")
                    tensor = tensor.reshape(target_shape)
            
            # Update the parameter
            params_dict[param_path] = jnp.array(tensor)
            success_count += 1
            
        except Exception as e:
            print(f"Error loading parameter {pt_name}: {e}")
            failed_count += 1
    
    # No need to close file handles for safetensors - they don't have a close method
    # The handles will be garbage collected when they go out of scope
    
    # Try to unflatten using the same separator used for flatten
    try:
        updated_params = unflatten_dict(params_dict, sep=sep)
    except Exception as e:
        print(f"Error unflattening with separator '{sep}': {e}")
        # Try with default separator
        updated_params = unflatten_dict(params_dict, sep='/')
    
    print(f"✅ Weight loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Successfully loaded {success_count}/{len(weight_map)} parameters, {failed_count} failed")
    
    return updated_params

def load_qwen_weights_v2(*args, **kwargs):
    """Alias for load_qwen_weights for compatibility."""
    return load_qwen_weights(*args, **kwargs)

def print_safetensors_param_names(model_path: str) -> None:
    """Print all parameter names in the safetensors files."""
    import logging
    logging_level = logging.INFO
    logger = logging.getLogger("PARAM_CHECK")
    logger.setLevel(logging_level)
    
    try:
        index_path = os.path.join(model_path, 'model.safetensors.index.json')
        if not os.path.exists(index_path):
            logger.error(f"Safetensors index file not found at {index_path}")
            return
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        logger.info(f"Found {len(weight_map)} parameters in safetensors files")
        
        embed_params = []
        for i, (name, file_name) in enumerate(weight_map.items()):
            logger.info(f"{i+1}. {name} -> {file_name}")
            if "embed" in name:
                logger.warning(f"EMBEDDING PARAMETER: {name}")
                embed_params.append(name)
        
        # Open the first file to explore parameter values
        first_file = None
        for file_name in set(weight_map.values()):
            first_file = file_name
            break
        
        if first_file:
            full_path = os.path.join(model_path, first_file)
            logger.info(f"Opening first safetensors file: {full_path}")
            
            from safetensors import safe_open
            with safe_open(full_path, framework="numpy") as f:
                # Get first few tensors
                tensors = list(f.keys())
                logger.info(f"Tensor names in {first_file}: {tensors[:10]}...")
        
        # Check Flax name conversions for embedding parameters
        logger.info("Checking Flax name conversion for parameters containing 'embed'")
        logger.info(f"Found {len(embed_params)} parameters containing 'embed'")
        for param in embed_params:
            flax_name = convert_weight_name_to_flax(param)
            logger.info(f"PyTorch: {param} -> Flax: {flax_name}")
            
            # Check actual structure of the parameter path
            flax_path_parts = flax_name.split('/')
            prefix = ""
            for i, part in enumerate(flax_path_parts):
                prefix = prefix + "/" + part if prefix else part
                logger.info(f"  - Path component {i+1}: {prefix}")
    except Exception as e:
        logger.error(f"Error checking parameters: {e}") 