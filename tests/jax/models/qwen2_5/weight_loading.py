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
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
from tqdm import tqdm
import safetensors.numpy

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
        
        # Initialize the parameter dictionary
        params = {}
        
        # Map of tensors to load directly from files
        file_handles = {}
        
        # Track which parameters we've loaded
        loaded_params = set()
        
        # Collect unique files
        all_files = set(param_file_map.values())
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
                
            file_path = param_file_map[name]
            
            # Create file handle if we don't have one yet
            if file_path not in file_handles:
                try:
                    # Report on file opening
                    file_name = os.path.basename(file_path)
                    print(f"Opening file: {file_name}")
                    sys.stdout.flush()
                    last_file_time = time.time()
                    
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
                    params[flax_name] = sharded_tensor
            else:
                # No sharding needed
                params[flax_name] = jnp.array(tensor)
            
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
        
        total_time = time.time() - start_time
        print(f"✅ Weight loading completed in {total_time:.2f} seconds")
        print(f"Loaded {len(params)} parameters")
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

def load_qwen_weights_v2(*args, **kwargs):
    """Alias for load_qwen_weights for compatibility."""
    return load_qwen_weights(*args, **kwargs) 