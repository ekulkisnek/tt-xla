# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for Qwen2.5 models.
"""

import os
import glob
import logging
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec as P
from safetensors.flax import load_file as safe_load_file
from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from transformers.utils import logging as transformers_logging

from tensor_parallel import get_partition_specs, create_device_mesh

# Setup logging
logger = logging.getLogger(__name__)
transformers_logger = transformers_logging.get_logger("transformers")
transformers_logger.setLevel(logging.INFO)

# Patch PyTorch's load function to handle the weights_only parameter in PyTorch 2.6+
def patch_torch_load():
    """
    Apply a patch to torch.load to handle PyTorch 2.6's weights_only parameter change.
    """
    try:
        import torch
        original_torch_load = torch.load
        
        # Create patched version that sets weights_only=False by default
        def patched_torch_load(f, *args, **kwargs):
            if 'weights_only' not in kwargs:
                logger.info("Setting weights_only=False for PyTorch 2.6+ compatibility")
                kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)
        
        # Replace the original function
        torch.load = patched_torch_load
        logger.info("✅ Applied patch for PyTorch 2.6+ weights_only parameter")
        return True
    except ImportError:
        logger.warning("⚠️ Could not patch torch.load (torch not imported yet)")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Could not patch torch.load: {e}")
        return False

# Patch transformers' load_pytorch_checkpoint function to handle weights_only issue
def patch_transformers_load_function():
    """
    Apply a patch to transformers' load_pytorch_checkpoint_in_flax_state_dict function
    to handle PyTorch 2.6's weights_only parameter change.
    """
    try:
        from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict as original_load_fn
        
        def patched_load_pytorch_checkpoint_in_flax_state_dict(
            flax_model, 
            pytorch_checkpoint_path, 
            is_sharded=False, 
            allow_missing_keys=False
        ):
            """Patched version that explicitly sets weights_only=False in the PyTorch loading code"""
            try:
                return original_load_fn(
                    flax_model, 
                    pytorch_checkpoint_path, 
                    is_sharded=is_sharded, 
                    allow_missing_keys=allow_missing_keys
                )
            except Exception as e:
                # If the first attempt fails with a weights_only error, try to patch
                # both torch.load and also modify the inner function behavior
                if "weights_only" in str(e):
                    logger.info("First loading attempt failed with weights_only error, applying patch...")
                    # Patch torch.load
                    patch_torch_load()
                    
                    # Try again
                    return original_load_fn(
                        flax_model, 
                        pytorch_checkpoint_path, 
                        is_sharded=is_sharded, 
                        allow_missing_keys=allow_missing_keys
                    )
                else:
                    # Re-raise if it's a different error
                    raise
        
        # Apply the patch
        import transformers.modeling_flax_pytorch_utils
        transformers.modeling_flax_pytorch_utils.load_pytorch_checkpoint_in_flax_state_dict = patched_load_pytorch_checkpoint_in_flax_state_dict
        logger.info("✅ Applied patch for transformers' load_pytorch_checkpoint_in_flax_state_dict")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not patch transformers' load function: {e}")
        return False

# Apply patches when this module is imported
_ = patch_torch_load()
_ = patch_transformers_load_function()

def get_checkpoint_files(checkpoint_dir: str) -> List[str]:
    """
    Get a list of all checkpoint files in the given directory.
    Supports safetensors and PyTorch formats.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        List of checkpoint filenames
    """
    # First check for the safetensors index file
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        import json
        with open(index_file, "r") as f:
            index = json.load(f)
            if "weight_map" in index:
                files = sorted(list(set(index["weight_map"].values())))
                return [os.path.join(checkpoint_dir, f) for f in files]
    
    # Check for safetensors files directly
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensors_files:
        logger.info(f"Found {len(safetensors_files)} safetensors files")
        return safetensors_files
    
    # Check for PyTorch files
    pytorch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.bin")))
    if pytorch_files:
        logger.info(f"Found {len(pytorch_files)} PyTorch checkpoint files")
        return pytorch_files
    
    # Try the pytorch_model.bin file as a fallback
    pytorch_model_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_model_bin):
        logger.info(f"Found single PyTorch model file: {pytorch_model_bin}")
        return [pytorch_model_bin]
    
    raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

def load_qwen_weights(
    model_path: str,
    model: Any,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
) -> Dict:
    """
    Load weights from PyTorch checkpoint files and convert to Flax format using HuggingFace utilities.
    
    Args:
        model_path: Path to model checkpoint files
        model: The initialized Flax model
        config: Model configuration dictionary
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        debug: Whether to print debug information
        
    Returns:
        Dictionary of model parameters
    """
    start_time = time.time()
    
    logger.info(f"Loading weights from {model_path}")
    
    # Get all checkpoint files
    checkpoint_files = get_checkpoint_files(model_path)
    
    # Check if we need to shard the weights
    is_sharded = len(checkpoint_files) > 1
    logger.info(f"Found {'sharded' if is_sharded else 'single'} checkpoint with {len(checkpoint_files)} file(s)")
    
    # Use HuggingFace's utility to convert PyTorch weights to Flax format
    logger.info("Converting PyTorch weights to Flax format using HuggingFace utilities")
    try:
        # Make sure our patches are applied
        patch_torch_load()
        patch_transformers_load_function()
        
        # Try direct loading with safetensors if available
        if checkpoint_files[0].endswith('.safetensors'):
            try:
                # Try to use safetensors directly, which doesn't have the weights_only issue
                logger.info("Attempting to load with safetensors...")
                if is_sharded:
                    # Load each file and merge
                    all_params = {}
                    for file in checkpoint_files:
                        logger.info(f"Loading safetensors file: {file}")
                        params = safe_load_file(file)
                        all_params.update(params)
                    # Convert to Flax format
                    flax_state_dict = model.params_from_state_dict(unflatten_dict(all_params, sep="."))
                else:
                    # Single file
                    params = safe_load_file(checkpoint_files[0])
                    flax_state_dict = model.params_from_state_dict(unflatten_dict(params, sep="."))
                logger.info("Successfully loaded weights using safetensors")
            except Exception as e:
                logger.warning(f"Failed to load with safetensors: {e}")
                # Fall back to HuggingFace utility
                flax_state_dict = load_pytorch_checkpoint_in_flax_state_dict(
                    model, checkpoint_files, is_sharded=is_sharded, allow_missing_keys=True
                )
        else:
            # Allow some missing keys for flexibility
            flax_state_dict = load_pytorch_checkpoint_in_flax_state_dict(
                model, checkpoint_files, is_sharded=is_sharded, allow_missing_keys=True
            )
        
        logger.info(f"Successfully converted PyTorch weights to Flax format")
    except Exception as e:
        logger.error(f"Error converting PyTorch weights to Flax format: {e}")
        raise ValueError(f"Failed to load weights from {model_path}: {e}")
    
    # Apply tensor parallelism if mesh is provided
    if mesh is not None:
        logger.info(f"Applying tensor parallelism with mesh shape {mesh.devices.shape}")
        
        # Create partition specs for the model parameters
        partition_specs = get_partition_specs(config)
        
        # Flatten the partition specs for easier lookup
        flat_partition_specs = flatten_dict(partition_specs)
        
        # Flatten the state dict for easier modification
        flat_state_dict = flatten_dict(flax_state_dict)
        
        # Apply sharding to each parameter
        with mesh:
            for key, param in flat_state_dict.items():
                # Find the matching partition spec
                spec_key = key
                
                # Try to find the partition spec for this parameter
                if spec_key in flat_partition_specs:
                    spec = flat_partition_specs[spec_key]
                    
                    # Apply the partition spec
                    if spec is not None:
                        # Create a sharded array
                        flat_state_dict[key] = jax.device_put(param, jax.sharding.NamedSharding(mesh, spec))
                        
                        if debug:
                            logger.info(f"Applied partition spec {spec} to {key}")
                    else:
                        # No sharding needed for this parameter
                        flat_state_dict[key] = jax.device_put(param)
                else:
                    # No partition spec found, don't shard
                    flat_state_dict[key] = jax.device_put(param)
                    
                    if debug:
                        logger.info(f"No partition spec found for {key}")
        
        # Unflatten the state dict
        flax_state_dict = unflatten_dict(flat_state_dict)
    
    load_time = time.time() - start_time
    logger.info(f"Weight loading completed in {load_time:.2f} seconds")
    
    return flax_state_dict

def load_safetensors_weights(
    model_path: str,
    model: Any,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Dict:
    """
    Load weights directly from safetensors files.
    
    Args:
        model_path: Path to model checkpoint files
        model: The initialized Flax model
        config: Model configuration dictionary
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        
    Returns:
        Dictionary of model parameters
    """
    # Find the safetensors file
    if os.path.isdir(model_path):
        safetensors_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(safetensors_file):
            # Try to find any safetensors file
            safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
            if safetensors_files:
                safetensors_file = safetensors_files[0]
            else:
                raise FileNotFoundError(f"No safetensors file found in {model_path}")
    else:
        safetensors_file = model_path
    
    logger.info(f"Loading safetensors weights from {safetensors_file}")
    
    # Load the weights directly
    params = safe_load_file(safetensors_file)
    
    # Unflatten the dictionary
    params = unflatten_dict(params, sep=".")
    
    # Apply tensor parallelism if mesh is provided
    if mesh is not None:
        # This would need additional logic to apply the sharding
        # Similar to load_qwen_weights
        logger.warning("Tensor parallelism for direct safetensors loading is not fully implemented")
    
    return params

def init_model_from_weights(
    model_class,
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
):
    """
    Initialize a model and load weights.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model weights
        config: Model configuration dictionary
        mesh: JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        debug: Whether to print debug information
        
    Returns:
        Tuple of (model instance, loaded parameters)
    """
    # Initialize model
    model = model_class(
        config=config,
        mesh=mesh,
        dtype=jnp.bfloat16,
        param_dtype=param_dtype,
    )
    
    # Load weights
    params = load_qwen_weights(
        model_path=model_path,
        model=model,
        config=config,
        mesh=mesh,
        param_dtype=param_dtype,
        debug=debug
    )
    
    return model, params

def load_partial_qwen_weights(model_path, target_params, config, mesh, num_layers=None, logger=None):
    """
    Load weights from a full-sized Qwen model into a reduced-size model configuration.
    This function selectively copies weights to match the smaller architecture.
    
    Args:
        model_path: Path to the full model weights
        target_params: Target parameter structure (from model.init())
        config: Reduced size configuration
        mesh: Device mesh
        num_layers: Number of layers to load
        logger: Logger for status messages
    
    Returns:
        Dictionary with loaded parameters matching the reduced structure
    """
    import os
    import json
    import numpy as np
    from safetensors import safe_open
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    
    # Simple logging function if no logger provided
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log_info(f"Loading partial weights from {model_path} for reduced size model")
    
    # Create a copy of target_params to modify
    new_params = jax.tree_util.tree_map(lambda x: x, target_params)
    
    # Load original model config to get original dimensions
    original_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(original_config_path):
        with open(original_config_path, "r") as f:
            original_config = json.load(f)
        log_info(f"Original model: hidden_size={original_config.get('hidden_size')}, layers={original_config.get('num_hidden_layers')}")
    else:
        log_info(f"No config.json found at {model_path}, assuming standard Qwen2 7B architecture")
        original_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 14336
        }
    
    # Get list of weight files (safetensors format)
    weight_files = []
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            weight_files.append(os.path.join(model_path, file))
    
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    
    log_info(f"Found {len(weight_files)} weight files")
    
    # Target sizes from reduced config
    target_hidden_size = config["hidden_size"]
    target_layers = config["num_hidden_layers"]
    target_heads = config["num_attention_heads"]
    target_kv_heads = config["num_key_value_heads"]
    target_intermediate = config["intermediate_size"]
    
    # Original sizes
    orig_hidden_size = original_config.get("hidden_size", 4096)
    orig_layers = original_config.get("num_hidden_layers", 32) 
    orig_heads = original_config.get("num_attention_heads", 32)
    orig_kv_heads = original_config.get("num_key_value_heads", 8)
    orig_intermediate = original_config.get("intermediate_size", 14336)
    
    # Scaling factors
    hidden_factor = target_hidden_size / orig_hidden_size
    intermediate_factor = target_intermediate / orig_intermediate
    
    # Load weights from each file
    for file_path in weight_files:
        log_info(f"Processing {os.path.basename(file_path)}...")
        
        with safe_open(file_path, framework="numpy") as f:
            weight_keys = f.keys()
            
            for key in weight_keys:
                # Extract layer number for layer-specific weights
                layer_match = None
                if ".layers." in key:
                    parts = key.split(".layers.")
                    if len(parts) > 1:
                        layer_parts = parts[1].split(".")
                        try:
                            layer_num = int(layer_parts[0])
                            # Skip layers beyond our target layer count
                            if layer_num >= target_layers:
                                continue
                        except ValueError:
                            pass
                
                # Try to find the corresponding key in our target params
                tensor = f.get_tensor(key)
                param_key = key
                
                # Convert PyTorch key format to JAX format if needed
                # This is a simplified example and may need adjustment for your specific model
                param_key = param_key.replace("model.layers", "transformer.h")
                param_key = param_key.replace("self_attn", "attn")
                param_key = param_key.replace("q_proj", "q")
                param_key = param_key.replace("k_proj", "k")
                param_key = param_key.replace("v_proj", "v")
                param_key = param_key.replace("o_proj", "o")
                param_key = param_key.replace("mlp.gate_proj", "mlp.w1")
                param_key = param_key.replace("mlp.up_proj", "mlp.w2")
                param_key = param_key.replace("mlp.down_proj", "mlp.w3")
                param_key = param_key.replace("input_layernorm", "ln_1")
                param_key = param_key.replace("post_attention_layernorm", "ln_2")
                
                # Handle embedding conversion
                if "embed_tokens.weight" in param_key:
                    param_key = "transformer.embed_tokens.embedding"
                
                # Handle different paths in the params tree
                param_path = param_key.split(".")
                
                # Try to find the tensor location in our parameter tree
                current = new_params
                found = True
                
                for i, part in enumerate(param_path):
                    if part in current:
                        if i == len(param_path) - 1:  # Last part
                            # Found the target - now resize if needed
                            target_shape = current[part].shape
                            if tensor.shape != target_shape:
                                log_info(f"Resizing {param_key}: {tensor.shape} -> {target_shape}")
                                
                                # Handle different weight types
                                if "embed_tokens" in param_key or "wte" in param_key:
                                    # For embeddings, keep vocab dim and resize hidden dim
                                    if len(tensor.shape) == 2 and len(target_shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "q" in param_path[-1] or "k" in param_path[-1] or "v" in param_path[-1]:
                                    # For attention weights
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "attn.o" in param_key or "attn.c_proj" in param_key:
                                    # For attention output projection
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                elif "mlp" in param_key:
                                    # For MLP weights
                                    if len(tensor.shape) == 2:
                                        tensor = tensor[:target_shape[0], :target_shape[1]]
                                
                                else:
                                    # General case - try to match dimensions if possible
                                    try:
                                        # Simple truncation for each dimension
                                        slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
                                        tensor_resized = np.zeros(target_shape, dtype=tensor.dtype)
                                        tensor_view = tensor[slices]
                                        target_slices = tuple(slice(0, s.stop) for s in slices)
                                        tensor_resized[target_slices] = tensor_view
                                        tensor = tensor_resized
                                    except Exception as e:
                                        log_info(f"Error resizing {param_key}: {e}")
                                        continue
                            
                            # Convert to the right dtype
                            try:
                                tensor = jnp.array(tensor, dtype=current[part].dtype)
                                
                                # Apply the correct partitioning spec
                                if hasattr(current[part], "sharding"):
                                    sharding = current[part].sharding
                                    tensor = jax.device_put(tensor, sharding)
                                
                                # Update the parameter
                                current[part] = tensor
                                log_info(f"Loaded {param_key}")
                            except Exception as e:
                                log_info(f"Error setting {param_key}: {e}")
                        else:
                            current = current[part]
                    else:
                        found = False
                        break
                
                if not found:
                    log_info(f"Key {param_key} not found in target params")
    
    log_info("Partial weight loading complete")
    return new_params 