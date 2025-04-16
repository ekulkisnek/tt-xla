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
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec as P
from safetensors.flax import load_file as safe_load_file
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from transformers.utils import logging as transformers_logging

from tensor_parallel import get_partition_specs, create_device_mesh

# Setup logging
logger = logging.getLogger(__name__)
transformers_logger = transformers_logging.get_logger("transformers")
transformers_logger.setLevel(logging.INFO)

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

def adapt_parameter_names(params: Dict) -> Dict:
    """
    Adapt parameter names between HuggingFace's conventions and our model's expectations.
    
    Args:
        params: The loaded parameters
        
    Returns:
        Adapted parameters
    """
    # Flatten the dictionary for easier manipulation
    flat_params = flatten_dict(params)
    new_params = {}
    
    # Fix parameter names if needed
    for path, param in flat_params.items():
        new_path = path
        
        # Fix embedding parameter names to use "weight" instead of "embedding"
        if 'embedding' in path[-1] and 'embed_tokens' in '.'.join([str(p) for p in path[:-1]]):
            new_path = path[:-1] + ('weight',)
        
        new_params[new_path] = param
    
    # Unflatten and return
    return unflatten_dict(new_params)

def convert_tensor_dtype(tensor, target_dtype=None):
    """
    Convert tensor to the right format and dtype for JAX.
    
    Args:
        tensor: Input tensor which can be PyTorch, NumPy, or already JAX
        target_dtype: Target dtype, if None uses the tensor's original dtype
        
    Returns:
        NumPy array with the right dtype
    """
    import torch
    import numpy as np
    
    # Handle PyTorch tensors
    if hasattr(tensor, 'detach') and hasattr(tensor, 'cpu') and hasattr(tensor, 'numpy'):
        # It's a PyTorch tensor
        if str(tensor.dtype) == 'torch.bfloat16':
            # Convert bfloat16 to float32 first (PyTorch's bfloat16 needs special handling)
            tensor = tensor.to(torch.float32)
        
        # Convert to NumPy
        tensor = tensor.detach().cpu().numpy()
    
    # Handle other torch dtypes that might still be in the tensor's dtype attribute
    if hasattr(tensor, 'dtype') and isinstance(tensor.dtype, str) and tensor.dtype.startswith('torch.'):
        # If it's still a string representation of a torch dtype, convert to numpy
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float16': np.float16,
            'torch.bfloat16': np.float32,  # No bfloat16 in numpy, use float32
            'torch.int32': np.int32,
            'torch.int64': np.int64,
            'torch.bool': np.bool_,
        }
        numpy_dtype = dtype_map.get(tensor.dtype, np.float32)
        tensor = np.array(tensor, dtype=numpy_dtype)
    
    # Apply target dtype if specified
    if target_dtype is not None:
        tensor = tensor.astype(target_dtype)
    
    return tensor

def load_safetensors_file(filepath):
    """
    Load a safetensors file directly using the safetensors library.
    
    Args:
        filepath: Path to the safetensors file
        
    Returns:
        Dictionary of tensors
    """
    try:
        import safetensors.torch
        # First try with torch version which is faster
        tensors = safetensors.torch.load_file(filepath)
        
        # Convert any torch tensors to numpy
        converted = {}
        for k, v in tensors.items():
            converted[k] = convert_tensor_dtype(v)
        return converted
    except ImportError:
        # Fall back to numpy version if torch not available
        import safetensors.numpy
        return safetensors.numpy.load_file(filepath)

def load_qwen_weights(
    model_path: str,
    model: Any,
    config: Dict[str, Any],
    mesh: Optional[Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    debug: bool = False,
) -> Dict:
    """
    Load weights from checkpoint files and convert to Flax format.
    
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
    
    try:
        merged_state_dict = {}
        
        # Load the weights from each file
        for shard_file in checkpoint_files:
            logger.info(f"Loading file {shard_file}")
            
            if shard_file.endswith('.safetensors'):
                # Load safetensors files directly
                shard_dict = load_safetensors_file(shard_file)
                logger.info(f"Loaded safetensors file with {len(shard_dict)} parameters")
            else:
                # For PyTorch bin files
                import torch
                shard_dict = torch.load(shard_file, map_location="cpu", weights_only=False)
                logger.info(f"Loaded PyTorch file with {len(shard_dict)} parameters")
                
                # Convert PyTorch tensors to numpy arrays
                converted_dict = {}
                for k, v in shard_dict.items():
                    converted_dict[k] = convert_tensor_dtype(v)
                shard_dict = converted_dict
            
            # Add shard parameters to the merged state dict
            merged_state_dict.update(shard_dict)
        
        logger.info(f"Merged state dict contains {len(merged_state_dict)} total parameters")
        
        # Generate a random state dict with the same structure as the model
        # for parameter name mapping
        flax_model_params = model.params if hasattr(model, 'params') else {}
        random_state_dict = flatten_dict(flax_model_params)
        
        # Convert weights from PyTorch format to Flax
        flax_state_dict = {}
        
        # Apply parameter conversion and ensure the right shapes/dtypes
        for pt_key, pt_tensor in merged_state_dict.items():
            # Convert '.' to nested structure
            pt_tuple_key = tuple(pt_key.split('.'))
            
            # Linear/Dense layers need to have their weights transposed
            if 'weight' in pt_key and pt_tensor.ndim == 2:
                if any(name in pt_key for name in ['query', 'key', 'value', 'dense', 'proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head']):
                    pt_tensor = pt_tensor.T
                    
            # For RMSNorm, rename 'weight' to 'scale'
            if 'norm' in pt_key and 'weight' in pt_key:
                pt_tuple_key = tuple(k if k != 'weight' else 'scale' for k in pt_tuple_key)
                
            # Fix other parameter names based on Flax naming conventions
            if pt_tuple_key[-1] == 'weight' and pt_tuple_key[0] != 'lm_head':
                pt_tuple_key = pt_tuple_key[:-1] + ('kernel',)
                
            # Add 'params' prefix for Flax model
            if not pt_tuple_key[0] == 'params':
                pt_tuple_key = ('params',) + pt_tuple_key
                
            # Convert to target parameter dtype if specified
            if param_dtype is not None and pt_tensor.dtype != np.bool_:
                pt_tensor = pt_tensor.astype(param_dtype)
                
            # Add to flax state dict
            flax_state_dict[pt_tuple_key] = pt_tensor
            
        # Unflatten the state dict to nest structure
        flax_state_dict = unflatten_dict(flax_state_dict)
        
        # Adapt parameter names for our model's expectations
        flax_state_dict = adapt_parameter_names(flax_state_dict)
        
        logger.info(f"Successfully converted weights to Flax format")
    except Exception as e:
        logger.error(f"Error converting weights to Flax format: {e}")
        # Make a more informative error
        import traceback
        error_details = traceback.format_exc()
        raise ValueError(f"Failed to load weights from {model_path}: {e}\n\nDetails:\n{error_details}")
    
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