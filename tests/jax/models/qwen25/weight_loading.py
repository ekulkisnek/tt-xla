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
        # Allow some missing keys for flexibility
        flax_state_dict = load_pytorch_checkpoint_in_flax_state_dict(
            model, checkpoint_files, is_sharded=is_sharded, allow_missing_keys=True
        )
        logger.info(f"Successfully converted PyTorch weights to Flax format")
    except Exception as e:
        logger.error(f"Error converting PyTorch weights to Flax format: {e}")
        raise
    
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