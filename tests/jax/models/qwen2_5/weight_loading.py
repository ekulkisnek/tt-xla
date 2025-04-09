# Copyright 2024 TensorTrace Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight loading utilities for Qwen2.5"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec
from safetensors.flax import load_file as safe_load_file
from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from transformers.utils import logging

from .configuration_qwen2_5 import Qwen25Config
from .modeling_flax_qwen2_5 import FlaxQwen25ForCausalLM, FlaxQwen25Model
from .tensor_parallel import create_device_mesh, get_partition_specs

logger = logging.get_logger(__name__)


def get_checkpoint_files(checkpoint_dir: str) -> List[str]:
    """
    Get a list of all checkpoint files in the given directory.
    Supports safetensors and PyTorch formats.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        List of checkpoint filenames
    """
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if safetensors_files:
        index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            logger.info(f"Found safetensors index file: {index_file}")
            import json
            with open(index_file, "r") as f:
                index = json.load(f)
                if "weight_map" in index:
                    files = sorted(list(set(index["weight_map"].values())))
                    return [os.path.join(checkpoint_dir, f) for f in files]
        logger.info(f"Found {len(safetensors_files)} safetensors files")
        return safetensors_files
    
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


def convert_qwen25_checkpoint(
    checkpoint_dir: str,
    config: Optional[Qwen25Config] = None,
    dtype: jnp.dtype = jnp.float16,
    with_lm_head: bool = True,
    mesh: Optional[Mesh] = None,
    partition_rules: Optional[Dict] = None,
) -> Union[FlaxQwen25Model, FlaxQwen25ForCausalLM]:
    """
    Load a Qwen2.5 checkpoint from HuggingFace format into a Flax model with tensor parallelism.
    Uses memory-efficient loading by processing weights in chunks.
    """
    if config is None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        if not isinstance(config, Qwen25Config):
            logger.warning(f"Converting config from {type(config).__name__} to Qwen25Config")
            config_dict = config.to_dict()
            config = Qwen25Config(**config_dict)
    
    # Initialize model with proper tensor parallelism
    logger.info(f"Initializing {'FlaxQwen25ForCausalLM' if with_lm_head else 'FlaxQwen25Model'} with tensor parallelism")
    model_class = FlaxQwen25ForCausalLM if with_lm_head else FlaxQwen25Model
    
    with mesh:
        model = model_class.init_for_sharding(
            config=config,
            dtype=dtype,
            mesh=mesh,
            partition_rules=partition_rules,
        )
    
    # Get checkpoint files
    checkpoint_files = get_checkpoint_files(checkpoint_dir)
    
    # Load state dict from files in chunks
    logger.info(f"Converting weights from PyTorch to Flax")
    is_sharded = len(checkpoint_files) > 1
    
    # Process each file separately to save memory
    for checkpoint_file in checkpoint_files:
        logger.info(f"Processing {checkpoint_file}")
        if checkpoint_file.endswith('.safetensors'):
            state_dict = safe_load_file(checkpoint_file)
        else:
            import torch
            state_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        
        # Convert weights to Flax format
        flax_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            if value.dtype != dtype:
                value = value.astype(dtype)
            flax_state_dict[key] = value
        
        # Update model parameters
        model.params.update(flax_state_dict)
        
        # Clear memory
        del state_dict
        del flax_state_dict
        import gc
        gc.collect()
    
    # Create partition specs for state dict
    param_specs = get_partition_specs(model.params, partition_rules)
    
    # Apply tensor parallelism to the loaded weights
    with mesh:
        def apply_sharding(params, specs):
            return jax.tree_util.tree_map(
                lambda p, s: jax.lax.with_sharding_constraint(p, s) if p is not None else None,
                params,
                specs,
                is_leaf=lambda x: x is None
            )
        
        model.params = apply_sharding(model.params, param_specs)
    
    return model 