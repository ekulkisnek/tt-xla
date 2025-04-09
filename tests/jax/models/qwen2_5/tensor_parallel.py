# Copyright 2024 TensorTrace Inc. and HuggingFace Inc. team. All rights reserved.
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
"""JAX tensor parallelism utilities for Qwen2.5"""

import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import numpy as np

from .sharding import with_sharding_constraint


def create_device_mesh(mesh_shape: Tuple[int, int]) -> np.ndarray:
    """Create a device mesh for Tenstorrent hardware."""
    # Get available Tenstorrent devices
    devices = jax.devices()
    tt_devices = [d for d in devices if d.platform == 'tt']
    
    if not tt_devices:
        raise RuntimeError("No Tenstorrent devices found. Make sure JAX is configured to use Tenstorrent hardware.")
    
    num_devices = len(tt_devices)
    requested_devices = np.prod(mesh_shape)
    
    if num_devices < requested_devices:
        raise ValueError(
            f"Not enough Tenstorrent devices. Requested {requested_devices} devices but only {num_devices} available."
        )
    
    # Use the first N devices where N = requested_devices
    devices_to_use = tt_devices[:requested_devices]
    
    # Reshape devices into mesh
    device_mesh = np.array(devices_to_use).reshape(mesh_shape)
    return device_mesh


def get_partition_specs(config):
    """Get partition specs for model parameters."""
    # Define base partition specs
    base_specs = {
        'kernel': PartitionSpec('mp', None),
        'bias': PartitionSpec('mp'),
        'embedding': PartitionSpec('mp', None),
        'norm': PartitionSpec(None),
        'lm_head': PartitionSpec('mp', None),
    }
    
    # Create partition specs for each layer
    layer_specs = {}
    for i in range(config.num_hidden_layers):
        layer_specs[f'layers_{i}'] = {
            'attention': {
                'q_proj': base_specs,
                'k_proj': base_specs,
                'v_proj': base_specs,
                'o_proj': base_specs,
            },
            'mlp': {
                'gate_proj': base_specs,
                'up_proj': base_specs,
                'down_proj': base_specs,
            },
            'input_layernorm': base_specs,
            'post_attention_layernorm': base_specs,
        }
    
    # Combine all specs
    partition_specs = {
        'model': {
            'embed_tokens': base_specs,
            'norm': base_specs,
            'layers': layer_specs,
            'lm_head': base_specs,
        }
    }
    
    return partition_specs 