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
from flax import nnx
import numpy as np


def create_device_mesh(mesh_shape: Tuple[int, ...] = None, axis_names: Tuple[str, ...] = None) -> Mesh:
    """
    Create a device mesh for tensor parallelism.
    
    Args:
        mesh_shape: Shape of the device mesh. If None, will use all available devices.
        axis_names: Names of the mesh axes. Default is ('data', 'model').
    
    Returns:
        jax.sharding.Mesh: The device mesh.
    """
    if mesh_shape is None:
        # Get number of devices and create a simple 1D mesh
        num_devices = jax.device_count()
        mesh_shape = (1, num_devices)  # (data, model)
    
    if axis_names is None:
        axis_names = ('data', 'model')
    
    devices = jax.devices()
    if len(devices) != mesh_shape[0] * mesh_shape[1]:
        # If meshape doesn't match device count, adjust it
        mesh_shape = (1, len(devices))
    
    # Create a list of devices and reshape it to the mesh shape
    device_mesh = np.array(devices).reshape(mesh_shape)
    return Mesh(device_mesh, axis_names)


def with_sharding_constraint(x, partitioning):
    """Apply sharding constraint to a tensor."""
    return jax.lax.with_sharding_constraint(x, partitioning)


def get_partition_specs(state_dict, partition_rules):
    """
    Convert a state dict to partition specs based on partition rules.
    
    Args:
        state_dict: The state dict to convert.
        partition_rules: A dictionary of patterns to partition specs.
    
    Returns:
        A dictionary of partition specs.
    """
    import re
    from flax.traverse_util import flatten_dict, unflatten_dict
    
    flat_params = flatten_dict(state_dict)
    flat_partitioning = {}
    
    for path, value in flat_params.items():
        path_str = ".".join(path)
        found = False
        
        for pattern, rule in partition_rules.items():
            # Convert glob pattern to regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", r".*")
            if re.match(regex_pattern, path_str):
                if rule == "colwise":
                    flat_partitioning[path] = PartitionSpec(None, "model")
                elif rule == "rowwise":
                    flat_partitioning[path] = PartitionSpec("model", None)
                elif rule == "replicated":
                    flat_partitioning[path] = PartitionSpec()
                else:
                    flat_partitioning[path] = rule
                found = True
                break
        
        if not found:
            # Default to replicated
            flat_partitioning[path] = PartitionSpec()
    
    return unflatten_dict(flat_partitioning)


class FlaxQwen25WithSharding:
    """
    Mixin class to add sharding functionality to Flax Qwen2.5 models.
    """
    
    @classmethod
    def init_for_sharding(
        cls,
        config,
        input_shape=(1, 1),
        seed=0,
        dtype=jnp.float32,
        mesh=None,
        partition_rules=None,
    ):
        """
        Initialize the model with sharding constraints.
        
        Args:
            config: Model configuration
            input_shape: Shape of input tensor
            seed: Random seed
            dtype: Model dtype
            mesh: Device mesh for tensor parallelism
            partition_rules: Rules for tensor partitioning
            
        Returns:
            Initialized model with sharded parameters
        """
        # Create default mesh if not provided
        if mesh is None:
            mesh = create_device_mesh()
        
        # Use default partitioning rules if not provided
        if partition_rules is None and hasattr(config, "base_model_tp_plan"):
            partition_rules = config.base_model_tp_plan
        elif partition_rules is None:
            # Default rules for transformer models
            partition_rules = {
                "self_attn.q_proj": "colwise",
                "self_attn.k_proj": "colwise",
                "self_attn.v_proj": "colwise",
                "self_attn.o_proj": "rowwise",
                "mlp.gate_proj": "colwise",
                "mlp.up_proj": "colwise",
                "mlp.down_proj": "rowwise",
            }
        
        # Initialize the model with sharding constraints
        with mesh:
            model = cls(config, input_shape=input_shape, seed=seed, dtype=dtype)
            
            # Create partition specs for state dict
            param_specs = get_partition_specs(model.params, partition_rules)
            
            # Apply sharding constraints
            def apply_sharding(params, specs):
                return jax.tree_util.tree_map(
                    lambda p, s: with_sharding_constraint(p, s) if p is not None else None,
                    params,
                    specs,
                    is_leaf=lambda x: x is None
                )
            
            model.params = apply_sharding(model.params, param_specs)
            
        return model 