"""JAX sharding utilities for Qwen2.5"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import linen as nn

from .configuration_qwen2_5 import Qwen25Config


def with_sharding_constraint(x, mesh, partition_spec):
    """Apply sharding constraint to tensor."""
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, partition_spec))


class FlaxQwen25WithSharding(nn.Module):
    """Wrapper class to add sharding to Flax Qwen2.5 model."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.bfloat16
    mesh: Optional[Mesh] = None
    partition_specs: Optional[Dict] = None
    
    def setup(self):
        from .modeling_flax_qwen2_5 import FlaxQwen25ForCausalLM
        self.model = FlaxQwen25ForCausalLM(config=self.config, dtype=self.dtype)
    
    def __call__(self, input_ids, attention_mask=None, position_ids=None, deterministic=True, return_dict=True):
        if self.mesh is None or self.partition_specs is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                return_dict=return_dict
            )
        
        # Apply sharding to inputs
        sharded_inputs = {
            'input_ids': with_sharding_constraint(input_ids, self.mesh, PartitionSpec(None)),
            'attention_mask': with_sharding_constraint(attention_mask, self.mesh, PartitionSpec(None)) if attention_mask is not None else None,
            'position_ids': with_sharding_constraint(position_ids, self.mesh, PartitionSpec(None)) if position_ids is not None else None,
        }
        
        # Run model with sharded inputs
        outputs = self.model(
            **sharded_inputs,
            deterministic=deterministic,
            return_dict=return_dict
        )
        
        # Apply output sharding constraints
        if isinstance(outputs, dict):
            sharded_outputs = {}
            for k, v in outputs.items():
                if isinstance(v, jnp.ndarray):
                    sharded_outputs[k] = with_sharding_constraint(v, self.mesh, PartitionSpec(None))
                else:
                    sharded_outputs[k] = v
            return sharded_outputs
        else:
            return outputs 