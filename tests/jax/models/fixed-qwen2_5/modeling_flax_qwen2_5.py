# coding=utf-8
# Copyright 2024 Tenstorrent AI ULC. All rights reserved.
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

"""Flax Qwen2.5 model."""

from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.sharding import Mesh, PartitionSpec as P

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import logging

from .configuration_qwen2_5 import Qwen25Config

logger = logging.get_logger(__name__)

def create_sinusoidal_positions(num_pos: int, dim: int, theta: float = 10000.0) -> jnp.ndarray:
    """Create sinusoidal position embeddings."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(num_pos)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
    return emb[None, :, None, :]  # [1, seq_len, 1, dim]

class FlaxQwen25RMSNorm(nn.Module):
    """Qwen2.5 RMSNorm module."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.weight = self.param(
            "weight",
            jax.nn.initializers.ones,
            (self.config.hidden_size,),
            self.dtype
        )
        self.variance_epsilon = self.config.rms_norm_eps

    def __call__(self, hidden_states):
        variance = jnp.power(hidden_states, 2).mean(-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class FlaxQwen25Attention(nn.Module):
    """Multi-head attention with tensor parallelism support."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rotary_dim = self.config.rotary_dim

        self.q_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            name="q_proj",
        )
        self.k_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            name="k_proj",
        )
        self.v_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            name="v_proj",
        )
        self.o_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            name="o_proj",
        )

        self.rotary_emb = FlaxQwen25RotaryEmbedding(
            self.rotary_dim,
            dtype=self.dtype,
            name="rotary_emb",
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = jax.lax.with_sharding_constraint(
            query_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )
        key_states = jax.lax.with_sharding_constraint(
            key_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )
        value_states = jax.lax.with_sharding_constraint(
            value_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        query_states = query_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )

        query_states = jax.lax.with_sharding_constraint(
            query_states,
            jax.sharding.PartitionSpec("batch", None, "model", None),
        )
        key_states = jax.lax.with_sharding_constraint(
            key_states,
            jax.sharding.PartitionSpec("batch", None, "model", None),
        )
        value_states = jax.lax.with_sharding_constraint(
            value_states,
            jax.sharding.PartitionSpec("batch", None, "model", None),
        )

        query_states, key_states = self.rotary_emb(
            query_states, key_states, position_ids
        )

        if init_cache:
            key_states = jax.lax.with_sharding_constraint(
                key_states,
                jax.sharding.PartitionSpec("batch", None, "model", None),
            )
            value_states = jax.lax.with_sharding_constraint(
                value_states,
                jax.sharding.PartitionSpec("batch", None, "model", None),
            )
            return (key_states, value_states)

        attn_weights = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states)
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_output = jax.lax.with_sharding_constraint(
            attn_output,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class FlaxQwen25MLP(nn.Module):
    """MLP module with tensor parallelism support."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.gate_proj = nn.Dense(
            config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("mlp", "embed"))},
        )
        self.act_fn = ACT2FN[config.hidden_act]
    
    def __call__(self, x, deterministic: bool = True):
        # Apply gate and up projections
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # Apply tensor parallelism constraints
        gate_output = jax.lax.with_sharding_constraint(
            gate_output, P('batch', None, 'model'))
        up_output = jax.lax.with_sharding_constraint(
            up_output, P('batch', None, 'model'))
        
        # Apply SwiGLU activation
        hidden_states = self.act_fn(gate_output) * up_output
        
        # Apply down projection
        down_output = self.down_proj(hidden_states)
        
        # Apply final tensor parallelism constraint
        down_output = jax.lax.with_sharding_constraint(
            down_output, P('batch', None, None))
        
        return down_output

class FlaxQwen25DecoderLayer(nn.Module):
    """Transformer decoder layer with tensor parallelism support."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxQwen25Attention(
            self.config,
            dtype=self.dtype,
            name="self_attn",
        )
        self.mlp = FlaxQwen25MLP(
            self.config,
            dtype=self.dtype,
            name="mlp",
        )
        self.input_layernorm = FlaxQwen25RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="input_layernorm",
        )
        self.post_attention_layernorm = FlaxQwen25RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="post_attention_layernorm",
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        # Self Attention
        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        attn_output = jax.lax.with_sharding_constraint(
            attn_output,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        # Residual connection
        hidden_states = residual + attn_output
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        # Residual connection
        hidden_states = residual + hidden_states
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_outputs[1],)
        return outputs

class FlaxQwen25PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Qwen25Config
    base_model_prefix = "model"
    module_class: nn.Module = None
    
    def __init__(
        self,
        config: Qwen25Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # Initialize input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        module_init_outputs = self.module.init(
            rngs, input_ids, attention_mask, position_ids, return_dict=False
        )
        
        random_params = module_init_outputs["params"]
        
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in set(random_params) - set(params):
                params[missing_key] = random_params[missing_key]
            return freeze(unflatten_dict(params))
        else:
            return random_params

class FlaxQwen25Module(nn.Module):
    """Qwen2.5 transformer model."""
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
        )
        self.layers = [
            FlaxQwen25DecoderLayer(
                self.config,
                dtype=self.dtype,
                name=f"layer_{i}",
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.final_layernorm = FlaxQwen25RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="final_layernorm",
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :],
                (batch_size, seq_length),
            )

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            hidden_states = jax.lax.with_sharding_constraint(
                hidden_states,
                jax.sharding.PartitionSpec("batch", None, "model"),
            )
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states,
            jax.sharding.PartitionSpec("batch", None, "model"),
        )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_self_attns,)
        return outputs

class FlaxQwen25ForCausalLM(FlaxQwen25PreTrainedModel):
    """Qwen2.5 model for causal language modeling."""
    module_class = FlaxQwen25Module
    
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.ndarray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
        
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs 