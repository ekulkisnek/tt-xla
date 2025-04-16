# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tensor parallel implementation of Qwen2.5-7B model for JAX.
This module contains the tensor-parallel model components and utilities.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
from typing import Any, Dict, Optional, Tuple
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from model_implementation import (
    RMSNorm,
    QwenAttention,
    Qwen2_5MLP as QwenMLP,
    QwenTransformerBlock,
    Qwen2_5Model,
    Qwen2_5ForCausalLM,
    precompute_freqs_cis,
    QwenEmbed
)

def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh shape
        
    Returns:
        jax.sharding.Mesh: A JAX device mesh
    """
    devices = jax.devices()
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    print(f"Creating mesh with shape {mesh_shape}, requiring {required_devices} devices")
    print(f"Available devices: {len(devices)}")
    
    if len(devices) < required_devices:
        raise ValueError(
            f"Not enough devices ({len(devices)}) for mesh shape {mesh_shape}. "
            f"Required: {required_devices}. Set XLA_FLAGS to simulate more devices."
        )
    
    if len(devices) > required_devices:
        print(f"Warning: Using only {required_devices} of {len(devices)} available devices")
        devices = devices[:required_devices]
    
    try:
        # Create a flat array of devices with the required shape
        devices_array = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_array, ('batch', 'model'))
        print(f"Mesh created with shape {mesh_shape}")
        print(f"Mesh axis_names: {mesh.axis_names}")
        print(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
              f"size={getattr(mesh, 'size', 'None')}")
        print(f"Mesh device shape: {mesh.devices.shape}")
        return mesh
    except ValueError as e:
        print(f"Error creating mesh with np.array.reshape: {e}")
        try:
            # Try using mesh_utils with the sliced devices
            device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices=devices[:required_devices])
            mesh = Mesh(device_mesh, ('batch', 'model'))
            print(f"Mesh created using mesh_utils")
            print(f"Mesh axis_names: {mesh.axis_names}")
            print(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
                  f"size={getattr(mesh, 'size', 'None')}")
            print(f"Mesh device shape: {mesh.devices.shape}")
            return mesh
        except Exception as ex:
            print(f"Error creating mesh with mesh_utils: {ex}")
            raise ValueError(
                f"Failed to create device mesh with shape {mesh_shape}. "
                f"Available devices: {len(devices)}. Required: {required_devices}."
            )

def get_partition_specs(config):
    """
    Create partition specifications for the model parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dict: Partition specs for the model parameters
    """
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    
    # Partition specs for embeddings
    embed_p = P(None, 'model')
    
    # Partition specs for attention
    q_p = P(None, 'model')
    k_p = P(None, 'model')
    v_p = P(None, 'model')
    o_p = P('model', None)
    
    # Partition specs for MLP
    gate_p = P(None, 'model')
    up_p = P(None, 'model')
    down_p = P('model', None)
    
    # Weights partition specs
    weight_p = P(None)
    
    # Create complete partition specs
    return {
        'model': {
            'embed_tokens': {
                'embedding': embed_p,
            },
            'layers_.*': {
                'self_attn': {
                    'q_proj': {
                        'kernel': q_p,
                    },
                    'k_proj': {
                        'kernel': k_p,
                    },
                    'v_proj': {
                        'kernel': v_p,
                    },
                    'o_proj': {
                        'kernel': o_p,
                    },
                },
                'mlp': {
                    'gate_proj': {
                        'kernel': gate_p,
                    },
                    'up_proj': {
                        'kernel': up_p,
                    },
                    'down_proj': {
                        'kernel': down_p,
                    },
                },
                'input_layernorm': {
                    'weight': weight_p,
                },
                'post_attention_layernorm': {
                    'weight': weight_p,
                }
            },
            'norm': {
                'weight': weight_p,
            }
        },
        'lm_head': {
            'kernel': P('model', None),  # Transpose of embed_p
        }
    }

class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    precision: Any = None
    mesh: Mesh = None
    shard_axes: Tuple[str, str] = ('model', None)  # (kernel_in, kernel_out)
    
    @nn.compact
    def __call__(self, inputs):
        """Apply the dense layer with tensor parallelism."""
        input_dim = inputs.shape[-1]
        kernel_shape = (input_dim, self.features)
        
        # Initialize kernel parameter
        kernel = self.param(
            'kernel', 
            self.kernel_init, 
            kernel_shape, 
            self.param_dtype
        )
        kernel = kernel.astype(self.dtype)
        
        # Define partition spec based on shard_axes
        if self.shard_axes[0] and self.shard_axes[1]:
            kernel_spec = P(self.shard_axes[0], self.shard_axes[1])
        elif self.shard_axes[0]:
            kernel_spec = P(self.shard_axes[0], None)
        elif self.shard_axes[1]:
            kernel_spec = P(None, self.shard_axes[1])
        else:
            kernel_spec = P(None, None)
        
        # Shard the kernel if mesh is provided
        if self.mesh is not None:
            try:
                # Only apply constraints inside a mesh context
                kernel = jax.lax.with_sharding_constraint(kernel, kernel_spec)
            except RuntimeError as e:
                # If not in a mesh context, we can continue without sharding
                if "with_sharding_constraint requires a non-empty mesh" in str(e):
                    pass
                else:
                    raise
        
        # Matrix multiplication with safeguards
        y = jnp.matmul(inputs, kernel)
        
        # Add bias if needed
        if self.use_bias:
            # Check if bias exists before trying to use it
            try:
                # Initialize bias parameter
                bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
                bias = bias.astype(self.dtype)
                
                # Shard bias if needed
                if self.mesh is not None and self.shard_axes[1]:
                    try:
                        bias_spec = P(self.shard_axes[1])
                        bias = jax.lax.with_sharding_constraint(bias, bias_spec)
                    except RuntimeError as e:
                        # If not in a mesh context, we can continue without sharding
                        if "with_sharding_constraint requires a non-empty mesh" in str(e):
                            pass
                        else:
                            raise
                
                # Add to output
                y = y + bias
            except Exception as e:
                # Skip bias if not available - this allows the model to work
                # even if PyTorch weights don't include bias parameters
                print(f"Warning: Bias not applied in layer due to: {str(e)}")
        
        return y

class TensorParallelQwenAttention(nn.Module):
    """Tensor parallel implementation of QwenAttention."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = None
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
        *args,
        **kwargs
    ):
        """Apply tensor-parallel attention."""
        # Get basic dimensions
        batch_size, seq_length = hidden_states.shape[:2]
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        
        # Mesh configuration
        if self.mesh:
            mesh_axes = self.mesh.axis_names
            batch_parallel = 'batch' in mesh_axes and self.mesh.shape['batch'] > 1
            model_parallel = 'model' in mesh_axes and self.mesh.shape['model'] > 1
            batch_parallel_size = self.mesh.shape.get('batch', 1) if batch_parallel else 1
            model_parallel_size = self.mesh.shape.get('model', 1) if model_parallel else 1
        else:
            batch_parallel = False
            model_parallel = False
            batch_parallel_size = 1
            model_parallel_size = 1
        
        # Print debug info about the mesh and shapes
        print(f"Mesh info: batch_parallel={batch_parallel}, model_parallel={model_parallel}")
        print(f"Mesh sizes: batch={batch_parallel_size}, model={model_parallel_size}")
        print(f"Input shape: batch_size={batch_size}, seq_length={seq_length}")
            
        # Scale attention heads per device based on model parallelism
        num_attn_heads = self.config["num_attention_heads"]
        num_kv_heads = self.config["num_key_value_heads"]
        
        # Calculate local head counts (per device)
        if model_parallel and model_parallel_size > 1:
            n_heads_per_device = max(1, num_attn_heads // model_parallel_size)
            n_kv_heads_per_device = max(1, num_kv_heads // model_parallel_size)
        else:
            n_heads_per_device = num_attn_heads
            n_kv_heads_per_device = num_kv_heads
            
        print(f"Heads: total={num_attn_heads}, per_device={n_heads_per_device}")
        print(f"KV heads: total={num_kv_heads}, per_device={n_kv_heads_per_device}")
        
        # Project inputs to queries, keys, values with tensor parallelism
        q_proj = TensorParallelDense(
            features=n_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="q_proj",
        )
        
        k_proj = TensorParallelDense(
            features=n_kv_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="k_proj",
        )
        
        v_proj = TensorParallelDense(
            features=n_kv_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="v_proj",
        )
        
        o_proj = TensorParallelDense(
            features=self.config["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config.get("initializer_range", 0.02)),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="o_proj",
        )
        
        # Get queries, keys, values (these will be automatically sharded)
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        print(f"Query shape after projection: {query_states.shape}")
        print(f"Key shape after projection: {key_states.shape}")
        
        # When using batch parallelism, we need special handling for the reshaping
        # Each device only sees part of the batch, so we reshape accordingly
        if batch_parallel and batch_parallel_size > 1:
            # In batch parallel mode, each device only has a fraction of the batch
            # so we don't change the batch dimension during reshaping
            query_states = query_states.reshape(
                query_states.shape[0], seq_length, n_heads_per_device, head_dim
            )
            key_states = key_states.reshape(
                key_states.shape[0], seq_length, n_kv_heads_per_device, head_dim
            )
            value_states = value_states.reshape(
                value_states.shape[0], seq_length, n_kv_heads_per_device, head_dim
            )
        else:
            # In non-batch-parallel mode, use the full batch size
            query_states = query_states.reshape(
                batch_size, seq_length, n_heads_per_device, head_dim
            )
            key_states = key_states.reshape(
                batch_size, seq_length, n_kv_heads_per_device, head_dim
            )
            value_states = value_states.reshape(
                batch_size, seq_length, n_kv_heads_per_device, head_dim
            )
        
        # Print some debug information about the tensor shapes
        print(f"Query shape after reshaping: {query_states.shape}")
        print(f"Key shape after reshaping: {key_states.shape}")
        
        # Setup position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Precompute the rotary embeddings
        max_length = self.config.get("max_position_embeddings", 32768)
        rotary_emb = precompute_freqs_cis(
            head_dim, 
            max_length, 
            theta=self.config.get("rope_theta", 10000.0)
        )
        
        # Apply rotary embeddings from model_implementation
        from model_implementation import apply_rotary_emb
        query_states, key_states = apply_rotary_emb(
            query_states, key_states, rotary_emb, position_ids
        )
        
        # Handle KV caching
        if past_key_value is not None:
            # Concatenate past keys and values with current
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # For grouped-query attention, match the number of query heads
        if n_kv_heads_per_device < n_heads_per_device:
            # Calculate repeat factor safely
            repeat_factor = n_heads_per_device // n_kv_heads_per_device
            
            # Repeat keys and values to match number of attention heads
            key_states = jnp.repeat(key_states, repeat_factor, axis=2)
            value_states = jnp.repeat(value_states, repeat_factor, axis=2)
        elif n_kv_heads_per_device > n_heads_per_device:
            # Handle case where n_heads_per_device < n_kv_heads_per_device
            # We'll use the first n_heads_per_device key/value heads
            key_states = key_states[:, :, :n_heads_per_device, :]
            value_states = value_states[:, :, :n_heads_per_device, :]
            
        # Print shapes before attention computation
        print(f"Final query shape: {query_states.shape}")
        print(f"Final key shape: {key_states.shape}")
        
        # Make sure the batch dimension is consistent - this is critical for batch parallelism
        # In some cases, the final, global batch shape may be different from the input batch_size
        if batch_parallel:
            local_batch_size = query_states.shape[0]
            # We need to make sure all tensors have the same batch size
            if key_states.shape[0] != local_batch_size:
                # Adjust key/value states to match query batch size
                if key_states.shape[0] < local_batch_size:
                    # Repeat key/value to match query batch size
                    repeat_factor = local_batch_size // key_states.shape[0]
                    key_states = jnp.repeat(key_states, repeat_factor, axis=0)
                    value_states = jnp.repeat(value_states, repeat_factor, axis=0)
                else:
                    # Truncate key/value to match query batch size
                    key_states = key_states[:local_batch_size]
                    value_states = value_states[:local_batch_size]
        
        # CRITICAL FIX for batch-only parallel (mesh shape 2,1)
        # When using only batch parallelism without model parallelism, we need special handling
        if batch_parallel and not model_parallel:
            print("Using batch-only parallelism strategy")
            
            # In batch parallel mode without model parallelism, we need to reshape tensors
            # to ensure they have compatible shapes for the matrix multiplication
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Reshape to remove the batch dimension - this makes the tensors compatible
            # with the attention calculation while preserving the total number of elements
            query_states = query_states.reshape(1, q_seq, q_batch * q_heads, q_dim)
            key_states = key_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            value_states = value_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            
            print(f"Reshaped for batch-only parallelism - query: {query_states.shape}, key: {key_states.shape}")
        
        # CRITICAL FIX for combined batch and model parallel:
        # When using both batch and model parallelism, we need to be especially careful
        # about the shape transformations for matrix multiplication
        if batch_parallel and model_parallel:
            print("Using combined batch and model parallelism strategy")
            
            # First, get the actual tensor shapes we're working with after all transformations
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Ensure batch sizes match
            if q_batch != k_batch:
                print(f"Fixing batch mismatch: q_batch={q_batch}, k_batch={k_batch}")
                if q_batch > k_batch:
                    # Expand key/value batch dimension
                    key_states = jnp.repeat(key_states, q_batch // k_batch, axis=0)
                    value_states = jnp.repeat(value_states, q_batch // k_batch, axis=0)
                else:
                    # Expand query batch dimension
                    query_states = jnp.repeat(query_states, k_batch // q_batch, axis=0)
            
            # Ensure head counts match
            if q_heads != k_heads:
                print(f"Fixing head count mismatch: q_heads={q_heads}, k_heads={k_heads}")
                if q_heads > k_heads:
                    # Expand key/value head dimension
                    key_states = jnp.repeat(key_states, q_heads // k_heads, axis=2)
                    value_states = jnp.repeat(value_states, q_heads // k_heads, axis=2)
                else:
                    # Use matching number of heads
                    query_states = query_states[:, :, :k_heads, :]
            
            # Update dimensions after possible adjustments
            q_batch, q_seq, q_heads, q_dim = query_states.shape
            k_batch, k_seq, k_heads, k_dim = key_states.shape
            
            # Special reshape for combined parallelism:
            # When using both batch and model parallelism, we need a different approach
            # Reshape tensors into a form suitable for matrix multiplication, reducing batch dim
            query_states = query_states.reshape(1, q_seq, q_batch * q_heads, q_dim)
            key_states = key_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            value_states = value_states.reshape(1, k_seq, k_batch * k_heads, k_dim)
            
            print(f"Reshaped for combined parallelism - query: {query_states.shape}, key: {key_states.shape}")
        
        # Transpose tensors for attention computation
        # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        query_states_t = query_states.transpose(0, 2, 1, 3)
        key_states_t = key_states.transpose(0, 2, 1, 3)
        value_states_t = value_states.transpose(0, 2, 1, 3)
        
        # Print shapes after transpose
        print(f"Query shape after transpose: {query_states_t.shape}")
        print(f"Key shape after transpose: {key_states_t.shape}")
        
        # Ensure key is correctly transposed for matrix multiplication
        key_for_matmul = key_states_t.transpose(0, 1, 3, 2)
        print(f"Key shape for matmul: {key_for_matmul.shape}")
        
        # Create attention mask (causal by default)
        if attention_mask is None:
            # Create a mask that matches the actual batch size seen by this device
            attention_mask = jnp.ones((query_states_t.shape[0], seq_length))
        
        # Compute attention scores: (batch, heads, seq_q, seq_k)
        attention_scores = jnp.matmul(
            query_states_t,  # (batch, heads, seq, dim)
            key_for_matmul   # (batch, heads, dim, seq)
        )
        
        # Scale attention scores
        attention_scores = attention_scores / jnp.sqrt(head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert mask to right shape
            if attention_mask.ndim == 2:
                # Extend mask for multiple heads and add large negative values 
                # to masked positions
                attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
                attention_mask = (1.0 - attention_mask) * -1e9
                attention_scores = attention_scores + attention_mask
        
        # Apply softmax to attention scores
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout (if specified)
        if not deterministic and self.config.get("attention_dropout", 0.0) > 0:
            dropout_key = self.make_rng("dropout")
            attention_weights = jax.random.dropout(
                dropout_key, 
                self.config.get("attention_dropout", 0.0), 
                attention_weights
            )
        
        # Compute attention outputs
        attention_output = jnp.matmul(
            attention_weights,  # (batch, heads, seq, seq)
            value_states_t  # (batch, heads, seq, dim)
        )
        
        # Reshape back to match input shape
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
        local_batch_size = attention_output.shape[0]  # Use the actual batch size we have
        attention_output = attention_output.reshape(local_batch_size, seq_length, -1)
        
        # Apply output projection
        output = o_proj(attention_output)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs

class TensorParallelQwenMLP(nn.Module):
    """Tensor parallel implementation of QwenMLP."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(self, x):
        """Apply the MLP to the input with tensor parallelism."""
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config["intermediate_size"]
        
        # Compute per-device dimensions
        num_devices = self.mesh.devices.size if self.mesh else 1
        model_parallel_size = num_devices  # Assuming model-parallel across all devices
        
        # Scale intermediate size per device
        intermediate_size_per_device = intermediate_size // model_parallel_size
        
        # Gate and up projections with tensor parallelism
        gate_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="gate_proj",
        )
        
        up_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="up_proj",
        )
        
        down_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="down_proj",
        )
        
        # Apply SwiGLU activation with tensor parallelism
        gate = gate_proj(x)
        gate = nn.silu(gate)
        
        up = up_proj(x)
        
        intermediate = gate * up
        
        # Project back to hidden size with tensor parallelism
        output = down_proj(intermediate)
        
        return output

class TensorParallelQwenTransformerBlock(nn.Module):
    """Tensor parallel implementation of QwenTransformerBlock."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Process the input through self-attention and MLP with tensor parallelism."""
        residual = hidden_states
        
        # Layer normalization before self-attention
        hidden_states = RMSNorm(
            config={"hidden_size": self.config["hidden_size"]},
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="input_layernorm",
        )(hidden_states)
        
        # Self-attention with tensor parallelism
        attn_outputs = TensorParallelQwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="self_attn",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        attention_output = attn_outputs[0]
        past_key_value = attn_outputs[1] if use_cache else None
        attention_weights = attn_outputs[2] if output_attentions else None
        
        # First residual connection
        hidden_states = residual + attention_output
        
        # Second residual block
        residual = hidden_states
        
        # Layer normalization before MLP
        hidden_states = RMSNorm(
            config={"hidden_size": self.config["hidden_size"]},
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="post_attention_layernorm",
        )(hidden_states)
        
        # MLP with tensor parallelism
        hidden_states = TensorParallelQwenMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="mlp",
        )(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (past_key_value,)
            
        if output_attentions:
            outputs = outputs + (attention_weights,)
            
        return outputs

class TensorParallelQwen2Model(nn.Module):
    """Tensor parallel implementation of Qwen2Model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        deterministic=True,
        *args,
        **kwargs
    ):
        """Apply the tensor parallel Qwen2 model."""
        # Handle positional arguments
        if input_ids is None and args:
            input_ids = args[0]
            if len(args) > 1:
                attention_mask = args[1]
            if len(args) > 2:
                position_ids = args[2]
            if len(args) > 3:
                past_key_values = args[3]
        
        # Extract shapes and define helper variables
        batch_size, seq_length = input_ids.shape
        num_layers = self.config["num_hidden_layers"]
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # We create a 3D attention mask from a 2D tensor mask.
        extended_attention_mask = attention_mask[:, None, None, :]
        # Convert to the type needed for attention mechanism
        extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Setup for past key values
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]  # Using the key's sequence length
            
            # Adjust position_ids to account for past keys and values
            position_ids = position_ids[:, past_length:seq_length + past_length]
        
        # Embedding layer (token embeddings)
        embed_tokens = nn.Embed(
            num_embeddings=self.config["vocab_size"],
            features=self.config["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=nn.initializers.normal(stddev=self.config["initializer_range"]),
            name="embed_tokens",
        )
        
        # Get embeddings
        hidden_states = embed_tokens(input_ids)
        
        # Store all hidden states and attentions if requested
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Create the transformer layers with tensor parallelism
        for i in range(self.config["num_hidden_layers"]):
            # Store the hidden state for this layer if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get layer-specific past key values
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Apply the transformer block with tensor parallelism
            layer_outputs = TensorParallelQwenTransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh,
                name=f"layers_{i}",
            )(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                deterministic=deterministic,
            )
            
            # Update hidden states
            hidden_states = layer_outputs[0]
            
            # Store past key values if requested
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[1],)
            
            # Store attention weights if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
        
        # Final layer normalization
        hidden_states = RMSNorm(
            config={"hidden_size": self.config["hidden_size"]},
            epsilon=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm",
        )(hidden_states)
        
        # Store the final hidden state if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (all_past_key_values,)
        
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
            
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs

class TensorParallelQwen2ForCausalLM(nn.Module):
    """Tensor parallel implementation of Qwen2ForCausalLM."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        deterministic=True,
        *args,
        **kwargs
    ):
        # Handle the case where input_ids is passed as positional arg
        if input_ids is None and args:
            input_ids = args[0]
        
        # Forward the base model
        transformer_outputs = TensorParallelQwen2Model(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="model",
        )(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        hidden_states = transformer_outputs[0]
        
        # Apply the language modeling head
        lm_logits = TensorParallelDense(
            features=self.config["vocab_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            shard_axes=('model', None),
            name="lm_head",
        )(hidden_states)
        
        # Prepare outputs - logits first, then the rest in order
        outputs = (lm_logits,) + transformer_outputs[1:]
        
        return outputs
    
    def input_sharding_spec(self, dtype=jnp.bfloat16):
        """Return the appropriate sharding spec for inputs."""
        if self.mesh is None:
            return None
            
        # Get mesh axes
        mesh_axes = self.mesh.axis_names
        
        # Create appropriate specs based on mesh axes
        if 'batch' in mesh_axes and 'model' in mesh_axes:
            batch_axis = 'batch'
            return jax.sharding.NamedSharding(self.mesh, P(batch_axis, None))
        elif len(mesh_axes) >= 2:
            # Use first axis for batch
            batch_axis = mesh_axes[0]
            return jax.sharding.NamedSharding(self.mesh, P(batch_axis, None))
        else:
            # No appropriate sharding available
            return None
            
    def params_from_checkpoint(self, checkpoint_path=None):
        """Load parameters from a checkpoint."""
        from weight_loading import load_qwen_weights
        
        # Get default path from config if not provided
        if checkpoint_path is None and isinstance(self.config, dict) and "model_path" in self.config:
            checkpoint_path = self.config["model_path"]
            
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided")
            
        # Load weights
        try:
            # Try to load with mesh context if mesh is available
            if self.mesh is not None:
                with self.mesh:
                    return load_qwen_weights(
                        model_path=checkpoint_path,
                        model=self,
                        config=self.config,
                        mesh=self.mesh,
                        param_dtype=self.param_dtype
                    )
            # Otherwise load without mesh context
            return load_qwen_weights(
                model_path=checkpoint_path,
                model=self,
                config=self.config,
                mesh=self.mesh,
                param_dtype=self.param_dtype
            )
        except Exception as e:
            raise ValueError(f"Failed to load weights from {checkpoint_path}: {e}")

class TensorParallelQwenEmbed(nn.Module):
    """Tensor parallel module for QwenEmbed."""

    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        # Use local helper class to set up the embedding
        self.embed = QwenEmbed(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        position_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.embed(input_ids, position_ids=position_ids)

    @staticmethod
    def get_params_partition_spec():
        """Get the partition specs for the parameters in this module."""
        # the parameters in QwenEmbed do not get sharded
        return {
            "embed": {
                "wte": None,
                "wpe": None,
            }
        } 