#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple test script to run the Qwen2.5-7B model.
"""

import os
import sys
import argparse
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

# Import the model implementation directly
from model_implementation import Qwen2_5ForCausalLM
from tensor_parallel import TensorParallelQwen2ForCausalLM, create_device_mesh
from config import get_qwen2_7b_config
from weight_loading import load_qwen_weights, init_model_from_weights

def main():
    parser = argparse.ArgumentParser(description='Run Qwen2.5 model directly')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model weights directory')
    parser.add_argument('--use_tensor_parallel', action='store_true',
                        help='Whether to use tensor parallelism')
    parser.add_argument('--mesh_shape', type=str, default='1x8',
                        help='Shape of the device mesh for tensor parallelism (batch, model)')
    
    args = parser.parse_args()
    
    # Parse mesh shape
    mesh_parts = args.mesh_shape.split('x')
    if len(mesh_parts) != 2:
        print(f"Error: Invalid mesh shape '{args.mesh_shape}'. Format should be 'AxB'.")
        return 1
    
    mesh_shape = (int(mesh_parts[0]), int(mesh_parts[1]))
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    # Set XLA flags for device simulation if not already set
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={required_devices}"
        print(f"Set XLA_FLAGS to simulate {required_devices} devices")
    
    # Load model configuration
    config = get_qwen2_7b_config()
    print("Using default Qwen2.5-7B configuration")
    
    # Print model details
    print("\nModel Configuration:")
    print(f"- Hidden size: {config['hidden_size']}")
    print(f"- Layers: {config['num_hidden_layers']}")
    print(f"- Attention heads: {config['num_attention_heads']}")
    print(f"- KV heads: {config['num_key_value_heads']}")
    print(f"- Vocabulary size: {config['vocab_size']}")
    
    # Initialize model
    print("\nInitializing model...")
    
    # Use our auto model system to initialize the model
    dtype = jnp.bfloat16
    param_dtype = jnp.bfloat16
    
    if args.use_tensor_parallel:
        print(f"\nTensor Parallelism Configuration:")
        print(f"- Mesh shape: {mesh_shape[0]}x{mesh_shape[1]}")
        print(f"- Total devices: {mesh_shape[0] * mesh_shape[1]}")
        
        # Initialize with tensor parallelism
        mesh = create_device_mesh(mesh_shape)
        model = TensorParallelQwen2ForCausalLM(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh=mesh
        )
        
        # Generate input for testing
        batch_size = max(1, mesh_shape[0])  # Match batch dimension to mesh
        input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
        
        # Shard input for initialization
        input_sharding = NamedSharding(mesh, P('batch', None))
        sharded_input = jax.device_put(input_ids, input_sharding)
        
        # Initialize parameters with random weights within the mesh context
        with mesh:
            print("Executing within mesh context...")
            rng = jax.random.PRNGKey(0)
            params = model.init(rng, sharded_input)
        
            print("✅ Tensor-parallel model initialized with random weights")
            
            # Run a simple forward pass to test the model
            print("\nRunning forward pass...")
            outputs = model.apply(params, sharded_input)
        
    else:
        # Create standard model
        model = Qwen2_5ForCausalLM(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype
        )
        
        # Generate input for testing
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Initialize parameters with random weights
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, input_ids)
        
        print("✅ Standard model initialized with random weights")
        
        # Run a simple forward pass to test the model
        print("\nRunning forward pass...")
        outputs = model.apply(params, input_ids)
    
    # Print output information
    if isinstance(outputs, tuple):
        logits = outputs[0]
        print("Model outputs a tuple. Using first element as logits.")
    else:
        try:
            logits = outputs.logits
            print("Model outputs an object with logits attribute.")
        except AttributeError:
            print(f"Unexpected output type: {type(outputs)}")
            if hasattr(outputs, '__dict__'):
                print(f"Available attributes: {dir(outputs)}")
            return 1
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits mean: {jnp.mean(logits)}")
    print(f"Output logits min: {jnp.min(logits)}")
    print(f"Output logits max: {jnp.max(logits)}")
    
    # If loading from pretrained was requested and path is available, demonstrate it
    if args.model_path and os.path.exists(args.model_path):
        print("\nDemonstrating loading from pretrained weights...")
        try:
            if args.use_tensor_parallel:
                print(f"Loading tensor-parallel model from {args.model_path}...")
                params = load_qwen_weights(args.model_path, model)
                model_initialized = init_model_from_weights(model, params, config, input_ids_shape=(1, 16))
                print("✅ Successfully loaded tensor-parallel model from pretrained weights")
            else:
                print(f"Loading standard model from {args.model_path}...")
                params = load_qwen_weights(args.model_path, model)
                model_initialized = init_model_from_weights(model, params, config, input_ids_shape=(1, 16))
                print("✅ Successfully loaded model from pretrained weights")
        except Exception as e:
            print(f"❌ Failed to load pretrained model: {str(e)}")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 