#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simplified script to directly run the Qwen2.5-7B model using the auto model system.
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

# Import the model implementation
from tests.jax.models.failed_qwen2_5 import (
    AutoQwenModel,
    AutoQwenModelTensorParallel,
    get_model,
    load_qwen_config,
    get_qwen2_7b_config,
    get_small_config,
    create_device_mesh
)

# Set up logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Qwen2.5 model directly')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model weights directory')
    parser.add_argument('--use_tensor_parallel', action='store_true',
                        help='Whether to use tensor parallelism')
    parser.add_argument('--mesh_shape', type=str, default='1x8',
                        help='Shape of the device mesh for tensor parallelism (batch, model)')
    parser.add_argument('--use_small_config', action='store_true',
                        help='Whether to use a small model config for testing')
    parser.add_argument('--max_tokens', type=int, default=20,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--run_gsm8k', action='store_true',
                        help='Run GSM8K benchmark instead of interactive mode')
    
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
    if args.use_small_config:
        config = get_small_config(hidden_size=128, num_layers=2)
        print("Using small model configuration for testing")
    elif args.model_path and os.path.exists(os.path.join(args.model_path, "config.json")):
        config = load_qwen_config(args.model_path)
        print(f"Loaded configuration from {args.model_path}")
    else:
        config = get_qwen2_7b_config()
        print("Using default Qwen2.5-7B configuration")
    
    # Set model_type in config for auto classes
    config["model_type"] = "qwen2_5"
    
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
        
        # Initialize with the auto model system
        model = get_model(
            model_type="qwen2_5",
            use_tensor_parallel=True,
            mesh_shape=mesh_shape,
            config=config,
            dtype=dtype,
            param_dtype=param_dtype
        )
        
        # Create mesh for operations
        mesh = create_device_mesh(mesh_shape)
        
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
        # Create standard model using auto model system
        model = get_model(
            model_type="qwen2_5",
            use_tensor_parallel=False,
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
                model_pretrained = AutoQwenModelTensorParallel.from_pretrained(
                    args.model_path,
                    mesh_shape=mesh_shape,
                    dtype=dtype,
                    param_dtype=param_dtype
                )
                print("✅ Successfully loaded tensor-parallel model from pretrained weights")
            else:
                print(f"Loading standard model from {args.model_path}...")
                model_pretrained = AutoQwenModel.from_pretrained(
                    args.model_path,
                    dtype=dtype,
                    param_dtype=param_dtype
                )
                print("✅ Successfully loaded model from pretrained weights")
        except Exception as e:
            print(f"❌ Failed to load pretrained model: {str(e)}")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 