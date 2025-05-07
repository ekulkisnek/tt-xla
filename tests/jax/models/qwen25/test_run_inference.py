#!/usr/bin/env python3
"""
Test script to verify parameter loading in run_inference.py.

This tests the fix implemented in run_inference.py for KV projection weights.

Usage:
    python test_run_inference.py --model_path /path/to/model/directory
"""

import os
import sys
import json
import logging
import argparse
import jax
import jax.numpy as jnp
import run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_run_inference")

def verify_model_parameters(model_path):
    """Load model parameters and verify the shapes of key attention weights."""
    # Load model configuration first
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Model configuration: hidden_size={config['hidden_size']}, "
               f"num_heads={config['num_attention_heads']}, "
               f"num_layers={config['num_hidden_layers']}")
    
    # Process first safetensors file to check parameter shapes
    safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    if not safetensors_files:
        logger.error(f"No safetensors files found in {model_path}")
        return 1
    
    # Process only the first file to save time (should contain enough layers for testing)
    first_file = safetensors_files[0]
    logger.info(f"Processing file: {first_file}")
    file_path = os.path.join(model_path, first_file)
    
    # Use the process_safetensors_file from run_inference.py
    params = run_inference.process_safetensors_file(file_path, dtype=jnp.bfloat16)
    
    # Access the shapes of key attention weights
    attention_shapes = {}
    
    # Check first layer's Q/K/V projections
    try:
        # Get parameters for the first layer
        layer_0 = params["params"]["layers_0"]
        
        # Q projection
        q_proj_kernel = layer_0["self_attn"]["q_proj"]["kernel"]
        q_proj_bias = layer_0["self_attn"]["q_proj"]["bias"]
        
        # K projection
        k_proj_kernel = layer_0["self_attn"]["k_proj"]["kernel"]
        k_proj_bias = layer_0["self_attn"]["k_proj"]["bias"]
        
        # V projection
        v_proj_kernel = layer_0["self_attn"]["v_proj"]["kernel"]
        v_proj_bias = layer_0["self_attn"]["v_proj"]["bias"]
        
        # Output projection
        o_proj_kernel = layer_0["self_attn"]["o_proj"]["kernel"]
        
        # Store shapes
        attention_shapes["q_proj_kernel"] = q_proj_kernel.shape
        attention_shapes["q_proj_bias"] = q_proj_bias.shape
        attention_shapes["k_proj_kernel"] = k_proj_kernel.shape
        attention_shapes["k_proj_bias"] = k_proj_bias.shape
        attention_shapes["v_proj_kernel"] = v_proj_kernel.shape
        attention_shapes["v_proj_bias"] = v_proj_bias.shape
        attention_shapes["o_proj_kernel"] = o_proj_kernel.shape
        
        # Print shapes
        logger.info("Attention layer shapes for first layer:")
        for name, shape in attention_shapes.items():
            logger.info(f"  {name}: {shape}")
        
        # Verify correct shapes
        hidden_size = q_proj_kernel.shape[0]
        kv_dim = k_proj_kernel.shape[1]
        
        logger.info(f"\nModel dimensions: hidden_size={hidden_size}, kv_dim={kv_dim}")
        
        # Check Q projection shape
        if q_proj_kernel.shape == (hidden_size, hidden_size) and q_proj_bias.shape == (hidden_size,):
            logger.info("✓ Q projection shapes are correct")
        else:
            logger.error(f"✗ Q projection shapes are incorrect: kernel={q_proj_kernel.shape}, bias={q_proj_bias.shape}")
            return 1
        
        # Check K projection shape
        if k_proj_kernel.shape == (hidden_size, kv_dim) and k_proj_bias.shape == (kv_dim,):
            logger.info("✓ K projection shapes are correct")
        else:
            logger.error(f"✗ K projection shapes are incorrect: kernel={k_proj_kernel.shape}, bias={k_proj_bias.shape}")
            return 1
        
        # Check V projection shape
        if v_proj_kernel.shape == (hidden_size, kv_dim) and v_proj_bias.shape == (kv_dim,):
            logger.info("✓ V projection shapes are correct")
        else:
            logger.error(f"✗ V projection shapes are incorrect: kernel={v_proj_kernel.shape}, bias={v_proj_bias.shape}")
            return 1
        
        # Check O projection shape
        if o_proj_kernel.shape == (hidden_size, hidden_size):
            logger.info("✓ O projection shape is correct")
        else:
            logger.error(f"✗ O projection shape is incorrect: kernel={o_proj_kernel.shape}")
            return 1
        
        logger.info("\nAll attention parameter shapes are correct! ✓")
        
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Test run_inference.py parameter loading")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    
    args = parser.parse_args()
    
    return verify_model_parameters(args.model_path)

if __name__ == "__main__":
    sys.exit(main()) 