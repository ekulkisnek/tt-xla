#!/usr/bin/env python3
"""
Compare parameter counts between PyTorch and JAX implementations of Qwen2.5.

Usage:
    python compare_params.py --model_path /path/to/model/weights
"""

import os
import sys
import logging
import argparse
import json
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from functools import partial
import torch

# Add the directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

# Import Qwen25 model and parameter loading functions
from model import create_qwen25_model
from run_inference import process_safetensors_file, merge_param_dicts

# Import safetensors for weight loading
try:
    from safetensors import safe_open
except ImportError:
    logging.error("safetensors package is required. Install with: pip install safetensors")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qwen25_param_compare")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare parameter counts between PyTorch and JAX implementations")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def count_pytorch_params(model_path):
    """Count parameters in PyTorch model."""
    try:
        from transformers import AutoModelForCausalLM
        logger.info("Loading PyTorch model...")
        pt_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        total_pt = sum(p.numel() for p in pt_model.parameters())
        logger.info(f"PyTorch model loaded successfully")
        return total_pt
    except ImportError:
        logger.error("PyTorch not available. Install with: pip install torch transformers")
        return None
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {e}")
        return None

def count_jax_params(model_path, dtype=jnp.bfloat16):
    """Count parameters in JAX model."""
    try:
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model structure
        logger.info("Creating JAX model structure...")
        model = create_qwen25_model(config, dtype=dtype)
        
        # Load parameters
        logger.info("Loading JAX parameters...")
        params = {"params": {}}
        
        # Process weights from safetensors files
        safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        # Load parameters
        for filename in safetensors_files:
            weight_path = os.path.join(model_path, filename)
            logger.info(f"Processing file: {filename}")
            
            # Process one file at a time
            file_params = process_safetensors_file(weight_path, dtype=dtype)
            
            # Merge parameters
            params = merge_param_dicts(params, file_params)
        
        # Count parameters
        total_fx = sum(x.size for x in jax.tree_util.tree_leaves(params["params"]))
        return total_fx
        
    except Exception as e:
        logger.error(f"Error counting JAX parameters: {e}")
        return None

def main():
    """Compare parameter counts between PyTorch and JAX implementations."""
    # Parse command line arguments
    args = parse_args()
    
    # Set debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Count PyTorch parameters
    total_pt = count_pytorch_params(args.model_path)
    if total_pt is not None:
        logger.info(f"PyTorch parameters: {total_pt:,}")
    
    # Count JAX parameters
    total_fx = count_jax_params(args.model_path)
    if total_fx is not None:
        logger.info(f"JAX parameters: {total_fx:,}")
    
    # Compare counts if both were successful
    if total_pt is not None and total_fx is not None:
        if total_pt == total_fx:
            logger.info("✅ Parameter counts match exactly!")
        else:
            logger.error(f"❌ Parameter counts do not match! Difference: {abs(total_pt - total_fx):,}")
            logger.error(f"PyTorch: {total_pt:,}")
            logger.error(f"JAX: {total_fx:,}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 