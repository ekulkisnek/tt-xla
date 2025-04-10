#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Debug script for Qwen2.5 model architecture and weight shapes.
This script prints detailed shape information for model tensors during execution.
"""

import os
import sys
import time
import logging
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any, Union
import datetime

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

import numpy as np
from transformers import AutoTokenizer
from safetensors.numpy import safe_open
from tqdm import tqdm

# Import the model implementation
from . import (
    AutoQwenModel,
    get_model,
    load_qwen_config,
    get_small_config,
    create_device_mesh,
    load_qwen_weights,
    init_model_from_weights
)

from .model_implementation import (
    QwenAttention,
    QwenTransformerBlock,
    Qwen2Model
)

from .gsm8k_test import fix_parameter_shapes

# Configure logging with timestamp
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set up logger
logger = logging.getLogger("QWEN_DEBUG")
logger.setLevel(logging.DEBUG)

def inspect_safetensors(weights_path: str, max_keys: int = 30) -> Dict[str, Tuple[str, tuple]]:
    """
    Inspect the shapes of parameters in safetensors files.
    
    Args:
        weights_path: Path to the safetensors files
        max_keys: Maximum number of keys to print
    
    Returns:
        Dictionary mapping parameter names to their shapes
    """
    logger.info(f"Inspecting safetensors at {weights_path}")
    
    # Check if it's a sharded model
    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info(f"Found sharded model index at {index_file}")
        with open(index_file, "r") as f:
            index_data = json.load(f)
        
        # Get the weight_map to see which parameters are in which files
        weight_map = index_data.get("weight_map", {})
        
        if not weight_map:
            logger.warning("No weight map found in index file")
            return {}
        
        # Collect parameter info from the first few files
        param_info = {}
        files_processed = set()
        
        # Process first few files to get a representative sample
        keys_seen = 0
        for param_name, filename in weight_map.items():
            if keys_seen >= max_keys:
                break
                
            if filename in files_processed:
                continue
                
            file_path = os.path.join(weights_path, filename)
            logger.info(f"Opening safetensors file: {file_path}")
            
            try:
                with safe_open(file_path, framework="numpy") as f:
                    # Get a subset of keys to avoid massive output
                    file_keys = list(f.keys())
                    logger.info(f"File contains {len(file_keys)} parameters")
                    
                    # Only process a limited number of keys per file
                    for key in file_keys[:min(30, len(file_keys))]:
                        if keys_seen >= max_keys:
                            break
                        tensor = f.get_tensor(key)
                        param_info[key] = (str(tensor.dtype), tuple(tensor.shape))
                        keys_seen += 1
                
                files_processed.add(filename)
                
            except Exception as e:
                logger.error(f"Error opening {file_path}: {str(e)}")
        
        return param_info
    else:
        # Non-sharded model, look for a single safetensors file
        model_file = os.path.join(weights_path, "model.safetensors")
        if not os.path.exists(model_file):
            logger.error(f"No safetensors file found at {model_file}")
            return {}
        
        try:
            param_info = {}
            with safe_open(model_file, framework="numpy") as f:
                keys = list(f.keys())
                logger.info(f"Found {len(keys)} parameters in {model_file}")
                
                # Only process a limited number of keys
                for key in keys[:min(max_keys, len(keys))]:
                    tensor = f.get_tensor(key)
                    param_info[key] = (str(tensor.dtype), tuple(tensor.shape))
            
            return param_info
        except Exception as e:
            logger.error(f"Error opening {model_file}: {str(e)}")
            return {}

def analyze_model_config(config: Dict[str, Any]) -> None:
    """
    Analyze the model configuration and print detailed information.
    
    Args:
        config: Model configuration dictionary
    """
    logger.info("Model Configuration Analysis:")
    logger.info(f"  hidden_size: {config.get('hidden_size')}")
    logger.info(f"  intermediate_size: {config.get('intermediate_size')}")
    logger.info(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
    logger.info(f"  num_attention_heads: {config.get('num_attention_heads')}")
    logger.info(f"  num_key_value_heads: {config.get('num_key_value_heads', config.get('num_attention_heads'))}")
    
    # Calculate key dimensions
    hidden_size = config.get('hidden_size')
    num_attention_heads = config.get('num_attention_heads')
    num_key_value_heads = config.get('num_key_value_heads', num_attention_heads)
    
    head_dim = hidden_size // num_attention_heads
    
    logger.info(f"  head_dim: {head_dim}")
    logger.info(f"  qkv dimensions:")
    logger.info(f"    q_proj: (hidden_size, hidden_size) = ({hidden_size}, {hidden_size})")
    logger.info(f"    k_proj: (hidden_size, num_key_value_heads * head_dim) = ({hidden_size}, {num_key_value_heads * head_dim})")
    logger.info(f"    v_proj: (hidden_size, num_key_value_heads * head_dim) = ({hidden_size}, {num_key_value_heads * head_dim})")
    
    logger.info(f"  Expected reshaping dimensions:")
    logger.info(f"    query_states: batch_size, seq_length, num_attention_heads, head_dim = B, S, {num_attention_heads}, {head_dim}")
    logger.info(f"    key_states: batch_size, seq_length, num_key_value_heads, head_dim = B, S, {num_key_value_heads}, {head_dim}")
    logger.info(f"    value_states: batch_size, seq_length, num_key_value_heads, head_dim = B, S, {num_key_value_heads}, {head_dim}")
    
    # Check if model has grouped-query attention
    has_gqa = num_key_value_heads != num_attention_heads
    logger.info(f"  Has grouped-query attention (GQA): {has_gqa}")
    if has_gqa:
        logger.info(f"  GQA Ratio: {num_attention_heads} / {num_key_value_heads} = {num_attention_heads / num_key_value_heads}")

def debug_attention_shapes(config: Dict[str, Any]) -> None:
    """
    Debug attention mechanism shapes using JAX tracing.
    
    Args:
        config: Model configuration dictionary
    """
    logger.info("Debugging attention shapes:")
    
    # Extract key configuration values
    hidden_size = config.get('hidden_size', 3584)
    num_attention_heads = config.get('num_attention_heads', 28)
    num_key_value_heads = config.get('num_key_value_heads', 4)
    head_dim = hidden_size // num_attention_heads
    
    # Log the dimensions
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  num_attention_heads: {num_attention_heads}")
    logger.info(f"  num_key_value_heads: {num_key_value_heads}")
    logger.info(f"  head_dim: {head_dim}")
    
    # Create mock inputs
    batch_size = 1
    seq_length = 30
    hidden_states = jnp.ones((batch_size, seq_length, hidden_size), dtype=jnp.bfloat16)
    
    # Create attention module with tracing
    logger.info("Creating QwenAttention module for tracing")
    attention = QwenAttention(config=config)
    
    # Log QKV projection dimensions
    logger.info("Expected projection dimensions:")
    logger.info(f"  q_proj: input={hidden_states.shape}, output=(B, S, {hidden_size})")
    logger.info(f"  k_proj: input={hidden_states.shape}, output=(B, S, {num_key_value_heads * head_dim})")
    logger.info(f"  v_proj: input={hidden_states.shape}, output=(B, S, {num_key_value_heads * head_dim})")
    
    logger.info("Expected reshaping dimensions:")
    logger.info(f"  query_states: reshape to (B, S, {num_attention_heads}, {head_dim})")
    logger.info(f"  key_states: reshape to (B, S, {num_key_value_heads}, {head_dim})")
    logger.info(f"  value_states: reshape to (B, S, {num_key_value_heads}, {head_dim})")
    
    # Check if we have GQA
    if num_key_value_heads != num_attention_heads:
        repeat_factor = num_attention_heads // num_key_value_heads
        logger.info(f"GQA repeat factor: {repeat_factor}")
        logger.info(f"  After repeat - key/value states: (B, S, {num_attention_heads}, {head_dim})")
    
    logger.info("Final attention output dimensions:")
    logger.info(f"  Expected: (B, S, {hidden_size})")

def main():
    """Main function for the debug script."""
    parser = argparse.ArgumentParser(description="Debug Qwen2.5 model architecture")
    parser.add_argument("--weights_path", type=str, default="/Users/lu/Documents/tt-bounty-1/qwen2.5-7b", 
                        help="Path to the Qwen2.5 weights")
    parser.add_argument("--inspect_weights", action="store_true",
                        help="Inspect the weights in the safetensors files")
    parser.add_argument("--debug_attention", action="store_true",
                        help="Debug attention mechanism shapes")
    parser.add_argument("--max_keys", type=int, default=30,
                        help="Maximum number of keys to inspect")
    
    args = parser.parse_args()
    
    # Load configuration from the weights directory
    config_path = os.path.join(args.weights_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Use default config for Qwen2.5-7B
        logger.warning(f"No config.json found at {config_path}, using default Qwen2.5-7B config")
        config = {
            "hidden_size": 3584,
            "intermediate_size": 9216,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-06,
            "max_position_embeddings": 8192,
            "model_type": "qwen2"
        }
    
    # Analyze the model configuration
    analyze_model_config(config)
    
    # Inspect weights if requested
    if args.inspect_weights:
        param_info = inspect_safetensors(args.weights_path, args.max_keys)
        logger.info(f"Parameter information (first {len(param_info)} parameters):")
        for name, (dtype, shape) in param_info.items():
            logger.info(f"  {name}: {dtype} {shape}")
    
    # Debug attention shapes if requested
    if args.debug_attention:
        debug_attention_shapes(config)
    
    logger.info("Model debugging complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 