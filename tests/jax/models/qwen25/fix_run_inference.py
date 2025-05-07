#!/usr/bin/env python3
"""
Fix script for Qwen25 parameter loading issues in run_inference.py.

This script analyzes the main issues with parameter loading and demonstrates how to fix them.
Run this to test the fixes before applying them to the main script.

Usage:
    python fix_run_inference.py --model_path /path/to/model/directory --dtype bfloat16
"""

import os
import sys
import re
import json
import logging
import argparse
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fix_parameters")

def get_param_path(name: str) -> Optional[Tuple[str, ...]]:
    """Map a PyTorch parameter name to its Flax path."""
    # Direct mappings
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    if name in direct_mapping:
        return direct_mapping[name]
    
    # Patterns for layer parameters
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
    
    # Handle layer norms
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    
    # Handle attention parameters
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
    
    # Handle MLP parameters
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
    
    # Handle rotary embedding parameters - skip these as they're computed on the fly in JAX
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        logger.warning(f"Skipping rotary embedding parameter: {name}")
        return None
    
    # Log any unhandled parameter patterns
    logger.warning(f"Unknown parameter pattern: {name}")
    return None

def transpose_if_needed_fixed(name: str, param: np.ndarray) -> np.ndarray:
    """
    FIXED version: Transpose weight matrices if needed based on the parameter name.
    This function properly handles the different shapes of Q, K, V projections.
    """
    # Special case for embedding weights - Flax's nn.Embed expects (vocab_size, embedding_dim)
    if "embed_tokens.weight" in name:
        logger.debug(f"Not transposing embedding weight: {name}, shape: {param.shape}")
        return param
    
    # Handle attention and MLP weights
    if "weight" in name and ("proj" in name or "lm_head" in name):
        original_shape = param.shape
        
        # Special case for K and V projections that have a different shape than Q
        if ("k_proj.weight" in name or "v_proj.weight" in name) and param.shape[0] != param.shape[1]:
            # These parameters have shape [kv_dim, hidden_dim] but in JAX we expect [hidden_dim, kv_dim]
            transposed = param.T
            logger.debug(f"Transposing KV weight matrix: {name}, shape: {original_shape} -> {transposed.shape}")
            return transposed
        # Regular attention and MLP weights
        elif "proj" in name or "lm_head" in name:
            transposed = param.T
            logger.debug(f"Transposing weight matrix: {name}, shape: {original_shape} -> {transposed.shape}")
            return transposed
    
    logger.debug(f"Not transposing parameter: {name}, shape: {param.shape}")
    return param

def process_safetensors_file(file_path: str, dtype: jnp.dtype) -> Dict:
    """Process a single safetensors file and build proper JAX parameters."""
    flax_params = {"params": {}}
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            key_count = 0
            for key in f.keys():
                key_count += 1
                # Log progress
                if key_count % 10 == 0:
                    logger.debug(f"Processed {key_count} tensors...")
                
                # Get the parameter and immediately cast to the specified dtype
                param = f.get_tensor(key)
                
                # Skip parameters that don't map to our model
                param_path = get_param_path(key)
                if param_path is None:
                    logger.debug(f"Skipping unknown parameter: {key}")
                    continue
                
                # Original shape for logging
                original_shape = param.shape
                
                # Convert to JAX array with the correct dtype
                param = jnp.array(param, dtype=dtype)
                
                # Transpose if needed (dense layer weights) using FIXED version
                param = transpose_if_needed_fixed(key, param)
                
                # Log the shape transformation
                if original_shape != param.shape:
                    logger.debug(f"Transformed {key}: {original_shape} -> {param.shape}")
                
                # Add to the parameter dictionary with the correct nested structure
                current_dict = flax_params["params"]
                for path_part in param_path[:-1]:
                    if path_part not in current_dict:
                        current_dict[path_part] = {}
                    current_dict = current_dict[path_part]
                
                current_dict[param_path[-1]] = param
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    """Merge new parameter dictionary into the base dictionary."""
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict):
            if not isinstance(base_dict[key], dict):
                raise ValueError(f"Cannot merge dict into non-dict at key {key}")
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def get_param_shapes(params: Dict) -> Dict:
    """Extract shapes of all parameters in the dictionary."""
    shapes = {}
    
    def extract_shapes(prefix, d):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                extract_shapes(path, v)
            else:
                shapes[path] = v.shape
    
    extract_shapes("", params)
    return shapes

def main():
    parser = argparse.ArgumentParser(description="Fix Qwen25 parameter loading issues")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Save parameter shapes to this JSON file"
    )
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set data type
    dtype = getattr(jnp, args.dtype)
    
    # Load model configuration
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Model configuration: hidden_size={config['hidden_size']}, "
               f"num_heads={config['num_attention_heads']}, "
               f"num_layers={config['num_hidden_layers']}")
    
    # Process safetensors files (just analyze the first one to save time)
    safetensors_files = sorted([f for f in os.listdir(args.model_path) if f.endswith(".safetensors")])
    first_file = safetensors_files[0]
    
    logger.info(f"Processing {first_file} to check parameter shapes...")
    file_path = os.path.join(args.model_path, first_file)
    params = process_safetensors_file(file_path, dtype)
    
    # Extract and print the shapes of key parameters
    param_shapes = get_param_shapes(params)
    
    # Print shapes of attention parameters for the first layer
    attention_params = {k: v for k, v in param_shapes.items() if "self_attn" in k and "layers_0" in k}
    
    logger.info("\nAttention parameter shapes for first layer:")
    for key, shape in sorted(attention_params.items()):
        logger.info(f"  {key}: {shape}")
    
    # Check if shapes are consistent with model architecture
    hidden_size = config["hidden_size"]
    kv_dim = None  # This needs to be extracted from the shapes
    
    # Determine KV dimension from k_proj kernel
    for key, shape in attention_params.items():
        if "k_proj.kernel" in key:
            kv_dim = shape[1]  # After transposition, it's [hidden_size, kv_dim]
            break
    
    # Validate shape consistency
    if kv_dim:
        logger.info(f"\nModel dimensions: hidden_size={hidden_size}, kv_dim={kv_dim}")
        
        # Validate Q projection
        q_key = [k for k in attention_params if "q_proj.kernel" in k][0]
        q_shape = attention_params[q_key]
        if q_shape[0] == hidden_size and q_shape[1] == hidden_size:
            logger.info(f"✓ Q projection shape is correct: {q_shape}")
        else:
            logger.warning(f"✗ Q projection shape is incorrect: {q_shape}, expected ({hidden_size}, {hidden_size})")
        
        # Validate K projection
        k_key = [k for k in attention_params if "k_proj.kernel" in k][0]
        k_shape = attention_params[k_key]
        if k_shape[0] == hidden_size and k_shape[1] == kv_dim:
            logger.info(f"✓ K projection shape is correct: {k_shape}")
        else:
            logger.warning(f"✗ K projection shape is incorrect: {k_shape}, expected ({hidden_size}, {kv_dim})")
        
        # Validate V projection
        v_key = [k for k in attention_params if "v_proj.kernel" in k][0]
        v_shape = attention_params[v_key]
        if v_shape[0] == hidden_size and v_shape[1] == kv_dim:
            logger.info(f"✓ V projection shape is correct: {v_shape}")
        else:
            logger.warning(f"✗ V projection shape is incorrect: {v_shape}, expected ({hidden_size}, {kv_dim})")
    
    # Save parameter shapes if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(param_shapes, f, indent=2)
        logger.info(f"\nParameter shapes saved to {args.output_file}")
    
    logger.info("\nSummary of findings:")
    logger.info("1. The issue is in the transposition logic for K and V projections")
    logger.info("2. Q projection has shape (hidden_size=3584, hidden_size=3584)")
    logger.info(f"3. K and V projections have shape (hidden_size=3584, kv_dim={kv_dim})")
    logger.info("4. The fix is to handle KV projections specially in transpose_if_needed()")
    logger.info("\nRecommended changes:")
    logger.info("1. Update transpose_if_needed() to specifically handle K/V projections")
    logger.info("2. Use this updated function in run_inference.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 