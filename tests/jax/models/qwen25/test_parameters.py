#!/usr/bin/env python3
"""
Test script to verify parameter loading and configuration for Qwen25 model.
This script checks parameter shapes, statistics, and values to ensure correct loading.

Usage:
    python test_parameters.py --model_path /path/to/model/directory --dtype bfloat16
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
logger = logging.getLogger("parameter_test")

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

def transpose_if_needed(name: str, param: np.ndarray) -> np.ndarray:
    """Transpose weight matrices if needed based on the parameter name."""
    # Special case for embedding weights - Flax's nn.Embed expects (vocab_size, embedding_dim)
    if "embed_tokens.weight" in name:
        logger.debug(f"Not transposing embedding weight: {name}, shape: {param.shape}")
        return param
    
    # Other attention and MLP weights need to be transposed
    if "weight" in name and ("proj" in name or "lm_head" in name):
        original_shape = param.shape
        transposed = param.T
        logger.debug(f"Transposing weight matrix: {name}, shape: {original_shape} -> {transposed.shape}")
        return transposed
    
    logger.debug(f"Not transposing parameter: {name}, shape: {param.shape}")
    return param

def analyze_parameter(name: str, param: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> Dict:
    """Analyze a parameter's statistics and check for anomalies."""
    stats = {
        "name": name,
        "shape": param.shape,
        "dtype": str(param.dtype),
        "min": float(param.min()),
        "max": float(param.max()),
        "mean": float(param.mean()),
        "std": float(param.std()),
        "has_nan": bool(np.isnan(param).any()),
        "has_inf": bool(np.isinf(param).any()),
        "is_zero": bool(np.all(param == 0)),
        "is_constant": bool(np.all(param == param[0])),
        "shape_matches_expected": expected_shape is None or param.shape == expected_shape
    }
    
    # Add sample values
    if param.size > 0:
        sample_size = min(5, param.size)
        stats["sample_values"] = param.flatten()[:sample_size].tolist()
    
    return stats

def process_safetensors_file(file_path: str, dtype: jnp.dtype) -> Dict:
    """Process a single safetensors file and return parameter statistics."""
    param_stats = {}
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            for key in f.keys():
                # Get the raw tensor
                param = f.get_tensor(key)
                param_path = get_param_path(key)
                
                if param_path is None:
                    continue
                
                # Store the original shape before any processing
                original_shape = param.shape
                
                # Convert to JAX array with the correct dtype
                param = jnp.array(param, dtype=dtype)
                
                # Analyze parameter BEFORE transposition
                before_stats = analyze_parameter(key + "_before_transpose", np.array(param))
                param_stats[key + "_before_transpose"] = before_stats
                
                # Transpose if needed
                param = transpose_if_needed(key, param)
                
                # Analyze parameter AFTER transposition
                stats = analyze_parameter(key, np.array(param))
                
                # Add original shape for comparison
                stats["original_shape"] = original_shape
                
                param_stats[key] = stats
                
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise
    
    return param_stats

def verify_parameter_consistency(stats: Dict[str, Dict], config: Dict[str, Any] = None) -> List[str]:
    """Verify consistency across parameters and return any issues found."""
    issues = []
    
    # Extract parameters, excluding the "_before_transpose" versions
    actual_stats = {k: v for k, v in stats.items() if not k.endswith("_before_transpose")}
    
    # Group parameters by type
    embeddings = {k: v for k, v in actual_stats.items() if "embed_tokens" in k}
    attention_q = {k: v for k, v in actual_stats.items() if "self_attn" in k and "q_proj" in k}
    attention_k = {k: v for k, v in actual_stats.items() if "self_attn" in k and "k_proj" in k}
    attention_v = {k: v for k, v in actual_stats.items() if "self_attn" in k and "v_proj" in k}
    attention_o = {k: v for k, v in actual_stats.items() if "self_attn" in k and "o_proj" in k}
    mlp_gate = {k: v for k, v in actual_stats.items() if "mlp" in k and "gate_proj" in k}
    mlp_up = {k: v for k, v in actual_stats.items() if "mlp" in k and "up_proj" in k}
    mlp_down = {k: v for k, v in actual_stats.items() if "mlp" in k and "down_proj" in k}
    layer_norms = {k: v for k, v in actual_stats.items() if "layernorm" in k}
    
    # Check embeddings
    if embeddings:
        emb_shapes = {tuple(v["shape"]) for v in embeddings.values()}
        if len(emb_shapes) > 1:
            issues.append(f"Inconsistent embedding shapes: {emb_shapes}")
    
    # Check attention weights by type
    attention_types = [
        ("Q projection", attention_q),
        ("K projection", attention_k),
        ("V projection", attention_v),
        ("Output projection", attention_o)
    ]
    
    for name, attn_dict in attention_types:
        if attn_dict:
            shapes = {tuple(v["shape"]) for v in attn_dict.values()}
            if len(shapes) > 1:
                issues.append(f"Inconsistent {name} weight shapes: {shapes}")
            
            # Report all shapes for analysis
            weight_shapes = [f"{k}: {v['shape']} (original: {v.get('original_shape', 'unknown')})" 
                           for k, v in sorted(attn_dict.items())[:5]]  # Show first 5 examples
            logger.info(f"\n{name} weight shapes (sample):")
            for shape_info in weight_shapes:
                logger.info(f"  {shape_info}")
    
    # Check MLP weights by type
    mlp_types = [
        ("MLP Gate", mlp_gate),
        ("MLP Up", mlp_up),
        ("MLP Down", mlp_down)
    ]
    
    for name, mlp_dict in mlp_types:
        if mlp_dict:
            shapes = {tuple(v["shape"]) for v in mlp_dict.values()}
            if len(shapes) > 1:
                issues.append(f"Inconsistent {name} weight shapes: {shapes}")
            
            # Report all shapes for analysis
            weight_shapes = [f"{k}: {v['shape']} (original: {v.get('original_shape', 'unknown')})" 
                           for k, v in sorted(mlp_dict.items())[:5]]  # Show first 5 examples
            logger.info(f"\n{name} weight shapes (sample):")
            for shape_info in weight_shapes:
                logger.info(f"  {shape_info}")
    
    # Check for NaN/Inf values
    for name, stat in actual_stats.items():
        if stat["has_nan"]:
            issues.append(f"NaN values found in {name}")
        if stat["has_inf"]:
            issues.append(f"Inf values found in {name}")
        if stat["is_zero"]:
            issues.append(f"All zeros found in {name}")
        if stat["is_constant"]:
            issues.append(f"Constant values found in {name}")
    
    # Check against model configuration (if provided)
    if config:
        hidden_size = config.get("hidden_size")
        if hidden_size:
            # Verify embedding dimension
            for name, stat in embeddings.items():
                if stat["shape"][1] != hidden_size:
                    issues.append(f"Embedding dimension mismatch in {name}: expected {hidden_size}, got {stat['shape'][1]}")
            
            # Verify attention weights dimensions
            for name, stat in attention_q.items():
                # For transposed weights, second dimension should match hidden_size
                if len(stat["shape"]) == 2 and stat["shape"][1] != hidden_size:
                    issues.append(f"Q projection dimension mismatch in {name}: expected (..., {hidden_size}), got {stat['shape']}")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Test Qwen25 parameter loading")
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
        "--output_file",
        type=str,
        help="Save detailed parameter analysis to this JSON file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no_transpose",
        action="store_true",
        help="Disable parameter transposition to see raw shapes"
    )
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set data type
    dtype = getattr(jnp, args.dtype)
    
    # Override transposition function if requested
    if args.no_transpose:
        global transpose_if_needed
        def no_transpose(name, param):
            logger.debug(f"Transposition disabled, keeping original shape for {name}: {param.shape}")
            return param
        transpose_if_needed = no_transpose
    
    # Load model configuration
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Model configuration: hidden_size={config['hidden_size']}, "
               f"num_heads={config['num_attention_heads']}, "
               f"num_layers={config['num_hidden_layers']}")
    
    # Calculate expected shapes based on config
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_heads
    
    logger.info(f"Expected head_dim: {head_dim}")
    
    # Process all safetensors files
    safetensors_files = sorted([f for f in os.listdir(args.model_path) if f.endswith(".safetensors")])
    all_stats = {}
    
    for filename in safetensors_files:
        logger.info(f"Processing {filename}...")
        file_path = os.path.join(args.model_path, filename)
        file_stats = process_safetensors_file(file_path, dtype)
        all_stats.update(file_stats)
    
    # Verify parameter consistency
    issues = verify_parameter_consistency(all_stats, config)
    
    # Print summary
    logger.info("\nParameter Analysis Summary:")
    logger.info(f"Total parameters analyzed: {len([k for k in all_stats.keys() if not k.endswith('_before_transpose')])}")
    
    if issues:
        logger.warning("\nIssues found:")
        for issue in issues:
            logger.warning(f"- {issue}")
    else:
        logger.info("No issues found in parameter consistency")
    
    # Print detailed stats for key parameters
    key_params = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        # Sample one of each attention projection type from first layer
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        # Sample MLP weights from first layer
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight"
    ]
    
    logger.info("\nDetailed statistics for key parameters:")
    for param_name in key_params:
        if param_name in all_stats:
            stats = all_stats[param_name]
            before_stats = all_stats.get(param_name + "_before_transpose")
            
            logger.info(f"\n{param_name}:")
            logger.info(f"  Shape: {stats['shape']} (original: {stats.get('original_shape', 'unknown')})")
            if before_stats:
                logger.info(f"  Before transpose: {before_stats['shape']}")
            logger.info(f"  Dtype: {stats['dtype']}")
            logger.info(f"  Min: {stats['min']:.6f}")
            logger.info(f"  Max: {stats['max']:.6f}")
            logger.info(f"  Mean: {stats['mean']:.6f}")
            logger.info(f"  Std: {stats['std']:.6f}")
            logger.info(f"  Sample values: {stats['sample_values']}")
    
    # Save detailed analysis if requested
    if args.output_file:
        # Filter out _before_transpose entries to keep the file size reasonable
        filtered_stats = {k: v for k, v in all_stats.items() if not k.endswith("_before_transpose")}
        with open(args.output_file, 'w') as f:
            json.dump(filtered_stats, f, indent=2)
        logger.info(f"\nDetailed analysis saved to {args.output_file}")
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main()) 