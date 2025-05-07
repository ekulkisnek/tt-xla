#!/usr/bin/env python3
"""
Targeted test script to verify attention projection parameters for Qwen25 model.
This script specifically checks Q, K, V, and O projection shapes and values.

Usage:
    python test_attention_projections.py --model_path /path/to/model/directory --dtype bfloat16
"""

import os
import sys
import re
import logging
import argparse
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("attention_test")

def analyze_attention_projections(file_path: str, dtype: jnp.dtype) -> Dict:
    """Analyze attention projection parameters from a safetensors file."""
    projections = {
        'q_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'k_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'v_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'o_proj': {'weights': [], 'shapes': set(), 'layers': {}}
    }
    
    try:
        with safe_open(file_path, framework="numpy") as f:
            for key in f.keys():
                # Only process attention projection weights
                if not any(f"self_attn.{proj}_proj.weight" in key for proj in ['q', 'k', 'v', 'o']):
                    continue
                
                # Get the parameter
                param = f.get_tensor(key)
                
                # Determine projection type
                proj_type = None
                for p in ['q', 'k', 'v', 'o']:
                    if f"self_attn.{p}_proj.weight" in key:
                        proj_type = p
                        break
                
                if proj_type is None:
                    continue
                
                # Convert to JAX array with correct dtype
                param = jnp.array(param, dtype=dtype)
                
                # Store original shape
                original_shape = param.shape
                
                # Transpose if needed (except for Q projection)
                if proj_type != 'q':
                    param = jnp.transpose(param)
                
                # Get layer number
                layer_match = re.search(r'layers\.(\d+)', key)
                layer_num = int(layer_match.group(1)) if layer_match else -1
                
                # Store parameter info
                proj_info = {
                    'layer': layer_num,
                    'original_shape': original_shape,
                    'final_shape': param.shape,
                    'min': float(param.min()),
                    'max': float(param.max()),
                    'mean': float(param.mean()),
                    'std': float(param.std()),
                    'has_nan': bool(np.isnan(param).any()),
                    'has_inf': bool(np.isinf(param).any())
                }
                
                projections[f'{proj_type}_proj']['weights'].append(param)
                projections[f'{proj_type}_proj']['shapes'].add(param.shape)
                projections[f'{proj_type}_proj']['layers'][layer_num] = proj_info
                
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise
    
    return projections

def verify_attention_consistency(projections: Dict) -> List[str]:
    """Verify consistency of attention projections and return any issues found."""
    issues = []
    
    # Check shapes for each projection type
    for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        shapes = projections[proj_type]['shapes']
        if len(shapes) > 1:
            issues.append(f"Inconsistent {proj_type} shapes: {shapes}")
            
            # Log detailed shape information for each layer
            logger.info(f"\nDetailed {proj_type} shape analysis:")
            for layer_num, info in sorted(projections[proj_type]['layers'].items()):
                logger.info(f"Layer {layer_num}:")
                logger.info(f"  Original shape: {info['original_shape']}")
                logger.info(f"  Final shape: {info['final_shape']}")
                logger.info(f"  Stats: min={info['min']:.6f}, max={info['max']:.6f}, "
                          f"mean={info['mean']:.6f}, std={info['std']:.6f}")
    
    # Check for NaN/Inf values
    for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        for layer_num, info in projections[proj_type]['layers'].items():
            if info['has_nan']:
                issues.append(f"NaN values found in {proj_type} at layer {layer_num}")
            if info['has_inf']:
                issues.append(f"Inf values found in {proj_type} at layer {layer_num}")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Test Qwen25 attention projections")
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
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set data type
    dtype = getattr(jnp, args.dtype)
    
    # Process all safetensors files
    safetensors_files = sorted([f for f in os.listdir(args.model_path) if f.endswith(".safetensors")])
    all_projections = {
        'q_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'k_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'v_proj': {'weights': [], 'shapes': set(), 'layers': {}},
        'o_proj': {'weights': [], 'shapes': set(), 'layers': {}}
    }
    
    for filename in safetensors_files:
        logger.info(f"Processing {filename}...")
        file_path = os.path.join(args.model_path, filename)
        file_projections = analyze_attention_projections(file_path, dtype)
        
        # Merge projections
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            all_projections[proj_type]['weights'].extend(file_projections[proj_type]['weights'])
            all_projections[proj_type]['shapes'].update(file_projections[proj_type]['shapes'])
            all_projections[proj_type]['layers'].update(file_projections[proj_type]['layers'])
    
    # Verify consistency
    issues = verify_attention_consistency(all_projections)
    
    # Print summary
    logger.info("\nAttention Projection Analysis Summary:")
    for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        logger.info(f"\n{proj_type.upper()}:")
        logger.info(f"Total layers found: {len(all_projections[proj_type]['layers'])}")
        logger.info(f"Unique shapes: {all_projections[proj_type]['shapes']}")
    
    if issues:
        logger.warning("\nIssues found:")
        for issue in issues:
            logger.warning(f"- {issue}")
    else:
        logger.info("\nNo issues found in attention projections")
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main()) 