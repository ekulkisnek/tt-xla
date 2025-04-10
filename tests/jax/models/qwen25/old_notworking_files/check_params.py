#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Debug utility to print parameter names from Qwen2.5-7B safetensors files.
"""

import os
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("PARAM_CHECK")

# Import our weight loading utilities with the correct relative import
from tests.jax.models.qwen2_5.weight_loading import print_safetensors_param_names, convert_weight_name_to_flax

def main():
    """Print parameter names from safetensors files."""
    parser = argparse.ArgumentParser(description="Print Qwen2.5 parameter names")
    parser.add_argument("--weights_path", type=str, required=True,
                      help="Path to the model weights directory")
    parser.add_argument("--filter", type=str, default="",
                      help="Filter parameter names (case-insensitive)")
    
    args = parser.parse_args()
    
    # Verify weights path exists
    if not os.path.exists(args.weights_path):
        logger.error(f"Weights path {args.weights_path} does not exist")
        return 1
    
    # Print parameter names
    print_safetensors_param_names(args.weights_path)
    
    # Print a summary of Flax conversions if a filter is specified
    if args.filter:
        filter_term = args.filter.lower()
        logger.info(f"Checking Flax name conversion for parameters containing '{filter_term}'")
        
        # Load the index to get all parameter names
        index_path = os.path.join(args.weights_path, 'model.safetensors.index.json')
        if not os.path.exists(index_path):
            logger.error(f"Safetensors index file not found at {index_path}")
            return 1
            
        import json
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        
        # Filter for relevant parameters and show their Flax conversion
        matching_params = [name for name in weight_map.keys() if filter_term in name.lower()]
        
        if not matching_params:
            logger.info(f"No parameters found containing '{filter_term}'")
            return 0
            
        logger.info(f"Found {len(matching_params)} parameters containing '{filter_term}'")
        
        for param_name in sorted(matching_params):
            flax_name = convert_weight_name_to_flax(param_name)
            logger.info(f"PyTorch: {param_name} -> Flax: {flax_name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 