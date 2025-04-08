#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to directly run the Qwen2.5-7B tester.
"""

import os
import sys
import argparse

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import the tester
from tests.jax.models.qwen2_5.tester import Qwen25Tester, Qwen25SmallTester
from infra import ComparisonConfig, RunMode

def main():
    parser = argparse.ArgumentParser(description='Run Qwen2.5 tester')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model weights directory')
    parser.add_argument('--use_tensor_parallel', action='store_true',
                        help='Whether to use tensor parallelism')
    parser.add_argument('--mesh_shape', type=str, default='1x1',
                        help='Shape of the device mesh for tensor parallelism (batch, model)')
    parser.add_argument('--use_small_config', action='store_true',
                        help='Whether to use a small model config for testing')
    
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
    
    # Create the appropriate tester
    print("Creating tester...")
    if args.use_small_config:
        tester = Qwen25SmallTester(
            use_tensor_parallel=args.use_tensor_parallel,
            mesh_shape=mesh_shape
        )
        print("Using small model configuration for testing")
    else:
        tester = Qwen25Tester(
            model_path=args.model_path,
            use_tensor_parallel=args.use_tensor_parallel,
            mesh_shape=mesh_shape
        )
        if args.model_path:
            print(f"Using model weights from {args.model_path}")
        else:
            print("Using default Qwen2.5-7B configuration with random weights")
    
    # Run the test
    print("\nRunning tester...")
    tester.test()
    print("\nTest completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 