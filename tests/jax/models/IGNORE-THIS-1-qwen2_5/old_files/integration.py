# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration utilities for Qwen2.5-7B model.
This module provides utilities for integrating the model with the testing framework.
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Optional, Callable

from .tester import Qwen25Tester, Qwen25SmallTester
from .config import supported_mesh_configs
from .tensor_parallel import create_device_mesh
from .weight_loading import init_model_from_weights

# Define model name
MODEL_NAME = "qwen2.5-7b"

def get_supported_mesh_configs() -> List[Dict[str, Any]]:
    """
    Get the list of supported mesh configurations for the Qwen2.5-7B model.
    
    Returns:
        List of mesh configuration dictionaries
    """
    return supported_mesh_configs()

def get_tensor_parallel_test_configs() -> List[Dict[str, Any]]:
    """
    Get the list of test configurations for tensor parallelism.
    
    Returns:
        List of test configuration dictionaries
    """
    return [
        {
            "name": f"tp-{config['shape'][0]}x{config['shape'][1]}",
            "mesh_shape": config['shape'],
            "axis_names": config['axis_names'],
            "description": config['description'],
            "required_devices": config['shape'][0] * config['shape'][1]
        }
        for config in supported_mesh_configs()
    ]

def get_model_test_runner(
    model_path: Optional[str] = None,
    mesh_shape: Tuple[int, int] = (1, 8),
    use_small_config: bool = False
) -> Callable:
    """
    Get a test runner function for the model.
    
    Args:
        model_path: Path to the model weights directory
        mesh_shape: Shape of the device mesh for tensor parallelism
        use_small_config: Whether to use a small model config for testing
        
    Returns:
        Function that runs the test when called
    """
    def run_test():
        # Set XLA flags if needed
        required_devices = mesh_shape[0] * mesh_shape[1]
        if "XLA_FLAGS" not in os.environ:
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={required_devices}"
        
        # Create appropriate tester
        if use_small_config:
            tester = Qwen25SmallTester(
                use_tensor_parallel=True,
                mesh_shape=mesh_shape
            )
        else:
            tester = Qwen25Tester(
                model_path=model_path,
                use_tensor_parallel=True,
                mesh_shape=mesh_shape
            )
        
        # Run the test
        tester.test()
    
    return run_test

def run_tensor_parallel_tests(
    model_path: Optional[str] = None,
    mesh_shapes: Optional[List[Tuple[int, int]]] = None,
    use_small_config: bool = False
) -> Dict[str, bool]:
    """
    Run tensor parallel tests for various mesh shapes.
    
    Args:
        model_path: Path to the model weights directory
        mesh_shapes: List of mesh shapes to test, defaults to all supported shapes
        use_small_config: Whether to use a small model config for testing
        
    Returns:
        Dictionary mapping mesh shape string to test success boolean
    """
    # Default to testing all supported mesh shapes if none specified
    if mesh_shapes is None:
        mesh_shapes = [config['shape'] for config in supported_mesh_configs()]
    
    results = {}
    
    for mesh_shape in mesh_shapes:
        shape_str = f"{mesh_shape[0]}x{mesh_shape[1]}"
        print(f"\n{'='*80}")
        print(f"Testing mesh shape {shape_str}")
        print(f"{'='*80}")
        
        required_devices = mesh_shape[0] * mesh_shape[1]
        
        # Set XLA flags for device simulation
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={required_devices}"
        
        try:
            # Create appropriate tester
            if use_small_config:
                tester = Qwen25SmallTester(
                    use_tensor_parallel=True,
                    mesh_shape=mesh_shape
                )
            else:
                tester = Qwen25Tester(
                    model_path=model_path,
                    use_tensor_parallel=True,
                    mesh_shape=mesh_shape
                )
            
            # Run the test
            tester.test()
            results[shape_str] = True
            print(f"✅ Test successful for mesh shape {shape_str}")
        except Exception as e:
            results[shape_str] = False
            print(f"❌ Test failed for mesh shape {shape_str}: {str(e)}")
    
    # Print summary
    print("\nTest Summary:")
    for shape_str, success in results.items():
        print(f"  - {shape_str}: {'✅ Success' if success else '❌ Failed'}")
    
    return results

def load_and_run_inference(
    model_path: str,
    mesh_shape: Tuple[int, int] = (1, 8),
    input_text: str = "Hello, world!"
) -> str:
    """
    Load the model and run inference on the given input text.
    
    Args:
        model_path: Path to the model weights directory
        mesh_shape: Shape of the device mesh for tensor parallelism
        input_text: Input text to generate from
        
    Returns:
        Generated text
    """
    from transformers import AutoTokenizer
    
    # Set XLA flags for device simulation if needed
    required_devices = mesh_shape[0] * mesh_shape[1]
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={required_devices}"
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        try:
            print("Falling back to HuggingFace tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        except:
            print("Failed to load tokenizer")
            return "Error: Failed to load tokenizer"
    
    # Create mesh
    mesh = create_device_mesh(mesh_shape)
    
    # Load model
    try:
        from .config import load_qwen_config
        config = load_qwen_config(model_path)
        model, params = init_model_from_weights(
            model_class="TensorParallelQwen2ForCausalLM",
            model_path=model_path,
            config=config,
            mesh=mesh
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: Failed to load model: {str(e)}"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    # TODO: Implement actual inference logic here
    # This is a placeholder for the actual inference code
    
    return "Inference functionality not fully implemented in this integration example" 