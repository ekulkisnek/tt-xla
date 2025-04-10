# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Registry functions for Qwen2.5-7B model.
"""

from typing import Dict, Any, Optional

from infra import Framework
from tests.utils import ModelTask, ModelSource, ModelGroup

from .tester import Qwen25Tester, Qwen25SmallTester
from .config import supported_mesh_configs


def get_model_metadata() -> Dict[str, Any]:
    """
    Return metadata about the Qwen2.5-7B model for discovery.
    
    Returns:
        Dict: Model metadata
    """
    return {
        "name": "qwen2.5-7b",
        "description": "Qwen2.5-7B model with tensor parallelism",
        "task": ModelTask.NLP_CAUSAL_LM,
        "source": ModelSource.CUSTOM,
        "framework": Framework.JAX,
        "group": ModelGroup.PRIORITY,
        "supports_tensor_parallel": True,
        "supported_mesh_shapes": [
            f"{config['shape'][0]}x{config['shape'][1]}" for config in supported_mesh_configs()
        ],
        "authors": ["Tenstorrent AI"],
        "version": "1.0.0"
    }


def register_model_factory(registry: Any) -> None:
    """
    Register Qwen2.5-7B model factories with the registry.
    
    Args:
        registry: Registry to register factories with
    """
    # Standard model factory
    registry.register_factory(
        "qwen2.5-7b", 
        lambda model_path: Qwen25Tester(
            model_path=model_path,
            use_tensor_parallel=False
        )
    )
    
    # Small model factory for quick testing
    registry.register_factory(
        "qwen2.5-7b-small", 
        lambda model_path: Qwen25SmallTester(
            use_tensor_parallel=False
        )
    )
    
    # Register tensor-parallel model factories for each supported mesh configuration
    for config in supported_mesh_configs():
        mesh_shape = config['shape']
        name_suffix = f"tp-{mesh_shape[0]}x{mesh_shape[1]}"
        
        # Full model with tensor parallelism
        registry.register_factory(
            f"qwen2.5-7b-{name_suffix}",
            lambda model_path, mesh_shape=mesh_shape: Qwen25Tester(
                model_path=model_path,
                use_tensor_parallel=True,
                mesh_shape=mesh_shape
            )
        )
        
        # Small model with tensor parallelism for quick testing
        registry.register_factory(
            f"qwen2.5-7b-small-{name_suffix}",
            lambda model_path, mesh_shape=mesh_shape: Qwen25SmallTester(
                use_tensor_parallel=True,
                mesh_shape=mesh_shape
            )
        )


def get_model_example() -> Dict[str, Any]:
    """
    Return an example configuration for using the model.
    
    Returns:
        Dict: Example configuration
    """
    return {
        "name": "qwen2.5-7b",
        "mesh_shape": "1x8",
        "usage": """
# Set up environment for tensor parallelism
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Run test with tensor parallelism
pytest tt-xla/tests/jax/models/qwen2_5/test_qwen25.py::test_qwen25_tp_1x8

# Or use the small model for quick testing
pytest tt-xla/tests/jax/models/qwen2_5/test_qwen25.py::test_qwen25_small_tp_1x8
"""
    } 