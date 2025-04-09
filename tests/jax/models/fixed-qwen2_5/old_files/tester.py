# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tester for Qwen2.5-7B model with tensor parallelism support.
This file implements the ModelTester pattern used throughout the codebase.
"""

import os
from typing import Dict, Mapping, Sequence, Optional, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from infra import ComparisonConfig, ModelTester, RunMode
from tests.infra.multichip_utils import enable_shardy, ShardingMode

from .model_implementation import Qwen2ForCausalLM
from .tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from .config import (
    load_qwen_config,
    get_qwen2_7b_config,
    get_small_config,
    supported_mesh_configs
)
from .weight_loading import load_qwen_weights, init_model_from_weights


class Qwen25Tester(ModelTester):
    """
    Tester for Qwen2.5-7B model with tensor parallelism support.
    
    This tester supports both standard non-parallel model testing and
    tensor-parallel model testing with various mesh configurations.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_tensor_parallel: bool = False,
        mesh_shape: Tuple[int, int] = (1, 8),
        use_small_config: bool = False,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        """
        Initialize the Qwen2.5-7B tester.
        
        Args:
            model_path: Path to the model weights directory
            use_tensor_parallel: Whether to use tensor parallelism
            mesh_shape: Shape of the device mesh for tensor parallelism (batch, model)
            use_small_config: Whether to use a small model config for testing
            comparison_config: Configuration for comparing outputs
            run_mode: Whether to test inference or training
        """
        self._model_path = model_path
        self._use_tensor_parallel = use_tensor_parallel
        self._mesh_shape = mesh_shape
        self._use_small_config = use_small_config
        
        # Validate mesh shape if using tensor parallelism
        if use_tensor_parallel:
            self._validate_mesh_shape(mesh_shape)
            self._mesh = create_device_mesh(mesh_shape)
        else:
            self._mesh = None
        
        # Load model configuration
        if use_small_config:
            self._config = get_small_config(hidden_size=128, num_layers=2)
            print(f"Using small model configuration for testing")
        elif model_path and os.path.exists(os.path.join(model_path, "config.json")):
            self._config = load_qwen_config(model_path)
            print(f"Loaded configuration from {model_path}")
        else:
            self._config = get_qwen2_7b_config()
            print(f"Using default Qwen2.5-7B configuration")
        
        # Print model details
        self._log_model_details()
        
        # Initialize parameters to None - will be set in _get_model()
        self._params = None
        
        # Initialize super class after setting up our internal state
        super().__init__(comparison_config, run_mode)
    
    def _validate_mesh_shape(self, mesh_shape: Tuple[int, int]) -> None:
        """
        Validate that the mesh shape is supported and we have enough devices.
        
        Args:
            mesh_shape: The mesh shape to validate
        
        Raises:
            ValueError: If the mesh shape is invalid or not enough devices
        """
        required_devices = mesh_shape[0] * mesh_shape[1]
        available_devices = len(jax.devices())
        
        # Check if mesh shape is in the list of supported shapes
        supported_shapes = [config['shape'] for config in supported_mesh_configs()]
        if mesh_shape not in supported_shapes:
            print(f"Warning: Mesh shape {mesh_shape} is not in the list of officially supported shapes:")
            for shape in supported_shapes:
                print(f"  - {shape}")
        
        # Check if we have enough devices or XLA_FLAGS is set for simulation
        if available_devices < required_devices and "XLA_FLAGS" not in os.environ:
            print(f"Warning: Not enough devices ({available_devices}) for mesh shape {mesh_shape[0]}x{mesh_shape[1]}")
            print(f"Hint: Set XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}' to simulate")
    
    def _log_model_details(self) -> None:
        """Log model configuration details."""
        print("\nModel Configuration:")
        print(f"- Hidden size: {self._config['hidden_size']}")
        print(f"- Layers: {self._config['num_hidden_layers']}")
        print(f"- Attention heads: {self._config['num_attention_heads']}")
        print(f"- KV heads: {self._config['num_key_value_heads']}")
        print(f"- Vocabulary size: {self._config['vocab_size']}")
        
        if self._use_tensor_parallel:
            print(f"\nTensor Parallelism Configuration:")
            print(f"- Mesh shape: {self._mesh_shape[0]}x{self._mesh_shape[1]}")
            print(f"- Total devices: {self._mesh_shape[0] * self._mesh_shape[1]}")
    
    # @override
    def _get_model(self) -> jax.Array:
        """
        Create and initialize the model.
        
        Returns:
            The initialized model instance
        """
        print(f"\nInitializing {'tensor-parallel' if self._use_tensor_parallel else 'standard'} model...")
        
        if self._use_tensor_parallel:
            # Create tensor-parallel model
            with enable_shardy(True):
                if self._model_path and not self._use_small_config:
                    # Load model with real weights
                    try:
                        print(f"Loading model and weights from {self._model_path}...")
                        model, params = init_model_from_weights(
                            model_class=TensorParallelQwen2ForCausalLM,
                            model_path=self._model_path,
                            config=self._config,
                            mesh=self._mesh,
                            param_dtype=jnp.bfloat16,
                            input_ids_shape=(self._mesh_shape[0], 16)  # Batch size matches mesh
                        )
                        self._params = params
                        print("✅ Model loaded with pretrained weights")
                        return model
                    except Exception as e:
                        print(f"Error loading pretrained weights: {e}")
                        print("Falling back to random initialization")
                
                # Initialize model with random weights
                model = TensorParallelQwen2ForCausalLM(
                    config=self._config,
                    mesh=self._mesh,
                    dtype=jnp.bfloat16,
                    param_dtype=jnp.bfloat16
                )
                
                # Generate random parameters
                batch_size = max(1, self._mesh_shape[0])  # Match batch dimension to mesh
                input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
                
                # Shard input for initialization
                input_sharding = NamedSharding(self._mesh, P('batch', None))
                sharded_input = jax.device_put(input_ids, input_sharding)
                
                # Initialize parameters
                with self._mesh:
                    rng = jax.random.PRNGKey(0)
                    self._params = model.init(rng, sharded_input)
                
                print("✅ Tensor-parallel model initialized with random weights")
                return model
        else:
            # Create standard model
            model = Qwen2ForCausalLM(
                config=self._config,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16
            )
            
            if self._model_path and not self._use_small_config:
                # Try to load pretrained weights
                try:
                    print(f"Loading weights from {self._model_path}...")
                    self._params = load_qwen_weights(
                        model_path=self._model_path,
                        config=self._config,
                        param_dtype=jnp.bfloat16
                    )
                    print("✅ Weights loaded successfully")
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    print("Initializing with random weights")
                    # Initialize with random weights
                    rng = jax.random.PRNGKey(0)
                    self._params = model.init(rng, jnp.ones((1, 16), dtype=jnp.int32))
            else:
                # Initialize with random weights
                print("Initializing with random weights")
                rng = jax.random.PRNGKey(0)
                self._params = model.init(rng, jnp.ones((1, 16), dtype=jnp.int32))
            
            print("✅ Standard model initialized")
            return model
    
    # @override
    def _get_input_activations(self) -> jnp.ndarray:
        """
        Create input activations for model testing.
        
        Returns:
            Input tensor appropriately shaped for the model
        """
        # Create appropriate batch size based on parallelism setting
        if self._use_tensor_parallel:
            batch_size = max(1, self._mesh_shape[0])
        else:
            batch_size = 1
        
        # Create simple input - ones is fine for testing
        input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
        
        # Shard input if using tensor parallelism
        if self._use_tensor_parallel and self._mesh is not None:
            input_sharding = NamedSharding(self._mesh, P('batch', None))
            return jax.device_put(input_ids, input_sharding)
        
        return input_ids
    
    # @override
    def _get_forward_method_name(self) -> str:
        """
        Get the name of the forward pass method.
        
        Returns:
            Name of the forward method
        """
        return "apply"  # Flax models use 'apply' for forward pass
    
    # @override
    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Get keyword arguments for the forward pass.
        
        Returns:
            Dictionary of keyword arguments for the forward pass
        """
        return {
            "params": self._params,
            "input_ids": self._get_input_activations(),
        }
    
    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        """
        Get names of static arguments for JIT compilation.
        
        Returns:
            List of static argument names
        """
        return [
            "output_attentions",
            "output_hidden_states",
            "use_cache",
            "deterministic",
        ]


class Qwen25SmallTester(Qwen25Tester):
    """
    Tester for Qwen2.5 with a small model configuration for quick testing.
    """
    
    def __init__(
        self,
        use_tensor_parallel: bool = False,
        mesh_shape: Tuple[int, int] = (1, 8),
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        """
        Initialize the small Qwen2.5 tester.
        
        Args:
            use_tensor_parallel: Whether to use tensor parallelism
            mesh_shape: Shape of the device mesh for tensor parallelism
            comparison_config: Configuration for comparing outputs
            run_mode: Whether to test inference or training
        """
        super().__init__(
            model_path=None,
            use_tensor_parallel=use_tensor_parallel,
            mesh_shape=mesh_shape,
            use_small_config=True,
            comparison_config=comparison_config,
            run_mode=run_mode
        ) 