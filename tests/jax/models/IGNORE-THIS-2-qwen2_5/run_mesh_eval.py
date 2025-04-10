import argparse
import json
import os
import time
import gc
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from transformers import AutoTokenizer

from tests.jax.models.qwen2_5.configuration_qwen2_5 import Qwen25Config
from tests.jax.models.qwen2_5.modeling_flax_qwen2_5 import FlaxQwen25ForCausalLM
from tests.jax.models.qwen2_5.tensor_parallel import create_device_mesh, get_partition_specs
from tests.jax.models.qwen2_5.sharding import FlaxQwen25WithSharding
from tests.jax.models.qwen2_5.weight_loading import convert_qwen25_checkpoint
from tests.jax.models.qwen2_5.gsm8k_eval import evaluate_gsm8k, extract_answer, check_answer

# Configure JAX for CPU testing
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_jit', False)

def create_cpu_device_mesh(mesh_shape: Tuple[int, int]) -> np.ndarray:
    """Create a device mesh for CPU testing."""
    devices = jax.devices('cpu')
    if not devices:
        raise RuntimeError("No CPU devices found.")
    
    # For CPU testing, we'll use a single device but pretend it's multiple
    device = devices[0]
    device_mesh = np.array([[device] * mesh_shape[1]] * mesh_shape[0])
    return device_mesh

def clear_memory():
    """Clear memory between evaluations."""
    gc.collect()
    jax.clear_caches()

def run_evaluation_with_mesh(
    model_path: str,
    mesh_shape: Tuple[int, int],
    num_examples: int = 1,
    max_length: int = 2048,
    batch_size: int = 1,
) -> Dict:
    """Run evaluation with specific mesh configuration."""
    print(f"\nLoading config from {model_path}")
    config = Qwen25Config.from_pretrained(model_path)
    print(f"Config loaded: {config}")
    
    # Create device mesh for CPU
    device_mesh = create_cpu_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, ('dp', 'mp'))
    
    # Update config for evaluation
    config.use_cache = False  # Disable KV cache for memory efficiency
    
    # Create model with sharding
    print(f"Creating model with config: {config}")
    with mesh:
        model = FlaxQwen25WithSharding(
            config=config,
            dtype=jnp.bfloat16,
            mesh=mesh,
            partition_specs=get_partition_specs(config)
        )
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(0)
        input_shape = (batch_size, 1)
        input_ids = jnp.ones(input_shape, dtype=jnp.int32)
        attention_mask = jnp.ones(input_shape, dtype=jnp.int32)
        position_ids = jnp.zeros(input_shape, dtype=jnp.int32)
        
        params = model.init(
            rng,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )["params"]
        
        # Load pretrained weights
        params = convert_qwen25_checkpoint(model_path, params, config)
        
        # Run a small test forward pass
        outputs = model.apply(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )
        
        # Record results
        results = {
            'mesh_shape': mesh_shape,
            'batch_size': batch_size,
            'max_length': max_length,
            'num_examples': num_examples,
            'device_count': len(jax.devices()),
            'timestamp': datetime.now().isoformat(),
            'test_output_shape': str(outputs.logits.shape)
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 on GSM8K with different mesh configurations")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of examples to evaluate per mesh configuration",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum length of input sequence",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mesh_eval_results",
        help="Directory to save the evaluation results",
    )
    args = parser.parse_args()
    
    # Create output directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Define mesh configurations to test
    mesh_configs = [
        (1, 1),  # Single device
        (2, 4),  # 8 devices
        (1, 8),  # 8 devices in single row
        (1, 32), # 32 devices in single row
        (8, 4),  # 32 devices in 8x4 grid
    ]
    
    all_results = []
    
    for mesh_shape in mesh_configs:
        try:
            print(f"\nRunning evaluation with mesh shape {mesh_shape}")
            results = run_evaluation_with_mesh(
                model_path=args.model_path,
                mesh_shape=mesh_shape,
                num_examples=args.num_examples,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with mesh shape {mesh_shape}: {str(e)}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_file = results_dir / f'mesh_eval_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 