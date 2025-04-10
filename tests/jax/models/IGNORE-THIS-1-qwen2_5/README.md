# Qwen2.5-7B JAX/Flax Implementation with Tensor Parallelism

This is a fixed implementation of the Qwen2.5-7B model in JAX/Flax with tensor parallelism support. The implementation follows best practices from Hugging Face Transformers and provides efficient tensor parallelism across multiple devices.

## Features

- Full Qwen2.5-7B model implementation in JAX/Flax
- Tensor parallelism support with multiple mesh configurations (2x4, 1x8, 1x32, 8x4)
- Efficient weight loading with PyTorch to Flax conversion
- GSM8K evaluation with tensor parallelism
- Comprehensive test suite

## Requirements

- JAX >= 0.4.13
- Flax >= 0.7.5
- Transformers >= 4.36.0
- NumPy >= 1.24.0
- Datasets >= 2.14.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/tt-xla.git
cd tt-xla
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Loading the Model

```python
from tests.jax.models.fixed_qwen2_5.weight_loading import load_qwen_model

# Load model with tensor parallelism
model, tokenizer = load_qwen_model(
    model_path="/path/to/qwen2.5-7b",
    mesh_shape=(1, 8),  # Use 8-way tensor parallelism
    from_pt=True,  # Convert from PyTorch weights
    use_cache=True  # Cache converted weights for faster loading
)
```

### Running Inference

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Create device mesh
devices = jax.devices()
device_mesh = jnp.array(devices[:8]).reshape(1, 8)
mesh = Mesh(device_mesh, ('batch', 'model'))

# Run inference with tensor parallelism
with mesh:
    # Prepare input
    inputs = tokenizer("What is the capital of France?", return_tensors="np")
    
    # Apply sharding constraints
    input_ids = jax.lax.with_sharding_constraint(
        inputs["input_ids"], P('batch', None))
    
    # Generate output
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.0
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
```

### Evaluating on GSM8K

```python
from tests.jax.models.fixed_qwen2_5.gsm8k_eval import evaluate_gsm8k

# Run evaluation
results = evaluate_gsm8k(
    model_path="/path/to/qwen2.5-7b",
    mesh_shape=(1, 8),
    batch_size=1,
    max_new_tokens=512,
    temperature=0.0,
    output_file="gsm8k_results.json"
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct']}/{results['total']}")
```

## Implementation Details

### Model Architecture

The implementation follows the Qwen2.5 architecture with the following components:

- RMSNorm for layer normalization
- Grouped Query Attention (GQA) with 4 key/value heads
- SwiGLU activation in MLP
- Rotary Position Embeddings (RoPE)

### Tensor Parallelism

The implementation supports various tensor parallelism configurations through JAX's device mesh and sharding primitives:

- Embedding layer: Sharded on embedding dimension
- Attention:
  - Query/Key/Value projections: Sharded on output dimension
  - Output projection: Sharded on input dimension
- MLP:
  - Gate/Up projections: Sharded on output dimension
  - Down projection: Sharded on input dimension
- Layer norms: Replicated across devices

### Weight Loading

The implementation provides efficient weight loading with the following features:

- One-time conversion from PyTorch to Flax format
- Caching of converted weights for faster loading
- Automatic tensor sharding based on mesh configuration
- Support for loading from Hugging Face Hub or local files

## Testing

Run the test suite to verify the implementation:

```bash
python -m unittest tests.jax.models.fixed_qwen2_5.test_model
```

## License

Apache License 2.0

## Acknowledgments

- Hugging Face Transformers library for reference implementations
- Qwen team for the original PyTorch implementation
- JAX team for tensor parallelism primitives