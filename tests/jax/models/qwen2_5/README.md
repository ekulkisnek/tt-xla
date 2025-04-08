# Qwen2.5-7B JAX/Flax Implementation with Tensor Parallelism

This directory contains a JAX/Flax implementation of the Qwen2.5-7B model with tensor parallelism support, designed to efficiently run on multiple devices like TPUs and GPUs.

## Features

- **JAX/Flax Implementation**: Complete implementation of Qwen2.5 in JAX/Flax for efficient model training and inference
- **Tensor Parallelism**: Support for sharding model weights across multiple devices using JAX SPMD
- **GSPMD Integration**: Leverages JAX GSPMD features for efficient model sharding
- **Weight Conversion**: Utilities for converting PyTorch weights to Flax with proper sharding
- **GSM8K Evaluation**: Script for evaluating the model performance on the GSM8K benchmark

## Files

- `configuration_qwen2_5.py`: Configuration class for Qwen2.5 model parameters
- `modeling_flax_qwen2_5.py`: Core model implementation in Flax
- `tensor_parallel.py`: Utilities for tensor parallelism using JAX SPMD
- `weight_loading.py`: Weight loading and conversion from PyTorch to Flax
- `gsm8k_eval.py`: Evaluation script for the GSM8K math reasoning benchmark
- `__init__.py`: Package initialization file

## Usage

### Loading the Model with Tensor Parallelism

```python
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from tt_xla.tests.jax.models.qwen2_5 import (
    convert_qwen25_checkpoint,
    create_device_mesh,
)

# Create a device mesh for tensor parallelism
mesh = create_device_mesh()

# Load the model
model_path = "/path/to/qwen2.5-7b"
model = convert_qwen25_checkpoint(
    checkpoint_dir=model_path,
    dtype=jnp.float16,  # Or use jnp.bfloat16
    with_lm_head=True,
    mesh=mesh,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Generate text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="np")
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=128,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Evaluating on GSM8K

```bash
python -m tt_xla.tests.jax.models.qwen2_5.gsm8k_eval \
    --model_path /path/to/qwen2.5-7b \
    --output_dir ./gsm8k_results \
    --num_examples 10 \
    --max_new_tokens 512 \
    --use_bfloat16
```

## Implementation Details

This implementation follows the architecture of Qwen2.5 as described in the model documentation and has been closely aligned with the existing Hugging Face PyTorch implementation. 

Key architectural components:
- RoPE positional embeddings
- RMSNorm for layer normalization
- SwiGLU activation function
- Grouped-query attention for efficient computation
- Sliding window attention for handling long sequences (optional)

## Tensor Parallelism Strategy

The tensor parallelism strategy follows standard practices for transformer models:
- Attention projection matrices (Q, K, V) are column-wise sharded
- Output projection matrix (O) is row-wise sharded
- MLP gate and up projections are column-wise sharded
- MLP down projection is row-wise sharded

## References

- [Flax GSPMD Guide](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html)
- [JAX SPMD Programming Guide](https://jax.readthedocs.io/en/latest/spmd.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Qwen2.5-7B Model](https://huggingface.co/Qwen/Qwen2.5-7B) 