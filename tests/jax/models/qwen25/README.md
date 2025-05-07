# Tensor-Parallel Qwen2.5-7B JAX Implementation

A JAX implementation of Qwen2.5-7B with tensor parallelism support for multi-device inference.

## Overview

This implementation provides a memory-efficient, tensor-parallelized version of the Qwen2.5-7B model running in JAX. It supports various mesh shapes for tensor parallelism and is designed to run efficiently on multiple devices.

Key features:
- Tensor parallelism (not data parallelism)
- Memory-efficient parameter loading
- Support for various mesh shapes (1x8, 2x4, etc.)
- JAX JIT compilation for fast inference
- Incremental decoding with KV-cache

## Model Architecture

The Qwen2.5-7B model is a decoder-only transformer with the following components:

- **Token Embeddings**: Maps input token IDs to dense vectors
- **Transformer Layers** (28 layers):
  - **Self-Attention**: Multi-head attention with rotary position embeddings
  - **Feed-Forward Network**: SwiGLU activation with gate and up/down projections
- **Layer Normalization**: Applied before attention and FFN
- **Language Modeling Head**: Maps hidden states to vocabulary logits

### Tensor Parallelism Implementation

The model uses tensor parallelism to shard computation across multiple devices:

1. **Attention Computation**:
   - Query, Key, Value projections are sharded across the "model" dimension
   - Each device processes a subset of attention heads
   - Output projection is sharded in the opposite direction

2. **Feed-Forward Network**:
   - Gate and Up projections are sharded across the "model" dimension
   - Down projection is sharded in the opposite direction

3. **Language Modeling Head**:
   - Sharded across the "model" dimension to distribute the large vocabulary matrix

## Usage

### Requirements

- JAX >= 0.4.10
- Flax >= 0.7.2
- safetensors
- transformers

### Running the Model

Basic usage:

```bash
python qwen25_tp.py --model_path /path/to/qwen25-weights --prompt "Your prompt here" --mesh_shape 1,8
```

Options:
- `--model_path`: Path to the Qwen2.5-7B model weights (safetensors format with config.json)
- `--prompt`: Text prompt for generation
- `--max_tokens`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_k`: Top-k sampling parameter (default: 50)
- `--top_p`: Nucleus sampling threshold (default: 0.9)
- `--mesh_shape`: Device mesh shape for tensor parallelism (default: "1,8")
- `--dtype`: Data type for parameters (default: "bfloat16")
- `--simulate_tp`: Simulate tensor parallelism on a single device

### Supported Mesh Shapes

The implementation supports various mesh shapes for tensor parallelism:
- `1,8`: 1 data-parallel shard, 8 model-parallel shards
- `2,4`: 2 data-parallel shards, 4 model-parallel shards
- `1,32`: 1 data-parallel shard, 32 model-parallel shards
- `8,4`: 8 data-parallel shards, 4 model-parallel shards

## Memory Efficiency

The implementation includes several optimizations for memory efficiency:
1. **Streaming Parameter Loading**: Processes one safetensors file at a time
2. **Garbage Collection**: Actively frees memory during parameter loading
3. **KV-Cache Management**: Efficient handling of cached key-value states
4. **JAX JIT Compilation**: Optimizes computation through XLA

## Single-Device Usage

For systems with limited hardware, you can:
1. Use `--simulate_tp` to simulate tensor parallelism on a single device
2. The model automatically adjusts the mesh shape based on available devices

## Implementation Details

### Key Components

- `TensorParallelDense`: Dense layer with tensor parallelism support
- `QwenAttention`: Multi-head attention with tensor parallelism
- `QwenMLP`: Feed-forward network with tensor parallelism
- `Qwen25ForCausalLM`: Top-level model class

### Parameter Loading

The model loads parameters in a memory-efficient manner by:
1. Processing one safetensors file at a time
2. Converting parameters to JAX arrays with the correct dtype
3. Transposing weight matrices as needed
4. Merging parameters into the parameter dictionary
5. Garbage collecting after processing each file

## License

This implementation follows the license of the original Qwen2.5 model.

## Citation

If you use this implementation in your work, please cite both this repository and the original Qwen2.5 model.

# Qwen 2.5 Parameter Loading Fix

This directory contains scripts for loading and testing the Qwen 2.5 model with JAX.

## Issue Overview

The original parameter loading logic in `run_inference.py` had an issue with the transposition of key (K) and value (V) projection weights. While query (Q) projections have shape (hidden_size, hidden_size), K and V projections have shape (kv_dim, hidden_size), where:

- hidden_size = 3584
- kv_dim = 512 (smaller than hidden_size due to grouped-query attention)

When these weights were transposed without accounting for the different shapes, they resulted in incorrect dimensions that caused nonsensical model outputs.

## Fix Description

The fix modifies the `transpose_if_needed()` function in `run_inference.py` to handle K and V projections specifically:

```python
def transpose_if_needed(name, param):
    """Transpose weight matrices if needed based on the parameter name."""
    # Special case for embedding weights - Flax's nn.Embed expects (vocab_size, embedding_dim)
    if "embed_tokens.weight" in name:
        # Do not transpose embedding weights
        return param
    
    # Other attention and MLP weights need to be transposed
    if "weight" in name and ("proj" in name or "lm_head" in name):
        # Special case for K and V projections that have a different shape than Q
        if ("k_proj.weight" in name or "v_proj.weight" in name) and param.shape[0] != param.shape[1]:
            # These parameters have shape [kv_dim, hidden_dim] but in JAX we expect [hidden_dim, kv_dim]
            return jnp.transpose(param)
        # For other attention and MLP weight matrices
        return jnp.transpose(param)
    
    return param
```

This ensures that:
1. K/V weights (shape [512, 3584]) are transposed to [3584, 512]
2. Q weights (shape [3584, 3584]) are transposed to [3584, 3584]
3. Embedding weights are not transposed
4. Other linear weights are transposed as needed

## Testing

Two test scripts are provided:

1. `test_parameters.py`: A detailed parameter analysis that examines shapes, statistics, and validates compatibility
2. `test_run_inference.py`: A direct test of the parameter loading logic in `run_inference.py`

To run the tests:

```bash
python3 tt-xla/tests/jax/models/qwen25/test_parameters.py --model_path /path/to/Qwen2.5-7B --debug
python3 tt-xla/tests/jax/models/qwen25/test_run_inference.py --model_path /path/to/Qwen2.5-7B
```

## Running the Model

After fixing the parameter loading, the model can be run with:

```bash
python3 tt-xla/tests/jax/models/qwen25/run_inference.py --model_path /path/to/Qwen2.5-7B --prompt "Hello, how are you today?" --max_tokens 100 --output_file outputs/test_output.txt
```

## Related Files

- `run_inference.py`: Main inference script with the parameter loading fix
- `test_parameters.py`: Enhanced parameter testing script
- `test_run_inference.py`: Minimal test for the loading logic
- `fix_run_inference.py`: Demonstration of the fix with detailed diagnostics