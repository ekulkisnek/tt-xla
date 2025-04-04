# Qwen2.5-7B Tensor-Parallel JAX Implementation

This repository contains a complete tensor-parallel implementation of the Qwen2.5-7B model using JAX and Flax, designed to run efficiently across multiple devices through JAX's tensor parallelism features.

## Model Architecture

Qwen2.5-7B is a transformer-based language model with the following specifications:

- 7 billion parameters
- 28 transformer layers
- 3584 hidden size
- 18944 intermediate size (MLP)
- 28 attention heads
- 4 key-value heads (Grouped Query Attention)
- RMSNorm for layer normalization
- SwiGLU activation in the MLP layers
- Rotary position embeddings (RoPE)

## Tensor Parallelism Implementation

This implementation leverages JAX's device mesh and sharding primitives to distribute computation across multiple devices. The model supports multiple tensor-parallel mesh configurations:

| Mesh Shape | Total Devices | Description |
|------------|---------------|-------------|
| 2x4        | 8             | 2 batch × 4 model parallel |
| 1x8        | 8             | Pure model parallelism with 8 devices |
| 1x32       | 32            | Pure model parallelism with 32 devices |
| 8x4        | 32            | 8 batch × 4 model parallel |

### Tensor Parallelism Architecture

The model parameters are sharded across devices according to the following pattern:

- **Attention Projections**: Query, Key, and Value projections are sharded across the model dimension
- **Attention Output Projection**: Sharded to collect attention heads computed on different devices
- **MLP Projections**: Gate and Up projections are sharded across the model dimension, while Down projection is sharded in the opposite direction
- **Embeddings**: Optionally sharded for improved memory efficiency
- **Layer Norms**: Not sharded as they are computationally inexpensive

## Directory Structure

```
qwen2_5/
├── config.py                # Model configuration utilities
├── model_implementation.py  # Core model implementation
├── tensor_parallel.py       # Tensor-parallel model components
├── weight_loading.py        # Utilities for loading pretrained weights
├── direct_run.py            # Direct execution script
├── __init__.py              # Package initialization
├── README.md                # This documentation
├── AGENT_HISTORY.md         # Development history and issue tracking
└── old_files/               # Legacy code and documentation
    ├── test_qwen25.py           # Test suite for the model
    ├── integration.py           # Integration utilities
    ├── register.py              # Model registration utilities
    ├── tester.py                # Testing framework
    ├── run_tester.py            # Test runner script
    ├── OLD_README.md            # Previous README version
    ├── inference_README.md      # Instructions for interactive inference
    ├── MULTICORE_README.md      # Multi-core processing guide
    ├── INSTRUCTIONS.md          # Verification instructions
    └── ...                      # Other test scripts and examples
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- JAX 0.4.11+ (0.5.0 recommended)
- FLAX 0.7.2+ (0.10.4 recommended)
- NumPy 1.24+

### Python Environment Setup

It's recommended to use a virtual environment to avoid conflicts with other projects:

```bash
# Create a virtual environment
python -m venv qwen2_5_env

# Activate the virtual environment
# On Linux/Mac:
source qwen2_5_env/bin/activate
# On Windows:
# qwen2_5_env\Scripts\activate
```

### Installing Dependencies

Install all required dependencies:

```bash
# Install core dependencies
pip install jax==0.5.0 jaxlib==0.5.0 flax==0.10.4 
pip install transformers datasets safetensors tqdm

# Additional dependencies that may be required
pip install einops fsspec jaxtyping sentencepiece 
```

For a complete development environment matching the original project:
```bash
# Clone the repository if not already done
git clone https://github.com/your-repo/tt-xla.git
cd tt-xla

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Setting up JAX for Tensor Parallelism

Configure JAX to simulate multiple devices for tensor parallelism:

```bash
# For 8 devices (2x4 or 1x8 mesh configurations)
export XLA_FLAGS="--xla_force_host_platform_device_count=8"  

# For 32 devices (1x32 or 8x4 mesh configurations)
export XLA_FLAGS="--xla_force_host_platform_device_count=32"
```

### Model Weights Setup

Ensure the Qwen2.5-7B model weights are available at the correct location:
```
tt-bounty-1/
├── tt-xla/
│   └── tests/
│       └── jax/
│           └── models/
│               └── qwen2_5/
└── qwen2.5-7b/
    ├── config.json
    ├── tokenizer.json
    ├── model.safetensors.index.json
    └── model-0000[1-4]-of-00004.safetensors
```

You can download the model weights from Hugging Face:
```bash
# Install git-lfs if not already installed
# apt-get install git-lfs  # On Ubuntu/Debian
# brew install git-lfs     # On macOS

# Clone the model repository
git clone https://huggingface.co/Qwen/Qwen2.5-7B
```

## Usage

### Interactive Inference

Run interactive inference with the model:

```bash
# Use default 1x8 mesh shape
python direct_run.py

# Specify model path and mesh shape
python direct_run.py --model_path /path/to/qwen2.5-7b --mesh_shape 2x4
```

### Command-Line Arguments

The `direct_run.py` script supports several command-line arguments:

```
--model_path PATH       Path to the model weights directory
--use_tensor_parallel   Whether to use tensor parallelism
--mesh_shape AxB       Shape of the device mesh (batch, model)
--use_small_config      Use a small model config for testing
--max_tokens NUM        Maximum new tokens to generate (default: 20)
--run_gsm8k             Run GSM8K benchmark instead of interactive chat
```

Example usage:
```bash
# Run with custom mesh shape and model path
python direct_run.py --mesh_shape 2x4 --model_path /path/to/qwen2.5-7b

# Run with small test configuration
python direct_run.py --use_small_config
```

### Interactive Mode

When running in interactive mode, you can:

1. Type prompts and see responses from the model
2. Continue the conversation with follow-up questions
3. Type 'exit' or 'quit' to end the session

Example interaction:
```
Enter your prompt: What is the capital of France?
[Model response]

Enter your prompt: Tell me more about its history.
[Model response]
```

### Multi-Core Processing

The implementation supports multi-core processing for tensor parallelism:

1. Set XLA_FLAGS to use multiple devices:
   ```bash
   export XLA_FLAGS="--xla_force_host_platform_device_count=8"
   ```

2. Run with a mesh shape that utilizes multiple cores:
   ```bash
   python direct_run.py --mesh_shape 1x8
   ```

### GSM8K Benchmark Verification

For benchmark verification, use the script in the old_files directory:

```bash
python old_files/verify_gsm8k_scores.py --model_path /path/to/qwen2.5-7b
```

### Additional Verification Scripts

For advanced testing and verification, legacy scripts are available in the old_files directory:

```bash
# Test weight loading
python old_files/test_weight_loading.py --model_path /path/to/qwen2.5-7b

# Verify tensor-parallel implementation with multiple mesh shapes
python old_files/verify_tensor_parallel_bounty.py --mesh_shapes 1x8,2x4

# Run minimal tensor parallel test
python old_files/minimal_tp_test.py --model_path /path/to/qwen2.5-7b
```

### Performance Considerations

1. **Memory Usage**: Using more cores requires more memory. For large mesh shapes like 1x32 or 8x4, ensure your system has sufficient RAM.

2. **Initialization Time**: The first run might take longer as JAX compiles optimized kernels.

3. **CPU Load**: All cores should show activity in system monitoring tools during inference.

4. **Mesh Configuration Trade-offs**:
   - More devices in the model dimension (e.g., 1x8) = more model parallelism (better for larger models)
   - More devices in the batch dimension (e.g., 8x1) = more data parallelism (better for throughput)

## How Tensor Parallelism Works

### Mesh Creation

A 2D mesh with 'batch' and 'model' dimensions is created to support both data parallelism and model parallelism:

```python
def create_device_mesh(mesh_shape):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices, ('batch', 'model'))
```

### Parameter Sharding

Parameters are sharded according to their position in the model:

- **Attention Heads**: Input projections (query, key, value) are sharded along the output dimension
- **MLP Layers**: Gate and up projections are sharded along the output dimension
- **Output Projections**: Sharded along the input dimension

### Communication Patterns

- All-to-all communication happens when switching between model and data parallelism
- All-gather operations combine partial results across the model dimension
- All-reduce operations combine partial results during the forward pass

## Implementation Details

### Weight Loading

The implementation loads Qwen2.5-7B weights from safetensors files:

1. The `load_qwen_weights` function loads weights from safetensors files
2. Weights are sharded according to their partition specs
3. Names are converted from HuggingFace format to Flax format

### TensorParallelDense Implementation

The `TensorParallelDense` class demonstrates how tensor parallelism is implemented:

1. Parameters are sharded according to specified axes
2. `with_sharding_constraint` enforces the sharding
3. JAX's GSPMD automatically handles communication between devices

## Troubleshooting

If you encounter issues:

1. **Not enough devices error**:
   - Set XLA_FLAGS environment variable to simulate the required number of devices

2. **Model weights not found**:
   - Check that the Qwen2.5-7B weights are in the correct location
   - Specify the model path explicitly with `--model_path`

3. **Memory issues**:
   - For large mesh shapes like 1x32 or 8x4, you may need more RAM
   - Try reducing the batch size
   - Try a smaller mesh configuration (e.g., 1x4 instead of 1x8):
     ```bash
     export XLA_FLAGS="--xla_force_host_platform_device_count=4"
     ```

4. **Single Core Usage**: 
   - Verify that XLA_FLAGS is correctly set before running tests
     ```bash
     echo $XLA_FLAGS
     ```

5. **Long Compilation Time**:
   - JAX needs to compile optimized kernels on first run
   - Subsequent runs will be faster as compiled artifacts are cached

## References

- [JAX Device Mesh Documentation](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Qwen2.5 Model Details](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [TT-Forge Documentation](https://github.com/tenstorrent/tt-forge)
- [TT-xla Repository](https://github.com/tenstorrent/tt-xla)