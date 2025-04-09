# Multi-Core Processing for Qwen2.5-7B JAX Implementation

This document explains the changes made to utilize multiple CPU cores for tensor parallelism in the Qwen2.5-7B JAX implementation.

## Key Changes

1. **CPU Utilization**: Modified test scripts to use a 1x8 device mesh instead of 1x1, enabling processing across 8 CPU cores simultaneously.

2. **XLA_FLAGS Configuration**: Set `XLA_FLAGS="--xla_force_host_platform_device_count=8"` to ensure JAX correctly simulates 8 devices even on systems with fewer physical cores.

3. **Full Model Usage**: Removed reduced layer counts that were used in debug tests, ensuring full model accuracy when generating content.

4. **Mesh Shape Configuration**: Changed all mesh shapes from (1, 1) to (1, 8) to properly distribute computation across cores.

5. **Generation Length**: Increased token generation limits to allow for more complete answers.

## Running the Tests

Use the provided `run_tests.sh` script to run all the updated tests:

```bash
cd tt-xla/tests/jax/models/qwen2_5
bash run_tests.sh
```

This script will:
- Configure the environment to use 8 CPU cores
- Run the minimal tensor parallel test
- Run the full model debug test
- Run the GSM8K verification test

## Manual Testing

You can also run individual tests with specific parameters:

```bash
# Set environment variable for multi-core processing
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# Run minimal test
python3 minimal_tp_test.py --model_path /path/to/qwen2.5-7b

# Run debug test with full model
python3 debug_test.py --model_path /path/to/qwen2.5-7b

# Run GSM8K verification
python3 verify_gsm8k_scores.py --model_path /path/to/qwen2.5-7b --num_samples 5
```

## Performance Considerations

1. **Memory Usage**: Using 8 cores requires more memory than a single core, ensure your system has sufficient RAM.

2. **Initialization Time**: The first run might take longer as JAX compiles optimized kernels.

3. **CPU Load**: All 8 cores should show activity in system monitoring tools during inference.

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Reduce the mesh size (e.g., try 1x4 instead of 1x8)
   ```bash
   export XLA_FLAGS="--xla_force_host_platform_device_count=4"
   ```

2. **Single Core Usage**: Verify that XLA_FLAGS is correctly set before running tests
   ```bash
   echo $XLA_FLAGS
   ```

3. **Model Path**: Ensure the model path is correctly set to the Qwen2.5-7B weights location 