#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This script runs the tensor parallel tests for Qwen2.5-7B model
# Uses 8 CPU cores for tensor parallelism with a 1x8 mesh

# Set model path - adjust if needed
MODEL_PATH="/Users/lu/Documents/tt-bounty-1/qwen2.5-7b"

# Set environment variables for multi-core processing
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
echo "Setting up environment to use 8 CPU cores for tensor parallelism..."
echo "XLA_FLAGS: $XLA_FLAGS"

# Print system info
echo "Running on $(uname -s) with $(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 'unknown number of') logical CPUs"
echo "Using model path: $MODEL_PATH"

# Ensure model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Function to run a test and check result
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo ""
    echo "=================================="
    echo "Running test: $test_name"
    echo "=================================="
    echo "Command: $test_command"
    echo ""
    
    # Run the test
    eval "$test_command"
    local result=$?
    
    if [ $result -eq 0 ]; then
        echo "✅ Test $test_name PASSED"
    else
        echo "❌ Test $test_name FAILED (exit code $result)"
    fi
    
    return $result
}

# Run the minimal tensor parallel test (1x8 mesh)
run_test "Minimal Tensor Parallel Test" "python3 minimal_tp_test.py --model_path \"$MODEL_PATH\""

# Run the debug test with full model
run_test "Full Model Debug Test" "python3 debug_test.py --model_path \"$MODEL_PATH\""

# Run GSM8K verification test with a few samples
run_test "GSM8K Verification" "python3 verify_gsm8k_scores.py --model_path \"$MODEL_PATH\" --num_samples 3 --mesh_shape 1,8"

echo ""
echo "All tests completed!" 