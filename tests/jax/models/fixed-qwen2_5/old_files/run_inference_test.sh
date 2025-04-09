#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to run inference tests on Qwen2.5-7B model with different mesh shapes
# This script sets up the environment and runs the interactive inference script

# Default values
MODEL_PATH="../../../../qwen2.5-7b"
MESH_SHAPE="1x8"
RUN_GSM8K=false
MAX_NEW_TOKENS=20
DEBUG=false
DEMO_MODEL=false

# Function to display usage
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --model-path PATH   Path to model weights (default: ../../../../qwen2.5-7b)"
  echo "  -s, --mesh-shape SHAPE  Mesh shape in format AxB (default: 1x8)"
  echo "                          Supported shapes: 1x1, 2x4, 1x8, 4x2"
  echo "  -g, --gsm8k             Run GSM8K benchmark instead of interactive chat"
  echo "  -t, --tokens NUM        Maximum new tokens to generate (default: 20)"
  echo "  -d, --debug             Enable debug mode with verbose output"
  echo "  --demo                  Use demo model instead of loading real weights"
  echo "  -h, --help              Display this help message"
  echo ""
  echo "Examples:"
  echo "  $0                      Run with default settings (1x8 mesh)"
  echo "  $0 -s 2x4               Run with 2x4 mesh shape"
  echo "  $0 -g -s 1x1            Run GSM8K benchmark with 1x1 mesh shape"
  echo "  $0 -m /path/to/model    Use custom model path"
  echo "  $0 --demo               Use demo model with random weights"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    -s|--mesh-shape)
      MESH_SHAPE="$2"
      shift 2
      ;;
    -g|--gsm8k)
      RUN_GSM8K=true
      shift
      ;;
    -t|--tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    -d|--debug)
      DEBUG=true
      shift
      ;;
    --demo)
      DEMO_MODEL=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Check if model path exists (unless using demo model)
if [ "$DEMO_MODEL" = false ] && [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path $MODEL_PATH does not exist"
  exit 1
fi

# Parse mesh shape to determine number of devices needed
read -r ROWS COLS <<< $(echo $MESH_SHAPE | tr 'x' ' ')
NUM_DEVICES=$((ROWS * COLS))

if [[ $NUM_DEVICES -lt 1 ]]; then
  echo "Error: Invalid mesh shape $MESH_SHAPE. Must have at least one device."
  exit 1
fi

# Setup environment for device simulation
echo "==== Setup Environment ===="
echo "Setting up environment for $MESH_SHAPE mesh shape ($NUM_DEVICES devices)"
export XLA_FLAGS="--xla_force_host_platform_device_count=$NUM_DEVICES"

# Print current settings
echo ""
echo "==== Inference Settings ===="
echo "- Model path: $MODEL_PATH"
echo "- Mesh shape: $MESH_SHAPE ($ROWS×$COLS)"
echo "- GSM8K benchmark: $RUN_GSM8K"
echo "- Max new tokens: $MAX_NEW_TOKENS"
echo "- Debug mode: $DEBUG"
echo "- Demo model: $DEMO_MODEL"
echo "- XLA_FLAGS: $XLA_FLAGS"
echo ""

# First test if the weight loading works
echo "==== Testing Weight Loading ===="
if [ "$DEMO_MODEL" = false ]; then
  echo "Testing weight loading from $MODEL_PATH..."
  python test_weight_loading.py --model_path="$MODEL_PATH" > /tmp/weight_loading_test.log 2>&1
  if [ $? -ne 0 ]; then
    echo "❌ Weight loading test failed!"
    echo "See error details below:"
    cat /tmp/weight_loading_test.log
    echo ""
    echo "Try using --demo flag to use a demo model instead."
    exit 1
  else
    echo "✅ Weight loading test passed!"
  fi
else
  echo "Skipping weight loading test (using demo model)"
fi

# Build command arguments
COMMON_ARGS=""
if [ "$DEMO_MODEL" = true ]; then
  COMMON_ARGS="$COMMON_ARGS --use_demo_model"
else
  COMMON_ARGS="$COMMON_ARGS --model_path=\"$MODEL_PATH\""
fi
COMMON_ARGS="$COMMON_ARGS --mesh_shape=\"$MESH_SHAPE\""
if [ "$DEBUG" = true ]; then
  COMMON_ARGS="$COMMON_ARGS --debug"
fi

# Run the appropriate script
if [ "$RUN_GSM8K" = true ]; then
  # Run GSM8K benchmark
  echo ""
  echo "==== Running GSM8K Benchmark ===="
  echo "This will validate the model on math reasoning problems"
  
  # The maximum number of samples to test (lower for faster results)
  NUM_SAMPLES=5
  GSM_ARGS="$COMMON_ARGS --num_samples=$NUM_SAMPLES --max_tokens=$MAX_NEW_TOKENS"
  
  echo "Command: python verify_gsm8k_scores.py $GSM_ARGS"
  echo "Starting benchmark (this may take several minutes)..."
  echo ""
  
  eval "python verify_gsm8k_scores.py $GSM_ARGS"
  GSM_RESULT=$?
  
  if [ $GSM_RESULT -ne 0 ]; then
    echo "❌ GSM8K benchmark failed with exit code $GSM_RESULT"
    exit $GSM_RESULT
  fi
else
  # Run interactive inference
  echo ""
  echo "==== Running Interactive Inference ===="
  echo "This will start an interactive chat session with the model"
  
  INF_ARGS="$COMMON_ARGS --max_tokens=$MAX_NEW_TOKENS"
  
  echo "Command: python interactive_inference.py $INF_ARGS"
  echo "Starting inference (press Ctrl+C to exit)..."
  echo ""
  
  eval "python interactive_inference.py $INF_ARGS"
  INF_RESULT=$?
  
  if [ $INF_RESULT -ne 0 ]; then
    echo "❌ Interactive inference failed with exit code $INF_RESULT"
    exit $INF_RESULT
  fi
fi

echo ""
echo "==== Test Completed Successfully ====" 