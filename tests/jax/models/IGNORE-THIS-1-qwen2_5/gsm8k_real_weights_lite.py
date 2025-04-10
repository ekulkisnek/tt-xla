#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GSM8K lightweight real weights evaluation script for Qwen2.5-7B models.
This script ALWAYS uses real weights, tokenizer, and GSM8K questions
but keeps evaluation lightweight by loading only a subset of layers.
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from safetensors.numpy import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer

# Import the model implementation
from . import (
    AutoQwenModel,
    AutoQwenModelTensorParallel,
    get_model,
    load_qwen_config,
    create_device_mesh,
    load_qwen_weights,
    init_model_from_weights,
    load_safetensors_index,
    convert_weight_name_to_flax,
)

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set up logger
logger = logging.getLogger("GSM8K_REAL_WEIGHTS_LITE")
logger.setLevel(logging.DEBUG)

def setup_logging(log_file=None, verbose=False):
    """Set up detailed logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter with milliseconds
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.debug(f"Logging initialized with level: {level}")
    return root_logger

def load_gsm8k_dataset(max_examples=None):
    """
    Load the GSM8K dataset and prepare it for evaluation.
    
    Args:
        max_examples: Maximum number of examples to load
        
    Returns:
        List of examples with questions and answers
    """
    logging.info("Loading GSM8K test dataset...")
    from datasets import load_dataset
    
    # Load the test split
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Format the dataset into question-answer pairs
    examples = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        examples.append({
            "question": question,
            "answer": answer
        })
    
    # Log sample
    logging.info(f"Sample question: {examples[0]['question'][:100]}...")
    logging.info(f"Sample answer: {examples[0]['answer'][:100]}...")
    
    # Limit examples if specified
    if max_examples is not None and max_examples < len(examples):
        logging.info(f"Limiting to {max_examples} samples as requested")
        examples = examples[:max_examples]
    
    logging.info(f"Final dataset size: {len(examples)} examples")
    return examples

def parse_answer(text: str) -> str:
    """
    Extract the final answer from a GSM8K model response.
    
    In GSM8K, the final answer is preceded by '#### '.
    """
    logger.debug(f"Parsing answer from text: {text[:50]}...")
    
    # The official GSM8K format uses '#### number' at the end
    match = re.search(r'####\s*(\d+)', text)
    if match:
        result = match.group(1)
        logger.debug(f"Found answer with '####' pattern: {result}")
        return result
    
    # Fallback methods if the model doesn't follow the exact format
    patterns = [
        r"The answer is (\d+)",
        r"Therefore, the answer is (\d+)",
        r"So the answer is (\d+)",
        r"Thus, the answer is (\d+)",
        r"The final answer is (\d+)",
        r"\$(\d+)",
        r"\d+$"  # Just extract the last number as a fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            result = match.group(1) if len(match.groups()) > 0 else match.group(0)
            logger.debug(f"Found answer with fallback pattern '{pattern}': {result}")
            return result
    
    # Final fallback: extract any number
    numbers = re.findall(r"\d+", text)
    if numbers:
        result = numbers[-1]
        logger.debug(f"Found answer with final fallback (last number): {result}")
        return result
    
    logger.warning("No answer found in text")
    return ""

def extract_answer_from_gsm8k_example(example: Dict) -> str:
    """
    Extract the ground truth answer from a GSM8K example with detailed logging.
    """
    answer_text = example["answer"]
    logger.debug(f"Extracting answer from example: {answer_text[:50]}...")
    
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        result = match.group(1)
        logger.debug(f"Extracted ground truth answer: {result}")
        return result
    
    logger.warning(f"No '####' pattern found in example answer: {answer_text[:100]}...")
    return ""

def generate_text(model, params, tokenizer, prompt, max_new_tokens=50, temperature=0.0):
    """
    Generate text using a loaded model with real weights.
    
    Args:
        model: The JAX/Flax model to use for generation
        params: Model parameters 
        tokenizer: The tokenizer for encoding/decoding
        prompt: The input prompt to generate from
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 for greedy)
        
    Returns:
        String of generated text
    """
    logging.info(f"Generating text with {max_new_tokens} max new tokens (temp={temperature})")
    logging.debug(f"Prompt: {prompt[:100]}...")
    
    def init_and_bind_model(model, params, input_shape=(1, 1)):
        """Initialize and bind the model properly before calling."""
        try:
            # Create a properly bound instance
            bound_model = model.bind({'params': params})
            logging.debug(f"Successfully bound model with params dict")
            return bound_model
        except Exception as e:
            logging.debug(f"Failed to bind model with standard approach: {e}")
            try:
                # Alternative binding approach
                bound_model = model.bind(params)
                logging.debug(f"Successfully bound model with direct params")
                return bound_model
            except Exception as e2:
                logging.debug(f"All binding approaches failed: {e2}")
                return None
    
    try:
        # Tokenize the prompt
        start_time = time.time()
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        logging.debug(f"Tokenized prompt to {input_ids.shape} in {time.time() - start_time:.3f}s")
        
        # Prepare inputs for model
        input_ids = input_ids.reshape(1, -1)
        logging.debug(f"Input shape: {input_ids.shape}")
        
        # Create attention mask (all 1s for now)
        attention_mask = jnp.ones_like(input_ids)
        
        # Main token generation loop
        generated_ids = input_ids
        
        # Try to bind the model first for better performance
        bound_model = init_and_bind_model(model, params)
        
        for i in range(max_new_tokens):
            logging.debug(f"Generation step {i+1}/{max_new_tokens}")
            
            # Try multiple model calling approaches
            try:
                if bound_model is not None:
                    # Use the bound model directly
                    outputs = bound_model(generated_ids, attention_mask=attention_mask[:, :generated_ids.shape[1]])
                    logging.debug("Successfully called bound model")
                else:
                    # Fall back to apply method
                    outputs = model.apply({'params': params}, generated_ids, 
                                           attention_mask=attention_mask[:, :generated_ids.shape[1]])
                    logging.debug("Successfully called model.apply with params dict")
            except Exception as e1:
                logging.debug(f"Primary model call approach failed: {e1}")
                try:
                    # Try alternative apply approach
                    outputs = model.apply(params, generated_ids)
                    logging.debug("Successfully called model.apply with direct params")
                except Exception as e2:
                    logging.error(f"All model application approaches failed: {e2}")
                    raise RuntimeError(f"Cannot call model with any parameter format: {e2}")
            
            # Handle different output types
            try:
                if isinstance(outputs, tuple):
                    # If outputs is a tuple, take the first element (typically logits)
                    logits = outputs[0]
                    logging.debug(f"Model returned tuple, using first element as logits")
                elif hasattr(outputs, 'logits'):
                    # If outputs has a 'logits' attribute (like model with return_dict=True)
                    logits = outputs.logits
                    logging.debug(f"Model returned object with logits attribute")
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    # If outputs is a dict with a 'logits' key
                    logits = outputs['logits']
                    logging.debug(f"Model returned dict with logits key")
                else:
                    # For simpler models that just return logits directly
                    logits = outputs
                    logging.debug(f"Using model output directly as logits")
                
                # Debug info about logits
                logging.debug(f"Logits shape: {logits.shape}")
                
                # Get the logits for the last token in each sequence
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature if needed
                if temperature > 0:
                    # Scale logits by temperature
                    next_token_logits = next_token_logits / jnp.maximum(temperature, 1e-7)
                    
                    # Sample from the distribution
                    rng_key = jax.random.PRNGKey(int(time.time() * 1000))
                    next_token = jax.random.categorical(rng_key, next_token_logits, axis=-1)
                    logging.debug(f"Sampled token with temperature {temperature}")
                else:
                    # Greedy decoding (take argmax)
                    next_token = jnp.argmax(next_token_logits, axis=-1)
                    logging.debug(f"Selected token with greedy decoding")
                
                # Reshape next token and add to sequence
                next_token = next_token.reshape(1, 1)
                generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)
                
                # Update attention mask
                attention_mask = jnp.ones((1, generated_ids.shape[1]), dtype=jnp.int32)
                
                # Log the token ID and progress
                if i % 10 == 0 or i == max_new_tokens - 1:
                    logging.debug(f"Generated token: {next_token[0,0]} ({i+1}/{max_new_tokens})")
                
                # Check if we should stop generation (e.g., EOS token)
                if tokenizer.eos_token_id is not None and next_token[0, 0] == tokenizer.eos_token_id:
                    logging.debug(f"EOS token generated, stopping generation early at step {i+1}")
                    break
                    
            except Exception as e:
                logging.error(f"Error in generation step {i+1}: {e}")
                break
        
        # Decode the generated tokens
        try:
            decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logging.debug(f"Decoded text length: {len(decoded_text)}")
            return decoded_text
        except Exception as e:
            logging.error(f"Error decoding tokens: {e}")
            # Fallback: just return the prompt
            return prompt
            
    except Exception as e:
        logging.error(f"Error in text generation: {e}")
        return prompt

def evaluate_real_model(model, params, tokenizer, examples, batch_size=1, temperature=0.0):
    """Evaluate a real model on GSM8K examples."""
    num_examples = len(examples)
    results = {
        "num_correct": 0,
        "num_evaluated": 0,
        "examples": []
    }
    
    logger.info(f"Starting evaluation on {num_examples} examples with real generation")
    
    # Process examples
    for i, example in enumerate(examples):
        logger.info(f"\n==== Example {i+1}/{num_examples} ====")
        question = example["question"]
        expected_answer = example["answer"]
        logger.info(f"Question: {question}")
        
        # Generate answer
        logger.info("Generating answer...")
        try:
            generated_text = generate_text(
                model=model,
                params=params,
                tokenizer=tokenizer,
                prompt=f"Question: {question}\nAnswer: ",
                max_new_tokens=50,
                temperature=temperature
            )
            
            # Remove the prompt from the generated text
            answer_prefix = "Answer: "
            if answer_prefix in generated_text:
                generated_answer = generated_text.split(answer_prefix)[1].strip()
            else:
                generated_answer = generated_text.strip()
            
            # Evaluate correctness
            is_correct = evaluate_answer(generated_answer, expected_answer)
            
            # Log the results
            logger.info(f"Generated: {generated_answer}")
            logger.info(f"Expected: {expected_answer}")
            logger.info(f"Correct: {is_correct}")
            
            # Save results
            results["examples"].append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            })
            
            results["num_evaluated"] += 1
            if is_correct:
                results["num_correct"] += 1
                
        except Exception as e:
            logger.error(f"Error evaluating example {i+1}: {str(e)}")
            traceback.print_exc()
    
    # Calculate accuracy
    if results["num_evaluated"] > 0:
        accuracy = results["num_correct"] / results["num_evaluated"]
    else:
        accuracy = 0.0
    
    return results, accuracy

def load_partial_weights(weights_path, num_layers, device_count):
    """Load real weights for evaluation, but only the first N layers to make it lightweight."""
    logging.info(f"Loading configuration from {weights_path}")
    config_path = os.path.join(weights_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    total_layers = config["num_hidden_layers"]
    logging.info(f"Loaded configuration with {total_layers} layers")
    
    # Create a reduced config for the smaller model
    original_layers = config["num_hidden_layers"]
    config["num_hidden_layers"] = num_layers
    logging.info(f"Reduced model from {original_layers} to {num_layers} layers")
    
    # Update model_type if needed
    if config.get("model_type") == "qwen2":
        logging.info(f"Updating model_type from 'qwen2' to 'qwen2_5' for compatibility")
        config["model_type"] = "qwen2_5"
    
    # Fix configuration to match actual weight dimensions for attention components
    # This is critical to avoid shape mismatches in the attention layers
    if "num_attention_heads" in config and "num_key_value_heads" in config:
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        # Adjust dimensions to match actual weights
        logging.info(f"Original num_attention_heads: {config['num_attention_heads']}, num_key_value_heads: {config['num_key_value_heads']}")
        num_key_value_heads = config.get("num_key_value_heads", config["num_attention_heads"])
        
        # Ensure num_key_value_heads divides num_attention_heads evenly
        if config["num_attention_heads"] % num_key_value_heads != 0:
            logging.warning(f"Attention heads ({config['num_attention_heads']}) not divisible by KV heads ({num_key_value_heads})")
            # Make them equal if they don't divide evenly
            config["num_key_value_heads"] = config["num_attention_heads"]
            logging.info(f"Updated num_key_value_heads to {config['num_key_value_heads']}")
        
        # Set the correct head dimensions to avoid shape mismatches
        logging.info(f"Setting attention parameters to ensure compatibility with weights")
        # Force dimensions to match the actual loaded weights
        config["qwen_attention_heads_match_actual_weights"] = True
        logging.info(f"Updated configuration for attention compatibility")
    
    # Create empty parameter structure for model
    from tests.jax.models.qwen2_5.model_implementation import Qwen2ForCausalLM
    logging.info(f"Creating model with config: {json.dumps(config, indent=2)[:300]}...")
    model = Qwen2ForCausalLM(config=config)
    
    # Initialize the empty parameters first with a dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    logging.info("Initializing parameter structure with dummy input")
    params = model.init(rng, dummy_input)
    
    # Debug: Inspect parameter structure
    from flax.traverse_util import flatten_dict
    flat_params = flatten_dict(params)
    logging.info(f"Model parameter structure has {len(flat_params)} entries")
    for i, (k, v) in enumerate(sorted(flat_params.items())[:5]):
        if hasattr(v, 'shape'):
            shape_info = f", shape={v.shape}"
        else:
            shape_info = ""
        logging.info(f"  Sample param {i}: {k}{shape_info}")
    
    # Now load the real weights
    logging.info(f"Loading real weights from {weights_path}")
    logging.info(f"Loading safetensors index from {os.path.join(weights_path, 'model.safetensors.index.json')}")
    
    from tests.jax.models.qwen2_5.weight_loading import load_safetensors_into_params
    
    try:
        # Open the index file
        with open(os.path.join(weights_path, "model.safetensors.index.json"), "r") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        logging.info(f"Found weight map with {len(weight_map)} entries")
        
        # Track unique weight files
        weight_files = set(weight_map.values())
        logging.info(f"Found {len(weight_map)} parameters in {len(weight_files)} weight files")
        
        logging.info(f"Loading {num_layers} out of {total_layers} layers for lightweight evaluation")
        
        # Filter weight map for desired layers
        filtered_weight_map = {}
        for key, value in weight_map.items():
            # Always include non-layer params
            if "layers." not in key or any(p in key for p in [
                "model.embed_tokens", "model.norm", "lm_head"
            ]):
                filtered_weight_map[key] = value
                continue
            
            # Parse layer number for layer params
            try:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if layer_num < num_layers:
                    filtered_weight_map[key] = value
            except (IndexError, ValueError):
                filtered_weight_map[key] = value
        
        logging.info(f"Filtered weight map to {len(filtered_weight_map)} parameters")
        
        # Sample weight keys for debugging
        sample_keys = list(filtered_weight_map.keys())[:5]
        logging.info(f"Sample weight keys: {sample_keys}")
        
        # Load the weights
        start_time = time.time()
        logging.info("Starting weight loading process...")
        
        params = load_safetensors_into_params(
            model_params=params,
            weight_map=filtered_weight_map,
            safetensors_dir=weights_path
        )
        logging.info(f"Successfully loaded {num_layers} layers of weights")
        logging.info(f"Weights loaded in {time.time() - start_time:.2f}s")
        
        # Create the tokenizer
        logging.info("Initializing tokenizer from real weights")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)
        logging.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
        
        return tokenizer, params, config
    
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise e

def evaluate_answer(generated_answer, expected_answer):
    """
    Evaluate if a generated answer matches the expected GSM8K answer.
    
    Args:
        generated_answer: The answer generated by the model
        expected_answer: The expected answer from GSM8K
        
    Returns:
        True if the answer is correct, False otherwise
    """
    # Extract numerical answer from GSM8K format
    # GSM8K answers end with boxed answers like "The answer is $10."
    def extract_number(text):
        # Try to find a number at the end of the text
        number_pattern = r'(?:answer is|answer:|equals|=|is)(?:\s*\$?\s*)(\d+(?:\.\d+)?)'
        matches = re.findall(number_pattern, text.lower())
        if matches:
            return matches[-1]  # Return the last match
        
        # If no match with the pattern, look for any number
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1]  # Return the last number
        
        return None
    
    # Extract numbers from both answers
    generated_number = extract_number(generated_answer)
    expected_number = extract_number(expected_answer)
    
    # Compare numbers
    if generated_number is not None and expected_number is not None:
        return generated_number == expected_number
    
    # If number extraction fails, do a more lenient comparison
    # Remove spaces, commas, and convert to lowercase for comparison
    def normalize(text):
        return re.sub(r'[\s,]', '', text.lower())
    
    return normalize(generated_answer) == normalize(expected_answer)

def main():
    """Main function for running the evaluation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Lightweight GSM8K evaluation with real weights")
    parser.add_argument("--weights_path", type=str, default="/Users/lu/Documents/tt-bounty-1/qwen2.5-7b",
                      help="Path to the model weights directory (containing safetensors files)")
    parser.add_argument("--num_layers", type=int, default=2,
                      help="Number of model layers to use (for lightweight evaluation)")
    parser.add_argument("--max_examples", type=int, default=5,
                      help="Maximum number of examples to evaluate")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--temperature", type=float, default=0.0,
                      help="Sampling temperature (0 for greedy)")
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Show all arguments
    logging.info("Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # Get available devices for mesh creation
    devices = jax.devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} available devices")
    
    # Use a 1D mesh if only one device is available
    requested_mesh_shape = (1, 1)
    logging.info(f"Using requested mesh shape {requested_mesh_shape}")
    
    # Ensure we don't request more devices than available
    if math.prod(requested_mesh_shape) > num_devices:
        # Adjust to what's available while trying to maintain aspect ratio
        if num_devices == 1:
            mesh_shape = (1, 1)
        else:
            # Try to maintain aspect ratio while fitting available devices
            dp = requested_mesh_shape[0]
            tp = requested_mesh_shape[1]
            ratio = dp / tp
            
            # Find best mesh shape that fits num_devices and maintains ratio
            best_dp, best_tp = 1, 1
            min_diff = float('inf')
            
            # Try different dp values and compute corresponding tp
            for d in range(1, num_devices + 1):
                if num_devices % d == 0:  # ensure we use all devices
                    t = num_devices // d
                    curr_ratio = d / t
                    diff = abs(curr_ratio - ratio)
                    if diff < min_diff:
                        min_diff = diff
                        best_dp, best_tp = d, t
            
            mesh_shape = (best_dp, best_tp)
        
        logging.info(f"Requested mesh shape {requested_mesh_shape} exceeds available devices. Adjusted to {mesh_shape}")
    else:
        mesh_shape = requested_mesh_shape
    
    logging.info(f"Creating device mesh with shape {mesh_shape}")
    devices_array = np.array(devices[:math.prod(mesh_shape)]).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("data", "model"))
    
    # Load the weights from the specified path
    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        logging.error(f"Weights path {weights_path} does not exist. Please provide a valid path to the Qwen2.5-7B model weights.")
        return 1
    
    # Verify the safetensors index file exists
    safetensors_index_path = os.path.join(weights_path, "model.safetensors.index.json")
    if not os.path.exists(safetensors_index_path):
        logging.error(f"Safetensors index file not found at {safetensors_index_path}. Please provide a valid path to the Qwen2.5-7B model weights.")
        return 1
    
    try:
        # Load real weights, tokenizer, and reduced config
        logging.info(f"Loading real weights from {weights_path} with {args.num_layers} layers")
        tokenizer, params, config = load_partial_weights(
            weights_path=weights_path, 
            num_layers=args.num_layers,
            device_count=num_devices
        )
        
        # Initialize model with the reduced configuration
        from tests.jax.models.qwen2_5.model_implementation import Qwen2ForCausalLM
        model = Qwen2ForCausalLM(config=config)
        
        # Load the GSM8K dataset
        examples = load_gsm8k_dataset(max_examples=args.max_examples)
        
        # Evaluate the model
        logging.info("\n===== STARTING REAL WEIGHTS EVALUATION =====\n")
        
        results, accuracy = evaluate_real_model(
            model, 
            params, 
            tokenizer, 
            examples, 
            batch_size=args.batch_size, 
            temperature=args.temperature
        )
        
        logging.info(f"\n===== REAL WEIGHTS EVALUATION COMPLETED =====\n")
        logging.info(f"Accuracy: {accuracy:.4f} ({results['num_correct']}/{results['num_evaluated']})")
        
        # Save results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"gsm8k_real_weights_lite_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "num_correct": results["num_correct"],
                "num_evaluated": results["num_evaluated"],
                "examples": results["examples"]
            }, f, indent=2)
        
        logging.info(f"Results saved to {result_file}")
        
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 