#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GSM8K full evaluation script for Qwen2.5-7B models.
This script loads real weights and runs real GSM8K questions with no fallbacks or dummy implementations.
"""

import os
import sys
import time
import logging
import json
import argparse
import re
import datetime
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Import the model implementation and necessary utilities
from . import (
    load_qwen_config,
    load_qwen_weights,
    Qwen2ForCausalLM
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("GSM8K_TEST")

def load_gsm8k_dataset(max_samples=None):
    """Load the GSM8K test dataset."""
    logger.info("Loading GSM8K test dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if max_samples is not None and max_samples < len(dataset):
        logger.info(f"Limiting to {max_samples} samples as requested")
        dataset = dataset.select(range(max_samples))
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset

def extract_answer(text: str) -> str:
    """Extract the final answer from GSM8K format."""
    # Standard GSM8K format uses '#### number'
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    
    # Extract last number as fallback
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
    
    return ""

def generate_text(
    model, 
    params, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 100,
    temperature: float = 0.0
) -> str:
    """Generate text using real model inference."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs.input_ids)
    attention_mask = jnp.array(inputs.attention_mask)
    
    # Initialize tracking variables
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    generated_tokens = []
    
    # Start time tracking for performance analysis
    start_time = time.time()
    
    # Process tokens auto-regressively
    for i in range(max_new_tokens):
        # Forward pass with the model
        outputs = model.apply(
            params, 
            current_input_ids,
            attention_mask=current_attention_mask
        )
        
        # Extract logits - handle different output types
        if hasattr(outputs, 'logits'):
            # FlaxCausalLMOutput case
            next_token_logits = outputs.logits[:, -1, :]
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # Tuple case
            next_token_logits = outputs[0][:, -1, :]
        else:
            # Direct logits case
            next_token_logits = outputs[:, -1, :]
        
        # Apply temperature if needed
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            # Sample from distribution
            key = jax.random.PRNGKey(int(time.time() * 1000000))
            next_token = jax.random.categorical(key, next_token_logits, axis=-1)
        else:
            # Greedy decoding
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Append to generated sequence
        token_id = next_token[0].item()
        generated_tokens.append(token_id)
        
        # Add to sequence
        next_token = next_token.reshape(-1, 1)
        current_input_ids = jnp.concatenate([current_input_ids, next_token], axis=1)
        token_attention = jnp.ones_like(next_token)
        current_attention_mask = jnp.concatenate([current_attention_mask, token_attention], axis=1)
        
        # Log progress for longer generations
        if (i+1) % 10 == 0:
            tokens_per_sec = (i+1) / (time.time() - start_time)
            logger.info(f"Generated {i+1} tokens ({tokens_per_sec:.2f} tokens/sec)")
        
        # Check for EOS token
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info(f"Reached EOS token after {i+1} tokens")
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def evaluate_accuracy(expected_answer: str, generated_answer: str) -> bool:
    """Evaluate if the generated answer matches the expected answer."""
    expected_num = extract_answer(expected_answer)
    generated_num = extract_answer(generated_answer)
    
    if not expected_num or not generated_num:
        return False
    
    return expected_num == generated_num

def fix_parameter_shapes(flat_params):
    """
    Fix parameter shapes by transposing weight matrices that have incorrect shapes.
    
    In JAX/Flax, linear layer weights are expected to be (in_dim, out_dim), but some
    pretrained models may have them in the transposed format (out_dim, in_dim).
    """
    fixed_params = {}
    
    # Scan parameters to understand the proper shapes 
    hidden_size = None
    intermediate_size = None
    
    # First determine model dimensions from parameter shapes
    for name, param in flat_params.items():
        if 'embed_tokens/embedding' in name:
            # vocab_size × hidden_size
            hidden_size = param.shape[1]
        elif 'mlp/gate_proj/kernel' in name:
            # Should be hidden_size × intermediate_size
            intermediate_size = max(param.shape)
            hidden_size = min(param.shape)
    
    logger.info(f"Detected model dimensions - hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
    
    # Now fix parameter shapes
    for name, param in flat_params.items():
        # Fix attention parameters (q_proj, k_proj, v_proj should be hidden_size × q_dim)
        if 'kernel' in name and any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
            # Input projection - should be (hidden_size, projection_dim)
            # If it's in the wrong format, transpose it
            if len(param.shape) == 2 and param.shape[1] < param.shape[0]:
                logger.info(f"Transposing parameter {name} from {param.shape} to {param.shape[::-1]}")
                fixed_params[name] = param.T
            else:
                fixed_params[name] = param
                
        # Fix output projections (o_proj should be q_dim × hidden_size)
        elif 'kernel' in name and 'o_proj' in name:
            # Output projection - should be (q_dim, hidden_size)
            # If it's in the wrong format, transpose it
            if len(param.shape) == 2 and hidden_size is not None and param.shape[0] != hidden_size:
                logger.info(f"Transposing parameter {name} from {param.shape} to {param.shape[::-1]}")
                fixed_params[name] = param.T
            else:
                fixed_params[name] = param
                
        # Fix MLP parameters
        elif 'kernel' in name and any(x in name for x in ['gate_proj', 'up_proj']):
            # MLP projections - should be (hidden_size, intermediate_size)
            # If dimensions are wrong, transpose
            if len(param.shape) == 2 and param.shape[0] > param.shape[1]:
                logger.info(f"Transposing parameter {name} from {param.shape} to {param.shape[::-1]}")
                fixed_params[name] = param.T
            else:
                fixed_params[name] = param
                
        # Fix down projection
        elif 'kernel' in name and 'down_proj' in name:
            # Down projection - should be (intermediate_size, hidden_size)
            # If dimensions are wrong, transpose
            if len(param.shape) == 2 and hidden_size is not None and intermediate_size is not None:
                # Check if the shape is incorrect
                if param.shape[0] != intermediate_size or param.shape[1] != hidden_size:
                    logger.info(f"Transposing parameter {name} from {param.shape} to {param.shape[::-1]}")
                    fixed_params[name] = param.T
                else:
                    fixed_params[name] = param
            else:
                fixed_params[name] = param
        else:
            # Keep other parameters as is
            fixed_params[name] = param
    
    return fixed_params

def main():
    """Main function for running GSM8K evaluation."""
    parser = argparse.ArgumentParser(description="GSM8K evaluation for Qwen2.5 model")
    parser.add_argument("--weights_path", type=str, required=True,
                      help="Path to the model weights directory")
    parser.add_argument("--max_samples", type=int, default=10,
                      help="Maximum number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                      help="Sampling temperature (0 for greedy)")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Path to save results")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                      choices=["float32", "float16", "bfloat16"],
                      help="Data type for model parameters")
    args = parser.parse_args()
    
    # Verify weights path exists
    if not os.path.exists(args.weights_path):
        logger.error(f"Weights path {args.weights_path} does not exist")
        return 1
    
    # Set data type based on argument
    if args.dtype == "float32":
        dtype = jnp.float32
    elif args.dtype == "float16":
        dtype = jnp.float16
    else:
        dtype = jnp.bfloat16
    
    # Load model configuration
    logger.info(f"Loading model configuration from {args.weights_path}")
    config = load_qwen_config(args.weights_path)
    
    # Create the model using the configuration
    logger.info("Creating model...")
    model = Qwen2ForCausalLM(config=config, dtype=dtype, param_dtype=dtype)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.weights_path, trust_remote_code=True)
    
    # Load the weights
    logger.info("Loading model weights...")
    weights_start_time = time.time()
    
    # Load weights directly from safetensors files
    flat_params = load_qwen_weights(
        model_path=args.weights_path,
        config=config,
        param_dtype=dtype
    )
    
    # Fix parameter shapes
    logger.info("Fixing parameter shapes...")
    flat_params = fix_parameter_shapes(flat_params)
    
    # Convert flat parameters to nested structure for Flax
    from flax.traverse_util import unflatten_dict
    params = unflatten_dict(flat_params, sep='/')
    
    logger.info(f"Weights loaded in {time.time() - weights_start_time:.2f}s")
    
    # Load the GSM8K dataset
    dataset = load_gsm8k_dataset(max_samples=args.max_samples)
    
    # Evaluate on the dataset
    logger.info("Starting evaluation...")
    results = []
    correct_count = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        expected_answer = example["answer"]
        
        logger.info(f"\n==== Example {i+1}/{len(dataset)} ====")
        logger.info(f"Question: {question}")
        
        # Create prompt
        prompt = f"Question: {question}\n\nAnswer:"
        
        # Generate answer
        logger.info("Generating answer...")
        start_time = time.time()
        generated_answer = generate_text(
            model=model,
            params={"params": params},  # Wrap in params collection as required by Flax
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        generation_time = time.time() - start_time
        
        # Check correctness
        is_correct = evaluate_accuracy(expected_answer, generated_answer)
        if is_correct:
            correct_count += 1
        
        logger.info(f"Generated answer ({generation_time:.2f}s): {generated_answer}")
        logger.info(f"Expected answer: {extract_answer(expected_answer)}")
        logger.info(f"Correct: {is_correct}")
        
        # Save result
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "generation_time": generation_time
        })
    
    # Calculate accuracy
    accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0
    logger.info(f"\n===== EVALUATION RESULTS =====")
    logger.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(dataset)})")
    
    # Save results if requested
    if args.output_file:
        output_path = args.output_file
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"gsm8k_results_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(dataset),
            "model_path": args.weights_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 