#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GSM8K real evaluation script for Qwen2.5-7B models.
This script loads real weights, uses real GSM8K questions, and performs actual
text generation with detailed logging.
"""

import os
import sys
import time
import logging
import json
import argparse
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import datetime

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Import the model implementation
from . import (
    AutoQwenModel,
    AutoQwenModelTensorParallel,
    get_model,
    load_qwen_config,
    get_qwen2_7b_config,
    create_device_mesh,
    load_qwen_weights,
    init_model_from_weights
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
logger = logging.getLogger("GSM8K_REAL_EVAL")
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

def load_gsm8k_dataset(split="test", max_samples=None):
    """Load GSM8K dataset with detailed logging."""
    logger.info(f"Loading GSM8K {split} dataset...")
    start_time = time.time()
    
    try:
        dataset = load_dataset("gsm8k", "main", split=split)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(dataset)} examples from GSM8K {split} set in {load_time:.2f}s")
        
        # Print sample if available
        if len(dataset) > 0:
            logger.info(f"Sample question: {dataset[0]['question'][:100]}...")
            logger.info(f"Sample answer: {dataset[0]['answer'][:100]}...")
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(dataset):
            logger.info(f"Limiting to {max_samples} samples as requested")
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load GSM8K dataset: {str(e)}")
        raise

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

def handle_calculation_annotations(text: str) -> str:
    """
    Replace calculation annotations with actual calculated values.
    In GSM8K format, calculations are annotated as <<expression=result>>.
    """
    def evaluate_calculation(match):
        calculation = match.group(1)
        # Return the result part (after the '=')
        if '=' in calculation:
            logger.debug(f"Processing calculation: {calculation}")
            return calculation.split('=')[1]
        return match.group(0)
    
    processed = re.sub(r'<<(.*?)>>', evaluate_calculation, text)
    
    # Log if any substitutions were made
    if processed != text:
        logger.debug("Processed calculation annotations in text")
    
    return processed

def generate_text(
    model, 
    params, 
    tokenizer, 
    prompt: str, 
    mesh=None,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    Generate text from the model with detailed logging.
    
    Args:
        model: The JAX model
        params: Model parameters
        tokenizer: Hugging Face tokenizer
        prompt: Text prompt
        mesh: Optional mesh for tensor parallelism
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Generated text as string
    """
    logger.info(f"Generating text with {max_new_tokens} max new tokens (temp={temperature})")
    logger.debug(f"Prompt: {prompt[:100]}...")
    
    generation_start_time = time.time()
    
    # Tokenize input
    tokenize_start = time.time()
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs.input_ids)
    logger.debug(f"Tokenized prompt to {input_ids.shape} in {time.time() - tokenize_start:.3f}s")
    
    # For tensor parallel models, need to handle sharding
    if mesh is not None:
        # For tensor parallel, we need to replicate the batch dimension to match mesh
        batch_size = max(1, mesh.shape[0])
        if input_ids.shape[0] != batch_size:
            input_ids = jnp.repeat(input_ids, batch_size, axis=0)
            logger.debug(f"Expanded input_ids to batch size {batch_size}: {input_ids.shape}")
        
        # Shard input according to mesh
        input_sharding = NamedSharding(mesh, P('batch', None))
        input_ids = jax.device_put(input_ids, input_sharding)
        logger.debug("Input sharded according to mesh")
    
    # Tracking variables for generation
    generated_tokens = []
    current_input_ids = input_ids
    
    with mesh if mesh is not None else jax.advanced_context_managers.NullContextManager():
        logger.debug(f"Starting generation loop with context manager")
        
        # Simple auto-regressive generation
        for i in range(max_new_tokens):
            step_start = time.time()
            
            # Forward pass to get logits
            outputs = model.apply(params, current_input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            
            # Get the last token's logits for each sequence
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Get the top-k tokens
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    # Convert to probabilities
                    top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)
                    
                    # Sort probabilities descending
                    sorted_probs = -jnp.sort(-top_k_probs, axis=-1)
                    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                    
                    # Create a mask for indices to keep
                    nucleus_mask = cumulative_probs <= top_p
                    # Always keep at least one token
                    nucleus_mask = jnp.concatenate([
                        jnp.ones_like(nucleus_mask[:, :1]), 
                        nucleus_mask[:, 1:]
                    ], axis=-1)
                    
                    # Apply the mask to probabilities
                    masked_probs = jnp.where(nucleus_mask, top_k_probs, 0.0)
                    # Renormalize
                    masked_probs = masked_probs / jnp.sum(masked_probs, axis=-1, keepdims=True)
                    
                    # Sample from the filtered distribution
                    key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**32))
                    selected_idx = jax.random.categorical(key, jnp.log(masked_probs), axis=-1)
                    next_token = jnp.take_along_axis(top_k_indices, selected_idx[:, None], axis=-1)[:, 0]
                else:
                    # Sample from top-k without top-p
                    key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**32))
                    next_token_probs = jax.nn.softmax(top_k_logits, axis=-1)
                    selected_idx = jax.random.categorical(key, jnp.log(next_token_probs), axis=-1)
                    next_token = jnp.take_along_axis(top_k_indices, selected_idx[:, None], axis=-1)[:, 0]
            else:
                # Greedy decoding
                next_token = jnp.argmax(next_token_logits, axis=-1)
            
            # Take the first batch item for output
            token_id = next_token[0].item()
            generated_tokens.append(token_id)
            
            # Prepare input for next iteration
            next_token = next_token.reshape(-1, 1)
            current_input_ids = jnp.concatenate([current_input_ids, next_token], axis=1)
            
            step_time = time.time() - step_start
            
            # Log every few tokens
            if i % 10 == 0 or i == max_new_tokens - 1:
                try:
                    # Decode tokens so far
                    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.info(f"Generation progress: {i+1}/{max_new_tokens} tokens - {len(decoded)} chars - {step_time:.3f}s/token")
                except Exception as e:
                    logger.error(f"Error decoding tokens: {e}")
            
            # Check for end of sequence token
            if token_id == tokenizer.eos_token_id:
                logger.info(f"Reached EOS token after {i+1} tokens")
                break
    
    # Decode the generated sequence
    if len(generated_tokens) > 0:
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        total_time = time.time() - generation_start_time
        tokens_per_second = len(generated_tokens) / total_time
        
        logger.info(f"Generation complete: {len(generated_tokens)} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")
        logger.debug(f"Generated text: {result[:100]}...")
        return result
    else:
        logger.warning("No tokens were generated")
        return ""

def evaluate_real_model(
    model, 
    params, 
    tokenizer,
    dataset, 
    mesh=None, 
    max_samples=None, 
    max_new_tokens=300,
    temperature=0.0
):
    """Evaluate a model on the GSM8K dataset with real generation."""
    results = []
    correct = 0
    total = 0
    
    # Limit samples if specified
    eval_dataset = dataset
    if max_samples is not None and max_samples < len(dataset):
        eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Starting evaluation on {len(eval_dataset)} examples with real generation")
    
    # Template for GSM8K
    prompt_template = "Question: {question}\n\nAnswer:"
    
    for i, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        example_start_time = time.time()
        logger.info(f"\n\n==== Example {i+1}/{len(eval_dataset)} ====")
        
        try:
            # Format the question
            question = example["question"]
            logger.info(f"Question: {question}")
            
            # Create prompt
            prompt = prompt_template.format(question=question)
            
            # Generate answer
            logger.info("Generating answer...")
            generated_text = generate_text(
                model=model,
                params=params,
                tokenizer=tokenizer,
                prompt=prompt,
                mesh=mesh,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Handle calculation annotations
            processed_text = handle_calculation_annotations(generated_text)
            
            # Parse the answer
            pred_answer = parse_answer(processed_text)
            true_answer = extract_answer_from_gsm8k_example(example)
            
            # Check correctness
            is_correct = pred_answer == true_answer
            if is_correct:
                correct += 1
                logger.info(f"✓ CORRECT: Predicted {pred_answer}, Expected {true_answer}")
            else:
                logger.info(f"✗ INCORRECT: Predicted {pred_answer}, Expected {true_answer}")
            
            total += 1
            
            # Store the result
            results.append({
                "question": question,
                "true_answer": true_answer,
                "generated_text": generated_text,
                "processed_text": processed_text,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "processing_time": time.time() - example_start_time
            })
            
            # Print running accuracy
            accuracy = correct / total
            logger.info(f"Running accuracy: {accuracy:.4f} ({correct}/{total})")
            
        except Exception as e:
            logger.error(f"Error evaluating example {i+1}: {str(e)}", exc_info=True)
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\nFinal accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return results, accuracy

def main():
    parser = argparse.ArgumentParser(description='Real GSM8K evaluation for Qwen2.5 models')
    parser.add_argument('--model_path', type=str, default='/Users/lu/Documents/tt-bounty-1/qwen2.5-7b',
                        help='Path to the model weights directory')
    parser.add_argument('--mesh_shape', type=str, default='1x8',
                        help='Shape of the device mesh for tensor parallelism (batch, model)')
    parser.add_argument('--max_examples', type=int, default=5,
                        help='Maximum number of examples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=300, 
                        help='Maximum number of tokens to generate for each example')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0.0 for greedy decoding)')
    parser.add_argument('--save_results', type=str, default='gsm8k_real_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--log_file', type=str, default='gsm8k_real_eval.log',
                        help='Path to save detailed logs')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    

    # Create timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_results:
        args.save_results = f"{timestamp}_{args.save_results}"
    if args.log_file:
        args.log_file = f"{timestamp}_{args.log_file}"
    
    # Set up detailed logging
    setup_logging(args.log_file, args.verbose)
    logger.info(f"Starting GSM8K real evaluation with Qwen2.5-7B")
    logger.info(f"Arguments: {args}")
    
    # Parse mesh shape
    mesh_parts = args.mesh_shape.split('x')
    if len(mesh_parts) != 2:
        logger.error(f"Invalid mesh shape '{args.mesh_shape}'. Format should be 'AxB'.")
        return 1
    
    mesh_shape = (int(mesh_parts[0]), int(mesh_parts[1]))
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    # Set XLA flags for device simulation if not already set
    if "XLA_FLAGS" not in os.environ:
        xla_flags = f"--xla_force_host_platform_device_count={required_devices}"
        os.environ["XLA_FLAGS"] = xla_flags
        logger.info(f"Set XLA_FLAGS to simulate {required_devices} devices: {xla_flags}")
    
    # Load model configuration
    logger.info(f"Loading model configuration from {args.model_path}")
    start_time = time.time()
    try:
        config = load_qwen_config(args.model_path)
        logger.info(f"Loaded configuration in {time.time() - start_time:.2f}s")
        
        # Print model details
        logger.info(f"Model configuration:")
        logger.info(f"- Hidden size: {config['hidden_size']}")
        logger.info(f"- Layers: {config['num_hidden_layers']}")
        logger.info(f"- Attention heads: {config['num_attention_heads']}")
        logger.info(f"- KV heads: {config['num_key_value_heads']}")
        logger.info(f"- Vocabulary size: {config['vocab_size']}")
        
        # Set model_type in config for auto classes
        config["model_type"] = "qwen2_5"
    except Exception as e:
        logger.error(f"Failed to load model configuration: {e}")
        return 1
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer_start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info(f"Loaded tokenizer in {time.time() - tokenizer_start:.2f}s: {tokenizer.__class__.__name__}")
        logger.info(f"Vocabulary size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return 1
    
    # Load GSM8K dataset first to verify it works
    logger.info("Loading GSM8K dataset")
    try:
        dataset = load_gsm8k_dataset("test", args.max_examples)
    except Exception as e:
        logger.error(f"Error loading GSM8K dataset: {e}")
        return 1
    
    # Initialize models with real weights
    dtype = jnp.bfloat16
    param_dtype = jnp.bfloat16
    
    logger.info(f"Creating tensor-parallel Qwen2.5 model with mesh shape {mesh_shape}")
    mesh_start = time.time()
    
    try:
        # Create device mesh
        mesh = create_device_mesh(mesh_shape)
        logger.info(f"Created device mesh in {time.time() - mesh_start:.2f}s")
        
        # Create tensor-parallel model
        model_start = time.time()
        tp_model = get_model(
            model_type="qwen2_5",
            use_tensor_parallel=True,
            mesh_shape=mesh_shape,
            config=config,
            dtype=dtype,
            param_dtype=param_dtype
        )
        logger.info(f"Created tensor-parallel model in {time.time() - model_start:.2f}s")
        
        # Initialize parameter shapes with a small input
        logger.info("Initializing parameter shapes...")
        batch_size = max(1, mesh_shape[0])
        input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
        input_sharding = NamedSharding(mesh, P('batch', None))
        sharded_input = jax.device_put(input_ids, input_sharding)
        
        # Initialize parameters
        init_start = time.time()
        with mesh:
            rng = jax.random.PRNGKey(0)
            params = tp_model.init(rng, sharded_input)
            logger.info(f"Initialized parameter shapes in {time.time() - init_start:.2f}s")
            
            # Load real weights
            logger.info(f"Loading real weights from {args.model_path}...")
            weights_start = time.time()
            
            try:
                loaded_params = load_qwen_weights(args.model_path, tp_model, mesh=mesh, param_dtype=param_dtype)
                logger.info(f"Loaded real weights in {time.time() - weights_start:.2f}s")
            except Exception as e:
                logger.error(f"Error loading weights: {e}", exc_info=True)
                logger.warning("Continuing with random weights for demonstration")
                loaded_params = params  # Use initialized random weights
        
        # Evaluate the model
        logger.info("\n===== STARTING EVALUATION WITH REAL WEIGHTS =====\n")
        eval_start = time.time()
        results, accuracy = evaluate_real_model(
            tp_model, 
            loaded_params, 
            tokenizer,
            dataset, 
            mesh=mesh, 
            max_samples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        eval_time = time.time() - eval_start
        logger.info(f"\nEvaluation completed in {eval_time:.2f}s")
        logger.info(f"Final accuracy: {accuracy:.4f}")
        
        # Save results if requested
        if args.save_results:
            results_data = {
                "model": "Qwen2.5-7B Tensor Parallel",
                "mesh_shape": args.mesh_shape,
                "max_examples": args.max_examples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "accuracy": accuracy,
                "eval_time": eval_time,
                "results": results
            }
            
            with open(args.save_results, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Saved evaluation results to {args.save_results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 