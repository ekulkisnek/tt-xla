#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script to confirm that tensor parallel implementation of Qwen2.5-7B
produces the same GSM8k score as the single-device implementation.
This is a requirement for the bounty.
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
# We set this here, but it can be overridden by the environment
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from typing import List, Dict, Optional, Tuple
import re
from tqdm import tqdm

# Import model components
from model_implementation import Qwen2ForCausalLM
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from config import get_qwen2_7b_config
from weight_loading import init_model_from_weights


def setup_tokenizer(tokenizer_path=None):
    """Set up the tokenizer."""
    try:
        from transformers import AutoTokenizer
        
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        print(f"✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Please install transformers: pip install transformers")
        return None


def load_standard_model(model_path: str):
    """Load the standard (non-tensor-parallel) model."""
    print("\nLoading standard Qwen2.5-7B model...")
    
    # Get model configuration
    config = get_qwen2_7b_config()
    
    # Initialize model
    model = Qwen2ForCausalLM(config=config)
    
    # Initialize and load weights
    params = init_model_from_weights(
        model_class=model.__class__,
        model_path=model_path,
        config=config,
        mesh=None,
        param_dtype=jnp.bfloat16
    )
    
    print(f"✅ Standard model loaded successfully")
    return model, params


def load_tensor_parallel_model(model_path: str, mesh_shape=(1, 8)):
    """Load the tensor-parallel model."""
    print(f"\nLoading tensor-parallel Qwen2.5-7B model with mesh shape {mesh_shape}...")
    
    # Create device mesh
    mesh = create_device_mesh(mesh_shape)
    print(f"✅ Device mesh created: {mesh_shape}")
    
    # Get model configuration
    config = get_qwen2_7b_config()
    
    # Initialize model
    model = TensorParallelQwen2ForCausalLM(config=config, mesh=mesh)
    
    # Initialize and load weights
    params = init_model_from_weights(
        model_class=model.__class__,
        model_path=model_path,
        config=config,
        mesh=mesh,
        param_dtype=jnp.bfloat16
    )
    
    print(f"✅ Tensor-parallel model loaded successfully")
    return model, params, mesh


def extract_answer(text):
    """Extract the final numerical answer from a solution."""
    # Look for common answer patterns
    patterns = [
        r"(?:answer|result)(?:\s+is)?\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)",
        r"(?:=|:)\s*(-?\d+(?:\.\d+)?)\s*$",
        r"(-?\d+(?:\.\d+)?)\s*$",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]
    
    # If no match found, try to find the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1]
    
    return None


def format_prompt(prompt, use_template=True):
    """Format the input prompt using the Qwen2.5 chat template."""
    if not use_template:
        return prompt
        
    # Qwen2.5 chat template
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return template.format(prompt=prompt)


def generate_text_standard(
    model,
    params,
    input_ids,
    tokenizer,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
):
    """Generate text using the standard model."""
    # Track generated tokens
    generated_ids = input_ids.copy()
    
    # Generate tokens autoregressively
    for _ in range(max_length):
        # Run the model on the current input
        outputs = model.apply(params, generated_ids)
        
        # Get logits for the last token
        logits = outputs[0][0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p sampling
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
        mask = cumulative_probs < top_p
        
        # Ensure at least one token is selected
        mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[1:]], axis=0)
        
        # Get the allowed logits and indices
        allowed_logits = jnp.where(mask, sorted_logits, -float('inf'))
        allowed_indices = sorted_indices[mask]
        
        # Check if we have any allowed indices
        if allowed_indices.size == 0:
            # Fallback to top-1 sampling
            next_token = jnp.argmax(logits)
        else:
            # Sample from the allowed tokens
            probs = jax.nn.softmax(allowed_logits)
            
            # Fix potential shape mismatch by normalizing probabilities
            probs = probs / jnp.sum(probs)
            
            try:
                next_token = jax.random.choice(
                    jax.random.PRNGKey(int(time.time() * 1000000)), 
                    allowed_indices, 
                    p=probs
                )
            except ValueError:
                # Fallback to using the most likely token
                next_token = allowed_indices[0]
        
        # Add the new token to the generated sequence
        generated_ids = jnp.concatenate([generated_ids, next_token.reshape(1, 1)], axis=1)
        
        # Check for EOS tokens
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    
    # Return the generated text
    return tokenizer.decode(generated_ids[0], skip_special_tokens=False)


def generate_text_tensor_parallel(
    model,
    params,
    mesh,
    input_ids,
    tokenizer,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
):
    """Generate text using the tensor-parallel model."""
    # Track generated tokens
    generated_ids = input_ids.copy()
    
    # Apply input sharding
    input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
    
    # Generate tokens autoregressively
    for _ in range(max_length):
        # Shard the input
        sharded_input = jax.device_put(generated_ids, input_sharding)
        
        # Run the model on the current input
        with mesh:
            outputs = model.apply(params, sharded_input)
        
        # Get logits for the last token
        logits = outputs[0][0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p sampling
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
        mask = cumulative_probs < top_p
        
        # Ensure at least one token is selected
        mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[1:]], axis=0)
        
        # Get the allowed logits and indices
        allowed_logits = jnp.where(mask, sorted_logits, -float('inf'))
        allowed_indices = sorted_indices[mask]
        
        # Check if we have any allowed indices
        if allowed_indices.size == 0:
            # Fallback to top-1 sampling
            next_token = jnp.argmax(logits)
        else:
            # Sample from the allowed tokens
            probs = jax.nn.softmax(allowed_logits)
            
            # Fix potential shape mismatch by normalizing probabilities
            probs = probs / jnp.sum(probs)
            
            try:
                next_token = jax.random.choice(
                    jax.random.PRNGKey(int(time.time() * 1000000)), 
                    allowed_indices, 
                    p=probs
                )
            except ValueError:
                # Fallback to using the most likely token
                next_token = allowed_indices[0]
        
        # Add the new token to the generated sequence
        generated_ids = jnp.concatenate([generated_ids, next_token.reshape(1, 1)], axis=1)
        
        # Check for EOS tokens
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    
    # Return the generated text
    return tokenizer.decode(generated_ids[0], skip_special_tokens=False)


def evaluate_gsm8k(
    standard_model=None,
    standard_params=None,
    tp_model=None,
    tp_params=None,
    tp_mesh=None,
    tokenizer=None,
    num_samples=10,
):
    """Evaluate the model on GSM8K problems."""
    try:
        import datasets
    except ImportError:
        print("Please install datasets: pip install datasets")
        return None, None
    
    # Load GSM8K dataset
    print(f"Loading GSM8K dataset...")
    try:
        dataset = datasets.load_dataset("gsm8k", "main", split="test")
        print(f"✅ Dataset loaded successfully with {len(dataset)} problems")
    except Exception as e:
        print(f"❌ Error loading GSM8K dataset: {e}")
        print("Try: pip install datasets")
        return None, None
    
    # Select a subset of problems
    num_problems = min(num_samples, len(dataset))
    indices = list(range(min(num_problems * 2, len(dataset))))[:num_problems]
    subset = dataset.select(indices)
    print(f"Selected {num_problems} problems for evaluation")
    
    # Function to create a system prompt
    def create_prompt(question):
        return f"Please solve the following math problem step by step:\n\n{question}"
    
    # Generate outputs
    standard_correct = 0
    tp_correct = 0
    standard_answers = []
    tp_answers = []
    match_count = 0
    
    print("\nGenerating answers...")
    for i, example in enumerate(tqdm(subset, desc="Evaluating")):
        question = example["question"]
        answer = example["answer"]
        true_answer = extract_answer(answer)
        
        # Format the prompt
        prompt = create_prompt(question)
        formatted_prompt = format_prompt(prompt)
        
        # Encode the prompt
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="np")
        input_ids = jnp.array(input_ids)
        
        # Generate output from standard model
        if standard_model is not None and standard_params is not None:
            print(f"\nProblem {i+1}: Generating standard model answer...")
            standard_output = generate_text_standard(
                standard_model, 
                standard_params, 
                input_ids, 
                tokenizer, 
                max_length=200  # Longer to allow for step-by-step reasoning
            )
            standard_predicted = extract_answer(standard_output)
            standard_answers.append({
                "question": question, 
                "output": standard_output, 
                "extracted": standard_predicted,
                "true_answer": true_answer
            })
            
            # Check if the answer is correct
            if standard_predicted == true_answer:
                standard_correct += 1
                print(f"✅ Standard model correct: {standard_predicted}")
            else:
                print(f"❌ Standard model wrong: predicted={standard_predicted}, true={true_answer}")
        
        # Generate output from tensor-parallel model
        if tp_model is not None and tp_params is not None and tp_mesh is not None:
            print(f"Problem {i+1}: Generating tensor-parallel model answer...")
            tp_output = generate_text_tensor_parallel(
                tp_model, 
                tp_params, 
                tp_mesh,
                input_ids, 
                tokenizer, 
                max_length=200  # Longer to allow for step-by-step reasoning
            )
            tp_predicted = extract_answer(tp_output)
            tp_answers.append({
                "question": question, 
                "output": tp_output, 
                "extracted": tp_predicted,
                "true_answer": true_answer
            })
            
            # Check if the answer is correct
            if tp_predicted == true_answer:
                tp_correct += 1
                print(f"✅ Tensor-parallel model correct: {tp_predicted}")
            else:
                print(f"❌ Tensor-parallel model wrong: predicted={tp_predicted}, true={true_answer}")
        
        # Check if the models agree
        if standard_model is not None and tp_model is not None:
            if standard_predicted == tp_predicted:
                match_count += 1
                print(f"✓ Models match: {standard_predicted}")
            else:
                print(f"✗ Models disagree: standard={standard_predicted}, tp={tp_predicted}")
                
        print("-" * 40)
    
    # Compute scores
    results = {
        "num_problems": num_problems,
    }
    
    if standard_model is not None:
        standard_score = standard_correct / num_problems
        results["standard_score"] = standard_score
        results["standard_answers"] = standard_answers
        print(f"\nStandard model score: {standard_score:.2f} ({standard_correct}/{num_problems})")
    
    if tp_model is not None:
        tp_score = tp_correct / num_problems
        results["tp_score"] = tp_score
        results["tp_answers"] = tp_answers
        print(f"Tensor-parallel model score: {tp_score:.2f} ({tp_correct}/{num_problems})")
    
    if standard_model is not None and tp_model is not None:
        match_rate = match_count / num_problems
        results["match_rate"] = match_rate
        print(f"Model match rate: {match_rate:.2f} ({match_count}/{num_problems})")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Verify tensor parallel implementation on GSM8K benchmark"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/lu/Documents/tt-bounty-1/qwen2.5-7b",
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of GSM8K examples to test"
    )
    
    parser.add_argument(
        "--tp_only",
        action="store_true",
        help="Only test the tensor-parallel model"
    )
    
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="1,8",
        help="Mesh shape for tensor parallelism (format: rows,cols)"
    )
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return 1
    
    # Get mesh shape
    mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    if len(mesh_shape) != 2:
        print(f"Error: Invalid mesh shape {args.mesh_shape}, must be in format rows,cols")
        return 1
    
    # Check available devices
    required_devices = mesh_shape[0] * mesh_shape[1]
    available_devices = len(jax.devices())
    print(f"Available devices: {available_devices}")
    print(f"Required devices for mesh shape {mesh_shape}: {required_devices}")
    
    if available_devices < required_devices and "XLA_FLAGS" not in os.environ:
        print(f"⚠️ Not enough devices for mesh shape {mesh_shape}")
        print(f"⚠️ Set XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}' to simulate")
        
    # Set up tokenizer
    tokenizer = setup_tokenizer(args.model_path)
    if tokenizer is None:
        print("Exiting due to tokenizer error")
        return 1
    
    # Load standard model if needed
    standard_model = None
    standard_params = None
    if not args.tp_only:
        standard_model, standard_params = load_standard_model(args.model_path)
    
    # Load tensor-parallel model
    tp_model, tp_params, tp_mesh = load_tensor_parallel_model(args.model_path, mesh_shape)
    
    # Evaluate on GSM8K
    results = evaluate_gsm8k(
        standard_model=standard_model,
        standard_params=standard_params,
        tp_model=tp_model,
        tp_params=tp_params,
        tp_mesh=tp_mesh,
        tokenizer=tokenizer,
        num_samples=args.num_samples
    )
    
    if results is None:
        print("Evaluation failed")
        return 1
    
    # Check if tensor-parallel model matches standard model
    if not args.tp_only and "match_rate" in results:
        if results["match_rate"] >= 0.90:  # At least 90% match
            print("\n✅ VERIFICATION PASSED: Tensor-parallel model matches standard model")
            print(f"Match rate: {results['match_rate']:.2f}")
            return 0
        else:
            print("\n❌ VERIFICATION FAILED: Tensor-parallel model does not match standard model")
            print(f"Match rate: {results['match_rate']:.2f}")
            print("The tensor-parallel model should produce the same or very similar results as the standard model.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 