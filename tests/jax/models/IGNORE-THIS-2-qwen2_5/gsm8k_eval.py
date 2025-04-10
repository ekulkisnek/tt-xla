# Copyright 2024 TensorTrace Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GSM8K evaluation for Qwen2.5 with JAX/Flax"""

import argparse
import datetime
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer
from transformers.utils import logging

from .configuration_qwen2_5 import Qwen25Config
from .modeling_flax_qwen2_5 import FlaxQwen25ForCausalLM
from .tensor_parallel import create_device_mesh
from .weight_loading import convert_qwen25_checkpoint

logger = logging.get_logger(__name__)

# GSM8K prompt template
GSM8K_PROMPT = """Solve this step-by-step:

{question}

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 on GSM8K")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gsm8k_results",
        help="Directory to save the evaluation results",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to evaluate (use a small number for testing)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0.0 means greedy decoding)",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        help="Use bfloat16 precision for model weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--gsm8k_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K dataset split to evaluate",
    )
    return parser.parse_args()


def setup_model_and_tokenizer(
    model_path: str, 
    mesh: Optional[Mesh] = None, 
    use_bfloat16: bool = False
) -> Tuple[FlaxQwen25ForCausalLM, AutoTokenizer]:
    """
    Set up the model and tokenizer.
    
    Args:
        model_path: Path to the model checkpoint directory
        mesh: JAX device mesh for tensor parallelism
        use_bfloat16: Whether to use bfloat16 precision for model weights
        
    Returns:
        model, tokenizer
    """
    # Create device mesh for tensor parallelism
    if mesh is None:
        mesh = create_device_mesh()
    
    # Set up model dtype
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float16
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with tensor parallelism
    model = convert_qwen25_checkpoint(
        checkpoint_dir=model_path,
        dtype=dtype,
        with_lm_head=True,
        mesh=mesh,
    )
    
    return model, tokenizer


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from the generated text.
    
    Args:
        text: Generated text
        
    Returns:
        Extracted answer or None if no answer is found
    """
    # Look for "The answer is" pattern
    pattern1 = r"The\s+answer\s+is\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    pattern2 = r"Therefore,?\s+the\s+answer\s+is\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    pattern3 = r"(?:Therefore|So|Thus|Hence),?\s+the\s+(?:final\s+)?(?:value|result|number|answer)\s+is\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    pattern4 = r"The\s+(?:final\s+)?(?:value|result|number)\s+is\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    pattern5 = r"(?:Thus|Therefore|So|Hence),?\s+the\s+final\s+answer\s+is\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    pattern6 = r"^\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*$"  # Standalone number at the end
    
    # Try each pattern
    for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
    
    # Try to find the last mentioned number as a fallback
    number_pattern = r"\s(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*$"
    match = re.search(number_pattern, text)
    if match:
        return match.group(1).replace(",", "")
    
    return None


def check_answer(pred_answer: str, true_answer: str) -> bool:
    """
    Check if the predicted answer matches the true answer.
    
    Args:
        pred_answer: Predicted answer
        true_answer: True answer
        
    Returns:
        Whether the answers match
    """
    if pred_answer is None:
        return False
    
    # Clean and normalize answers
    pred_answer = pred_answer.strip().replace(",", "")
    true_answer = true_answer.strip().replace(",", "")
    
    # Try to convert to float for numeric comparison
    try:
        pred_float = float(pred_answer)
        true_float = float(true_answer)
        return abs(pred_float - true_float) < 1e-5
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return pred_answer == true_answer


def evaluate_gsm8k(
    model: FlaxQwen25ForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    seed: int = 42,
    num_examples: Optional[int] = None,
) -> Dict:
    """
    Evaluate the model on GSM8K.
    
    Args:
        model: The Qwen2.5 model
        tokenizer: The tokenizer
        dataset: GSM8K dataset
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        seed: Random seed for sampling
        num_examples: Number of examples to evaluate (None for all)
        
    Returns:
        Evaluation results
    """
    if num_examples is not None:
        dataset = dataset[:num_examples]
    
    results = []
    correct = 0
    
    # Set up PRNG key for sampling
    rng_key = jax.random.PRNGKey(seed)
    
    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        
        # Extract the ground truth number
        true_answer_match = re.search(r"The answer is (\d{1,3}(?:,\d{3})*(?:\.\d+)?)\.", answer)
        if true_answer_match:
            true_answer = true_answer_match.group(1)
        else:
            logger.warning(f"Could not extract true answer from: {answer}")
            true_answer = ""
        
        # Format prompt
        prompt = GSM8K_PROMPT.format(question=question)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate
        if temperature > 0:
            rng_key, sample_key = jax.random.split(rng_key)
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prng_key=sample_key,
            )
        else:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_answer = generated_text[len(prompt):]
        
        # Extract the answer
        pred_answer = extract_answer(generated_answer)
        
        # Check the answer
        is_correct = check_answer(pred_answer, true_answer)
        if is_correct:
            correct += 1
        
        # Save the result
        results.append({
            "question": question,
            "true_answer": true_answer,
            "generated_answer": generated_answer,
            "predicted_answer": pred_answer,
            "correct": is_correct,
        })
        
        # Log progress
        logger.info(f"Processed {i+1}/{len(dataset)}: {'✓' if is_correct else '✗'} Pred: {pred_answer}, True: {true_answer}")
    
    # Calculate accuracy
    accuracy = correct / len(dataset) if dataset else 0.0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }


def main():
    args = parse_args()
    
    # Set up logging
    logging.set_verbosity_info()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up JAX device mesh
    mesh = create_device_mesh()
    
    # Set up model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_path}")
    model, tokenizer = setup_model_and_tokenizer(
        model_path=args.model_path,
        mesh=mesh,
        use_bfloat16=args.use_bfloat16,
    )
    
    # Load GSM8K dataset
    logger.info(f"Loading GSM8K {args.gsm8k_split} split")
    dataset = load_dataset("gsm8k", "main")[args.gsm8k_split]
    
    # Run evaluation
    logger.info(f"Evaluating on {args.num_examples} examples")
    start_time = time.time()
    results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        num_examples=args.num_examples,
    )
    end_time = time.time()
    
    # Log results
    logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    logger.info(f"Evaluation time: {end_time - start_time:.2f}s")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"gsm8k_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main() 