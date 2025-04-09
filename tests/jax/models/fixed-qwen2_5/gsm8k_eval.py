# coding=utf-8
# Copyright 2024 Tenstorrent AI ULC. All rights reserved.
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

"""
GSM8K evaluation script for Qwen2.5-7B models.
"""

import os
import sys
import logging
import json
import argparse
import re
from typing import Dict, List, Optional, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig

from .weight_loading import load_qwen_model

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def extract_answer(text: str) -> Optional[float]:
    """
    Extract the final answer from generated text.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Extracted final answer or None if no answer found
    """
    # Look for patterns like "The answer is X" or just the final number
    answer_patterns = [
        r"The answer is\s*(\d+\.?\d*)",
        r"The final answer is\s*(\d+\.?\d*)",
        r"Therefore,? the answer is\s*(\d+\.?\d*)",
        r"Thus,? the answer is\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)$",  # Just a number at the end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # If no match found, try to find any number in the last sentence
    sentences = text.split('.')
    last_sentence = sentences[-1]
    numbers = re.findall(r"(\d+\.?\d*)", last_sentence)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def check_answer(predicted: Optional[float], reference: Union[str, float]) -> bool:
    """
    Check if the predicted answer matches the reference.
    
    Args:
        predicted: Extracted answer from model output
        reference: Reference answer from the dataset
        
    Returns:
        Boolean indicating whether the answer is correct
    """
    if predicted is None:
        return False
    
    try:
        ref_value = float(reference)
        return abs(predicted - ref_value) < 1e-6
    except (ValueError, TypeError):
        return str(predicted) == str(reference)

def evaluate_gsm8k(
    model_path: str,
    mesh_shape: Tuple[int, int] = (1, 8),
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    num_samples: Optional[int] = None,
    output_file: Optional[str] = None,
    from_pt: bool = True,
    use_cache: bool = True,
) -> Dict:
    """
    Evaluate Qwen2.5 model on GSM8K dataset with tensor parallelism.
    
    Args:
        model_path: Path to model weights
        mesh_shape: Shape of device mesh (batch_dim, model_dim)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (0.0 for greedy)
        num_samples: Number of samples to evaluate (None for all)
        output_file: Path to save results (None to skip saving)
        from_pt: Whether to convert from PyTorch format
        use_cache: Whether to use cached Flax weights
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        RuntimeError: If device configuration is invalid or dataset loading fails
        ValueError: If input parameters are invalid
    """
    logger.info(f"Starting evaluation with mesh shape {mesh_shape}")
    
    # Validate input parameters
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature < 0:
        raise ValueError("Temperature must be non-negative")
    
    # Check device availability
    total_devices_needed = mesh_shape[0] * mesh_shape[1]
    available_devices = len(jax.devices())
    if available_devices < total_devices_needed:
        raise RuntimeError(
            f"Not enough devices available. Need {total_devices_needed} but only found {available_devices}. "
            f"Please adjust mesh_shape ({mesh_shape}) to match available devices."
        )
    
    try:
        # Load model with tensor parallelism
        logger.info(f"Loading model from {model_path}")
        model, tokenizer = load_qwen_model(
            model_path,
            mesh_shape=mesh_shape,
            from_pt=from_pt,
            use_cache=use_cache
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    try:
        # Load GSM8K dataset
        logger.info("Loading GSM8K dataset")
        dataset = load_dataset("gsm8k", "main", split="test")
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
    except Exception as e:
        raise RuntimeError(f"Failed to load GSM8K dataset: {str(e)}")
    
    # Create batches
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i + batch_size, len(dataset))]
        batches.append(batch)
    
    # Set up device mesh
    try:
        devices = jax.devices()
        device_mesh = jnp.array(devices[:total_devices_needed]).reshape(mesh_shape)
        mesh = Mesh(device_mesh, ('batch', 'model'))
        logger.info(f"Created device mesh with shape {mesh_shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to create device mesh: {str(e)}")
    
    results = []
    correct_count = 0
    total_count = 0
    
    try:
        with mesh:
            for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating")):
                # Prepare prompts
                prompts = [
                    f"Question: {item['question']}\nLet's solve this step by step:\n1)"
                    for item in batch
                ]
                
                try:
                    # Tokenize inputs
                    inputs = tokenizer(
                        prompts,
                        padding=True,
                        truncation=True,
                        return_tensors="np"
                    )
                    
                    # Apply sharding constraints
                    input_ids = jax.lax.with_sharding_constraint(
                        inputs["input_ids"], P('batch', None))
                    attention_mask = jax.lax.with_sharding_constraint(
                        inputs["attention_mask"], P('batch', None))
                    
                    # Generate answers
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0.0,
                    )
                    
                    # Decode outputs
                    generated_texts = tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
                    
                    # Process results for this batch
                    for item, generated_text in zip(batch, generated_texts):
                        predicted = extract_answer(generated_text)
                        is_correct = check_answer(predicted, item['answer'])
                        result = {
                            'question': item['question'],
                            'reference_answer': item['answer'],
                            'generated_text': generated_text,
                            'predicted_answer': predicted,
                            'correct': is_correct
                        }
                        results.append(result)
                        correct_count += int(is_correct)
                        total_count += 1
                        
                        if (batch_idx + 1) % 10 == 0:
                            logger.info(f"Current accuracy: {correct_count/total_count:.2%} ({correct_count}/{total_count})")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
    
    except Exception as e:
        logger.error(f"Evaluation interrupted: {str(e)}")
        if not results:
            raise RuntimeError("No results generated before error occurred")
    
    # Calculate final metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'results': results
    }
    
    # Save results if output file specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {str(e)}")
    
    logger.info(f"Final accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5 models on GSM8K')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model weights directory')
    parser.add_argument('--mesh_shape', type=str, default='1x8',
                      help='Shape of the device mesh for tensor parallelism (batch, model)')
    parser.add_argument('--max_examples', type=int, default=None,
                      help='Maximum number of examples to evaluate')
    parser.add_argument('--save_results', type=str, default=None,
                      help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Parse mesh shape
    mesh_parts = args.mesh_shape.split('x')
    if len(mesh_parts) != 2:
        logger.error(f"Invalid mesh shape '{args.mesh_shape}'. Format should be 'AxB'.")
        return 1
    
    mesh_shape = (int(mesh_parts[0]), int(mesh_parts[1]))
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    # Set XLA flags for device simulation if not already set
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={required_devices}"
        logger.info(f"Set XLA_FLAGS to simulate {required_devices} devices")
    
    # Run evaluation
    try:
        results = evaluate_gsm8k(
            model_path=args.model_path,
            mesh_shape=mesh_shape,
            num_samples=args.max_examples,
            output_file=args.save_results
        )
        
        print(f"\nResults:")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Correct: {results['correct_count']}/{results['total_count']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 