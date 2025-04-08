#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Inference script for Qwen2.5 in JAX with tensor parallelism
"""

import argparse
import logging
import sys
import time
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from tt_xla.tests.jax.models.qwen2_5 import (
    convert_qwen25_checkpoint,
    create_device_mesh,
    partition_rules_qwen25,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen2.5 model (HF format)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Prompt to use for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        help="Use bfloat16 precision (default is float16)",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=None,
        help="Number of partitions for tensor parallelism (defaults to device count)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--stream_output",
        action="store_true",
        help="Stream output tokens as they're generated",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX version: {jax.__version__}")
    
    # Choose precision
    dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float16
    
    # Create device mesh for tensor parallelism
    mesh = create_device_mesh(num_partitions=args.num_partitions)
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load the model
    logger.info(f"Loading model from {args.model_path} with tensor parallelism")
    start_time = time.time()
    model = convert_qwen25_checkpoint(
        checkpoint_dir=args.model_path,
        dtype=dtype,
        with_lm_head=True,
        mesh=mesh,
        partition_rules=partition_rules_qwen25,
    )
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Tokenize input
    logger.info(f"Prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Generate text
    start_time = time.time()
    
    if args.stream_output:
        # Stream tokens as they're generated
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True), end="", flush=True)
        
        for token in generate_stream(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repeat_penalty,
        ):
            print(tokenizer.decode([token], skip_special_tokens=True), end="", flush=True)
        print()  # Add newline at the end
    else:
        # Generate all tokens at once
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repeat_penalty,
        )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_second = tokens_generated / generation_time
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
        logger.info(f"Speed: {tokens_per_second:.2f} tokens/sec")
        
        if args.verbose:
            logger.info("\nOutput:")
        print(output_text)


def generate_stream(
    model,
    input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> List[int]:
    """Generate tokens one by one and yield them as they're generated."""
    batch_size = input_ids.shape[0]
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Get next token prediction
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )
        
        # Extract the new token
        new_token = outputs[0, -1].item()
        generated_tokens.append(new_token)
        
        # Yield the token for streaming
        yield new_token
        
        # Update input_ids for next iteration
        next_input_ids = np.zeros((batch_size, input_ids.shape[1] + 1), dtype=np.int32)
        next_input_ids[:, :-1] = input_ids
        next_input_ids[:, -1] = new_token
        input_ids = next_input_ids
        
        # Update attention mask if needed
        if attention_mask is not None:
            next_attention_mask = np.ones((batch_size, attention_mask.shape[1] + 1), dtype=np.int32)
            next_attention_mask[:, :-1] = attention_mask
            attention_mask = next_attention_mask


if __name__ == "__main__":
    main() 