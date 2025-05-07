#!/usr/bin/env python3
"""
Test script to verify Qwen25 tokenizer compatibility and functionality.
This script checks if the tokenizer can be loaded correctly and performs basic tokenization tests.

Usage:
    python test_tokenizer.py --model_path /path/to/model/directory
"""

import os
import sys
import argparse
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("tokenizer_test")

def check_tokenizer_files(model_path: str) -> Dict[str, bool]:
    """Check if all necessary tokenizer files exist in the model path."""
    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    file_status = {}
    for file in required_files:
        file_path = os.path.join(model_path, file)
        exists = os.path.exists(file_path)
        file_status[file] = exists
        logger.info(f"Checking {file}: {'✓' if exists else '✗'}")
    
    return file_status

def test_tokenizer(model_path: str):
    """Test the tokenizer with various inputs and print results."""
    try:
        from transformers import AutoTokenizer
        
        # Check tokenizer files first
        file_status = check_tokenizer_files(model_path)
        missing_files = [f for f, exists in file_status.items() if not exists]
        
        if missing_files:
            logger.warning(f"Missing tokenizer files: {', '.join(missing_files)}")
            logger.warning("Attempting to load tokenizer anyway...")
        
        # Try to load tokenizer from model path
        logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Test cases
        test_cases = [
            "Hello, how are you today?",
            "Write a short story about a robot learning to paint.",
            "The quick brown fox jumps over the lazy dog.",
            "你好，世界！",  # Chinese test
            "こんにちは、世界！",  # Japanese test
            "안녕하세요, 세계!",  # Korean test
            "1234567890",  # Numbers
            "!@#$%^&*()",  # Special characters
            "",  # Empty string
            "   ",  # Whitespace
        ]
        
        # Print tokenizer info
        logger.info("\nTokenizer Information:")
        logger.info(f"Vocabulary size: {len(tokenizer)}")
        logger.info(f"Model max length: {tokenizer.model_max_length}")
        logger.info(f"Padding token: {tokenizer.pad_token}")
        logger.info(f"EOS token: {tokenizer.eos_token}")
        logger.info(f"BOS token: {tokenizer.bos_token}")
        
        # Test each case
        logger.info("\nTesting Tokenization:")
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nTest Case {i}:")
            logger.info(f"Input: {repr(test_case)}")
            
            # Tokenize
            tokens = tokenizer.encode(test_case)
            decoded = tokenizer.decode(tokens)
            
            # Print results
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Decoded: {repr(decoded)}")
            
            # Verify round-trip
            if test_case.strip() == decoded.strip():
                logger.info("✓ Round-trip verification passed")
            else:
                logger.warning("✗ Round-trip verification failed")
                logger.warning(f"Original: {repr(test_case)}")
                logger.warning(f"Decoded:  {repr(decoded)}")
        
        # Test batch tokenization
        logger.info("\nTesting Batch Tokenization:")
        batch_inputs = test_cases[:3]  # Use first 3 test cases
        batch_outputs = tokenizer(batch_inputs, padding=True, return_tensors="pt")
        
        logger.info("Batch input:")
        for i, input_text in enumerate(batch_inputs):
            logger.info(f"{i}: {repr(input_text)}")
        
        logger.info("\nBatch output:")
        for i, (input_ids, attention_mask) in enumerate(zip(batch_outputs["input_ids"], batch_outputs["attention_mask"])):
            decoded = tokenizer.decode(input_ids)
            logger.info(f"{i}:")
            logger.info(f"  Input IDs: {input_ids.tolist()}")
            logger.info(f"  Attention Mask: {attention_mask.tolist()}")
            logger.info(f"  Decoded: {repr(decoded)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Qwen25 tokenizer compatibility")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    success = test_tokenizer(args.model_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 