#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to evaluate a Qwen 2.5 model on the GSM8K dataset.
"""

import os
import re
import json
import time
import argparse
import datetime
import logging
import sys
import copy
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import FrozenDict
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from flax.linen.partitioning import param_with_axes, with_sharding_constraint

# Import from the actual project structure - fix imports to use relative imports
from model_implementation import Qwen2_5Model, Qwen2_5ForCausalLM
from weight_loading import convert_weight_name_to_flax
from config import load_qwen_config

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

def extract_answer(text):
    """Extract numerical answer from text."""
    # Try to find the format "The answer is X" first
    match = re.search(r'[Tt]he answer is (\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # If that fails, look for the last number in the text
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return None

def check_answer(predicted, reference):
    """Check if the predicted answer matches the reference.
    
    Args:
        predicted: Predicted answer (number as string)
        reference: Reference answer text
        
    Returns:
        Boolean indicating if the answers match
    """
    if not predicted:
        return False
        
    # Extract answer from reference
    ref_matches = re.findall(r'(?:^|\s)(\d+(?:,\d+)*(?:\.\d+)?)', reference)
    if ref_matches:
        expected = ref_matches[-1].replace(',', '')
    else:
        return False
            
    # Compare answers (try both float and int comparisons)
    try:
        return float(predicted) == float(expected)
    except ValueError:
        # If float conversion fails, do string comparison
        return predicted == expected
    except Exception:
        # If any parsing fails, return False
        return False

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
    
    # Ensure params are in the right format - wrap in {"params": ...} if not already
    if not isinstance(params, dict) or "params" not in params:
        params = {"params": params}
    
    # Process tokens auto-regressively
    for i in range(max_new_tokens):
        # Forward pass with the model
        try:
            outputs = model.apply(
                params, 
                current_input_ids,
                attention_mask=current_attention_mask
            )
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            logging.info("Trying alternative parameter format...")
            # Try alternate parameter format as fallback
            try:
                outputs = model.apply(
                    {"params": params["params"]}, 
                    current_input_ids,
                    attention_mask=current_attention_mask
                )
            except Exception as e2:
                logging.error(f"Failed with alternative format as well: {e2}")
                raise
        
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

def fix_parameter_shapes(pt_params, config, logger):
    """Fix parameter shapes to match model expectations.
    
    Args:
        pt_params: Dictionary of parameters from PyTorch model
        config: Model configuration
        logger: Logger for debugging
        
    Returns:
        Dictionary of processed parameters
    """
    logger.info(f"Starting parameter shape fixing")
    
    # Print all keys in pt_params for debugging
    logger.info("Raw parameter keys:")
    for i, key in enumerate(sorted(pt_params.keys())):
        logger.info(f"  {i}: {key} - shape: {pt_params[key].shape}")
    
    # Create a mapping from PyTorch parameter names to Flax parameter structure
    logger.info("Creating parameter mapping...")
    
    # Look for the embedding parameter
    embedding_param_name = None
    for name in ["model.embed_tokens.weight"]:
        if name in pt_params:
            embedding_param_name = name
            logger.info(f"Found embedding parameter: {embedding_param_name}")
            break
    
    if not embedding_param_name:
        raise ValueError("Could not find embedding parameter in model weights")
    
    # Start with an empty dict for the params (removed the extra params layer)
    params = {
        "transformer": {
            "embed_tokens": {
                "embedding": pt_params[embedding_param_name]
            },
            "layers": {}
        }
    }
    
    # Add layers
    num_layers = config["num_hidden_layers"]
    logger.info(f"Setting up parameters for {num_layers} layers")
    
    for i in range(num_layers):
        try:
            layer_params = {
                "input_layernorm": {
                    "scale": pt_params[f"model.layers.{i}.input_layernorm.weight"]
                },
                "self_attn": {
                    "q_proj": {
                        "kernel": pt_params[f"model.layers.{i}.self_attn.q_proj.weight"].T
                    },
                    "k_proj": {
                        "kernel": pt_params[f"model.layers.{i}.self_attn.k_proj.weight"].T
                    },
                    "v_proj": {
                        "kernel": pt_params[f"model.layers.{i}.self_attn.v_proj.weight"].T
                    },
                    "o_proj": {
                        "kernel": pt_params[f"model.layers.{i}.self_attn.o_proj.weight"].T
                    }
                },
                "post_attention_layernorm": {
                    "scale": pt_params[f"model.layers.{i}.post_attention_layernorm.weight"]
                },
                "mlp": {
                    "gate_proj": {
                        "kernel": pt_params[f"model.layers.{i}.mlp.gate_proj.weight"].T
                    },
                    "up_proj": {
                        "kernel": pt_params[f"model.layers.{i}.mlp.up_proj.weight"].T
                    },
                    "down_proj": {
                        "kernel": pt_params[f"model.layers.{i}.mlp.down_proj.weight"].T
                    }
                }
            }
            
            # Add bias terms if present
            if f"model.layers.{i}.self_attn.q_proj.bias" in pt_params:
                layer_params["self_attn"]["q_proj"]["bias"] = pt_params[f"model.layers.{i}.self_attn.q_proj.bias"]
            if f"model.layers.{i}.self_attn.k_proj.bias" in pt_params:
                layer_params["self_attn"]["k_proj"]["bias"] = pt_params[f"model.layers.{i}.self_attn.k_proj.bias"]
            if f"model.layers.{i}.self_attn.v_proj.bias" in pt_params:
                layer_params["self_attn"]["v_proj"]["bias"] = pt_params[f"model.layers.{i}.self_attn.v_proj.bias"]
            
            # Use string key for layer index
            params["transformer"]["layers"][str(i)] = layer_params
        except KeyError as e:
            logger.error(f"Error processing layer {i}: {e}")
            raise
    
    # Add final layernorm
    params["transformer"]["norm"] = {
        "scale": pt_params["model.norm.weight"]
    }
    
    # Add LM head
    params["lm_head"] = {
        "kernel": pt_params["lm_head.weight"].T
    }
    
    # Print parameter structure
    logger.info("Parameter structure:")
    logger.info(json.dumps({
        "lm_head": {
            "kernel": list(params["lm_head"]["kernel"].shape),
        },
        "transformer": {
            "embed_tokens": {
                "embedding": list(params["transformer"]["embed_tokens"]["embedding"].shape),
            }
        }
    }, indent=2))

    # Ensure all keys are strings
    def ensure_string_keys(d):
        """Recursively ensure all keys in a dictionary are strings."""
        if not isinstance(d, dict):
            return d
        
        result = {}
        for k, v in d.items():
            # Convert key to string if it's not already
            str_key = str(k)
            # Recursively process dictionary values
            result[str_key] = ensure_string_keys(v) if isinstance(v, dict) else v
        return result
    
    # Apply string key conversion to parameters
    params = ensure_string_keys(params)
    
    # Return the processed parameters
    return params

def flatten_dict_params(params, parent_key='', sep='/'):
    """Flatten a nested dictionary, with path components joined by sep."""
    items = []
    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_params(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict_params(params, sep='/'):
    """Unflatten a dictionary with keys containing sep into a nested dictionary."""
    result = {}
    
    # First, normal unflattening
    for key, value in params.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    
    # Special handling for embedding - make sure it's accessible both ways
    # This helps with loading the embedding parameter reliably
    embed_paths = [
        ['model', 'embed_tokens', 'embedding'],
        ['model', 'embed_tokens', 'kernel'],
    ]
    
    # Check if either path exists
    embed_value = None
    embed_path_found = None
    
    for path in embed_paths:
        d = result
        found = True
        for part in path[:-1]:
            if part not in d:
                found = False
                break
            d = d[part]
        
        if found and path[-1] in d:
            embed_value = d[path[-1]]
            embed_path_found = path
            break
    
    # If we found an embedding parameter, make sure it's accessible through both paths
    if embed_value is not None:
        for path in embed_paths:
            if path != embed_path_found:
                d = result
                for part in path[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[path[-1]] = embed_value
    
    return result

def load_qwen_weights(weights_path: str, debug: bool = False) -> Dict[str, np.ndarray]:
    """
    Load PyTorch weights from safetensors files.
    
    Args:
        weights_path: Path to the model weights
        debug: Enable debug logging
        
    Returns:
        Dictionary of parameter names to numpy arrays
    """
    logger = logging.getLogger(__name__)
    
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors is required for loading weights. Install with 'pip install safetensors'.")
    
    # Check for index file to determine if model is sharded
    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        # Load sharded model
        logger.info(f"Loading model from sharded files using index: {index_file}")
        with open(index_file, "r") as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        
        # Create dictionary to hold all weights
        params = {}
        
        # Track loaded shards to avoid loading the same shard multiple times
        loaded_shards = set()
        
        # Load each weight from its corresponding shard
        for param_name, shard_file in weight_map.items():
            if shard_file not in loaded_shards:
                shard_path = os.path.join(weights_path, shard_file)
                logger.info(f"Loading shard: {shard_path}")
                
                with safe_open(shard_path, framework="numpy") as f:
                    for tensor_name in f.keys():
                        params[tensor_name] = f.get_tensor(tensor_name)
                
                loaded_shards.add(shard_file)
    else:
        # Load monolithic model
        model_file = os.path.join(weights_path, "model.safetensors")
        logger.info(f"Loading model from single file: {model_file}")
        
        # Load all tensors into a dictionary
        params = {}
        with safe_open(model_file, framework="numpy") as f:
            for tensor_name in f.keys():
                params[tensor_name] = f.get_tensor(tensor_name)
    
    # Apply parameter name mapping to convert from PyTorch to JAX format
    mapped_params = {}
    
    # Map parameter names using get_parameter_mapping
    for pytorch_name, tensor in params.items():
        flax_name = pytorch_name  # By default, use the original name
        mapped_params[flax_name] = tensor
    
    if debug:
        logger.info("All keys before fixing:")
        for key in sorted(mapped_params.keys()):
            logger.info(f"  {key}")
        
        # Also log all layernorm weights
        logger.info("\nLAYERNORM KEYS:")
        for key in sorted([k for k in mapped_params.keys() if "layernorm" in k.lower() or "norm" in k.lower()]):
            logger.info(f"  {key} -> {mapped_params[key].shape}")
    
    return mapped_params

def model_forward(model, input_ids, params, config, logger):
    """Run the model forward pass."""
    logger.info(f"Starting model forward pass with input shape {input_ids.shape}")
    
    # Copy input_ids to device
    input_ids = jax.device_put(input_ids)
    
    # Create position IDs
    batch_size, seq_length = input_ids.shape
    position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
    
    # Ensure all keys in params are strings, especially layer indices
    def ensure_string_keys(d):
        """Recursively ensure all keys in a dictionary are strings."""
        if not isinstance(d, dict):
            return d
        
        result = {}
        for k, v in d.items():
            # Convert key to string if it's not already
            str_key = str(k)
            # Recursively process dictionary values
            result[str_key] = ensure_string_keys(v) if isinstance(v, dict) else v
        return result
    
    # Make sure params has string keys at all levels
    params = ensure_string_keys(params)
    
    logger.debug(f"Parameter keys type check - transformer: {type(params.get('transformer', {})).__name__}")
    if 'transformer' in params and 'layers' in params['transformer']:
        layers = params['transformer']['layers']
        logger.debug(f"layers keys types: {[(k, type(k).__name__) for k in list(layers.keys())[:3]]}")
    
    try:
        # The model expects params to be wrapped in a dict with a "params" key
        outputs = model.apply(
            {"params": params},  # Keep the outer wrapper with "params" key
            input_ids,
            position_ids=position_ids,
            use_cache=True,
        )
        
        # Extract logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        logger.info(f"Forward pass successful, logits shape: {logits.shape}")
        return logits
    except Exception as e:
        logger.error(f"Error in model_forward: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def evaluate_gsm8k(model, params, args, logger):
    """Evaluate model on GSM8K dataset.
    
    Args:
        model: Model to evaluate
        params: Model parameters
        args: Arguments
        logger: Logger
    
    Returns:
        Accuracy
    """
    logger.info("Starting GSM8K evaluation")
    
    # Load dataset
    logger.info("Loading GSM8K dataset from Hugging Face")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.weights_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.weights_path)
    logger.info("Tokenizer loaded successfully")
    
    # Load config if provided
    config_path = os.path.join(args.weights_path, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except:
        logger.warning(f"Could not load config from {config_path}, using default config")
        config = load_config()
    
    # Limit number of examples if needed
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
        logger.info(f"Limited evaluation to {len(test_data)} samples")
    
    # Evaluate model on each example
    correct = 0
    results = []
    
    for i, example in enumerate(test_data):
        logger.info(f"Processing example {i+1}/{len(test_data)}")
        question = example["question"]
        answer = example["answer"]
        
        logger.debug(f"Question: {question}")
        logger.debug("Tokenizing input")
        
        # Prepare input
        input_ids = tokenizer.encode(question, return_tensors="np")
        input_ids = jnp.array(input_ids)
        
        # Truncate if necessary
        if args.max_length and input_ids.shape[1] > args.max_length:
            logger.warning(f"Truncating input from {input_ids.shape[1]} to {args.max_length}")
            input_ids = input_ids[:, :args.max_length]
        
        # Run model to get initial logits
        logger.info("Running model inference")
        logits = model_forward(
            model=model, 
            input_ids=input_ids, 
            params=params,
            config=config,
            logger=logger
        )
        
        # Get next token prediction
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]
        
        # Start with the first generated token
        generated_ids = jnp.concatenate([input_ids, next_token], axis=1)
        
        # Generate tokens
        for _ in range(args.max_tokens_to_generate - 1):
            # Forward pass with updated sequence
            current_logits = model_forward(
                model=model,
                input_ids=generated_ids,
                params=params,
                config=config,
                logger=logger
            )
            
            # Get next token prediction
            next_token_logits = current_logits[:, -1, :]
            next_token = jnp.argmax(next_token_logits, axis=-1)
            next_token = next_token[:, None]
            
            # Add to generated sequence
            generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)
            
            # Basic stopping: if the last token is EOS, break
            if next_token[0, 0] == tokenizer.eos_token_id:
                break
        
        # Convert generated tokens to text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Assess correctness
        is_correct = evaluate_accuracy(answer, generated_text)
        if is_correct:
            correct += 1
            
        # Save result
        results.append({
            "question": question,
            "expected": answer,
            "generated": generated_text,
            "correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0
    logger.info(f"Evaluation accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")
    
    # Save results
    with open("gsm8k_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    logger.info("Results saved to gsm8k_results.json")
    
    return accuracy

def shard_params_across_devices(params, config, num_devices):
    """
    Shard model parameters across multiple devices if available.
    
    Args:
        params: Model parameters dictionary
        config: Model configuration
        num_devices: Number of devices to shard across
        
    Returns:
        Sharded parameters dictionary if multiple devices are available
    """
    if num_devices <= 1:
        return params
    
    logger = logging.getLogger(__name__)
    logger.info(f"Sharding model across {num_devices} devices")
    
    # Create a 1D mesh for parameter sharding
    devices = mesh_utils.create_device_mesh((num_devices,))
    
    # Create sharding rules
    with jax.sharding.Mesh(devices, ('dp',)):
        # Helper function to get rules for specific param types
        def get_param_rule(param_name):
            # Shard large matrices along their largest dimension
            if 'kernel' in param_name or 'weight' in param_name:
                # For embedding layers, shard along vocabulary dimension
                if 'embed' in param_name:
                    return P('dp', None)
                # For attention layers, shard along hidden dimension
                elif 'attn' in param_name:
                    return P('dp', None)
                # For MLP layers, shard along intermediate dimension
                elif 'mlp' in param_name:
                    return P('dp', None)
                # For lm_head, shard along vocab dimension
                elif 'lm_head' in param_name:
                    return P(None, 'dp')
                # Default rule for other layers
                return P('dp', None)
            # For biases, don't shard
            elif 'bias' in param_name:
                return P(None)
            # For layernorm weights, don't shard
            elif 'layernorm' in param_name or 'norm' in param_name:
                return P(None)
            # Default: no sharding rule
            return None
        
        # Create sharded parameters dictionary
        sharded_params = {}
        for name, param in params.items():
            if hasattr(param, 'shape') and len(param.shape) > 0:
                rule = get_param_rule(name)
                if rule is not None:
                    try:
                        logger.debug(f"Sharding {name} with rule {rule}, shape {param.shape}")
                        # Create JAX array with sharding constraint
                        sharded_param = jax.device_put(param, jax.sharding.NamedSharding(
                            jax.sharding.Mesh(devices, ('dp',)), rule))
                        sharded_params[name] = sharded_param
                    except Exception as e:
                        logger.warning(f"Failed to shard {name}: {e}")
                        sharded_params[name] = param
                else:
                    sharded_params[name] = param
            else:
                sharded_params[name] = param
                
        logger.info(f"Successfully sharded {len(sharded_params)} parameters")
        return sharded_params

def main():
    """Main entry point for GSM8K evaluation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 model on GSM8K dataset")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_tokens_to_generate", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Enable Flax debug logging
        logging.getLogger("flax").setLevel(logging.DEBUG)
        # Set JAX logging to debug
        logging.getLogger("jax").setLevel(logging.DEBUG)
    
    logger.info(f"Starting evaluation with args: {args}")
    
    # Initialize JAX
    try:
        logger.info("Initializing JAX runtime")
        import jax
        # Use only CPU backend for simplicity
        jax.config.update('jax_platforms', 'cpu')
        logger.info(f"JAX runtime initialized with devices: {jax.devices()}")
    except Exception as e:
        logger.error(f"Error initializing JAX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Load model and weights
    try:
        logger.info(f"Loading weights from {args.weights_path}")
        # Simplified model initialization, directly loading parameters
        # rather than initializing model first
        from transformers import AutoConfig
        
        # Load model config
        config = AutoConfig.from_pretrained(args.weights_path)
        config_dict = config.to_dict()
        logger.info(f"Loaded model config with hidden_size={config_dict.get('hidden_size')}")
        
        # Initialize model
        from model_implementation import Qwen2_5Model, Qwen2_5ForCausalLM
        
        # Create model
        model = Qwen2_5ForCausalLM(
            config=config_dict,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        
        # Create a PRNG key
        key = jax.random.PRNGKey(0)
        
        # Load weights
        from transformers import AutoModelForCausalLM
        import numpy as np
        import torch
        
        # Load PyTorch model to extract weights
        logger.info("Loading PyTorch model to extract weights")
        pt_model = AutoModelForCausalLM.from_pretrained(args.weights_path)
        
        # Convert PyTorch state dict to NumPy
        logger.info("Converting PyTorch weights to NumPy")
        pt_params = {k: v.cpu().numpy() for k, v in pt_model.state_dict().items()}
        
        # Map PyTorch parameter names to Flax parameter structure
        logger.info("Mapping parameters to Flax structure")
        params = fix_parameter_shapes(pt_params, config_dict, logger)
        
        logger.info("Model and weights loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Evaluate on GSM8K
    try:
        logger.info("Starting GSM8K evaluation")
        accuracy = evaluate_gsm8k(model, params, args, logger)
        logger.info(f"Evaluation completed with accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Clearing JAX backend caches")
    jax.clear_caches()
    logger.info("Done")


if __name__ == "__main__":
    main() 