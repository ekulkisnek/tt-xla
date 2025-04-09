"""
Simple test script to verify that the model initialization works properly.
"""

import os
import sys
import json
import logging
import argparse
import jax
import jax.numpy as jnp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import our model implementation
from model_implementation import Qwen2_5Model, Qwen2_5ForCausalLM

def setup_logging(debug=False):
    """Set up logging with appropriate level."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if debug:
        logger.setLevel(logging.DEBUG)
        # Enable Flax debug logging
        logging.getLogger("flax").setLevel(logging.DEBUG)
        # Set JAX logging to debug
        logging.getLogger("jax").setLevel(logging.DEBUG)
    
    return logger

def load_parameters(weights_path, logger):
    """Load parameters from PyTorch model."""
    logger.info(f"Loading weights from {weights_path}")
    
    # Load PyTorch model to extract weights
    logger.info("Loading PyTorch model to extract weights")
    pt_model = AutoModelForCausalLM.from_pretrained(weights_path)
    
    # Convert PyTorch state dict to NumPy
    logger.info("Converting PyTorch weights to NumPy")
    pt_params = {k: v.cpu().numpy() for k, v in pt_model.state_dict().items()}
    
    # Print key structure
    logger.info("PyTorch model parameter keys:")
    for i, key in enumerate(sorted(pt_params.keys())):
        logger.info(f"  {i}: {key} - shape: {pt_params[key].shape}")
    
    return pt_params

def create_parameter_structure(pt_params, config, logger):
    """Create structured parameter dictionary for JAX model."""
    logger.info("Creating structured parameter dictionary")
    
    # Create a structured parameter dictionary
    params = {
        "params": {
            "transformer": {
                "embed_tokens": {
                    "embedding": pt_params["transformer.wte.weight"]
                },
                "layers": {}
            }
        }
    }
    
    # Add layers
    num_layers = config["num_hidden_layers"]
    for i in range(num_layers):
        layer_params = {
            "input_layernorm": {
                "scale": pt_params[f"transformer.h.{i}.ln_1.weight"]
            },
            "self_attn": {
                "q_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.attn.q_proj.weight"].T
                },
                "k_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.attn.k_proj.weight"].T
                },
                "v_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.attn.v_proj.weight"].T
                },
                "o_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.attn.o_proj.weight"].T
                }
            },
            "post_attention_layernorm": {
                "scale": pt_params[f"transformer.h.{i}.ln_2.weight"]
            },
            "mlp": {
                "gate_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.mlp.gate_proj.weight"].T
                },
                "up_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.mlp.up_proj.weight"].T
                },
                "down_proj": {
                    "kernel": pt_params[f"transformer.h.{i}.mlp.down_proj.weight"].T
                }
            }
        }
        
        params["params"]["transformer"]["layers"][str(i)] = layer_params
    
    # Add final layernorm
    params["params"]["transformer"]["norm"] = {
        "scale": pt_params["transformer.ln_f.weight"]
    }
    
    # Add LM head
    params["params"]["lm_head"] = {
        "kernel": pt_params["lm_head.weight"].T
    }
    
    return params

def test_model(model, params, tokenizer, prompt, logger):
    """Test the model with a simple forward pass."""
    logger.info(f"Testing model with prompt: {prompt}")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="jax")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    try:
        # Forward pass
        logger.info("Running model forward pass")
        outputs = model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False
        )
        
        # Extract logits
        logits = outputs[0]
        logger.info(f"Logits shape: {logits.shape}")
        
        # Get next token prediction
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Decode next token
        next_token_text = tokenizer.decode(next_token)
        logger.info(f"Next token prediction: {next_token_text}")
        
        return True, next_token_text
    
    except Exception as e:
        logger.error(f"Error in model forward pass: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Qwen2.5 model implementation")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt to test with")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    logger.info(f"Starting test with args: {args}")
    
    # Initialize JAX
    try:
        logger.info("Initializing JAX runtime")
        jax.config.update('jax_platforms', 'cpu')
        logger.info(f"JAX runtime initialized with devices: {jax.devices()}")
    except Exception as e:
        logger.error(f"Error initializing JAX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Load model configuration
    try:
        logger.info(f"Loading model configuration from {args.weights_path}")
        config = AutoConfig.from_pretrained(args.weights_path)
        config_dict = config.to_dict()
        logger.info(f"Model config: {json.dumps(config_dict, indent=2)}")
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Load tokenizer
    try:
        logger.info(f"Loading tokenizer from {args.weights_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.weights_path)
        logger.info(f"Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Load parameters from PyTorch model
    try:
        pt_params = load_parameters(args.weights_path, logger)
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Create parameter structure for JAX model
    try:
        params = create_parameter_structure(pt_params, config_dict, logger)
    except Exception as e:
        logger.error(f"Error creating parameter structure: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Initialize model
    try:
        logger.info("Initializing model")
        model = Qwen2_5ForCausalLM(
            config=config_dict,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Test model
    success, result = test_model(model, params, tokenizer, args.prompt, logger)
    
    if success:
        logger.info(f"Test completed successfully")
        logger.info(f"Result: {result}")
        return 0
    else:
        logger.error(f"Test failed: {result}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 