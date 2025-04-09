"""
Simple script to test model initialization without loading weights.
"""

import os
import sys
import json
import logging
import argparse
import jax
import jax.numpy as jnp
from transformers import AutoConfig

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

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Qwen2.5 model initialization")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config")
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
    
    # Create config manually if not provided
    if args.config_path:
        try:
            logger.info(f"Loading model configuration from {args.config_path}")
            config = AutoConfig.from_pretrained(args.config_path)
            config_dict = config.to_dict()
        except Exception as e:
            logger.error(f"Error loading model configuration: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
    else:
        # Create a minimal config for testing
        logger.info("Creating minimal model configuration")
        config_dict = {
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "vocab_size": 10000,
            "rms_norm_eps": 1e-6,
        }
    
    logger.info(f"Using model config: {json.dumps(config_dict, indent=2)}")
    
    # Initialize model components
    try:
        logger.info("Initializing model components")
        
        # Create PRNG key for initialization
        key = jax.random.PRNGKey(0)
        
        # Initialize the transformer model
        logger.info("Initializing Qwen2_5Model")
        transformer = Qwen2_5Model(
            config=config_dict,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        logger.info("Qwen2_5Model initialized successfully")
        
        # Initialize the causal LM model
        logger.info("Initializing Qwen2_5ForCausalLM")
        model = Qwen2_5ForCausalLM(
            config=config_dict,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        logger.info("Qwen2_5ForCausalLM initialized successfully")
        
        # Create dummy inputs
        batch_size = 1
        seq_length = 10
        input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        
        # Initialize parameters with a dummy forward pass
        logger.info("Initializing parameters with dummy inputs")
        variables = model.init(
            key,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False
        )
        
        # Log parameter structure
        logger.info("Parameter structure:")
        flat_params = jax.tree_util.tree_map(
            lambda x: getattr(x, "shape", None),
            variables
        )
        logger.info(json.dumps(jax.tree_util.tree_map(lambda x: str(x) if x is not None else None, flat_params), indent=2))
        
        # Try a forward pass with the initialized parameters
        logger.info("Trying forward pass with initialized parameters")
        outputs = model.apply(
            variables,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False
        )
        
        # Check outputs
        logits = outputs[0]
        logger.info(f"Logits shape: {logits.shape}")
        
        logger.info("Model initialization test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 