#!/usr/bin/env python3

"""
Debug test for Qwen2.5 model with full model configuration for proper testing.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from typing import Dict, List, Any, Optional

# Set default device count if not set
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# Local imports
from tensor_parallel import TensorParallelQwen2ForCausalLM, create_device_mesh
from weight_loading import load_qwen_weights
from verify_gsm8k_scores import setup_tokenizer
from config import load_qwen_config

def print_step(current_step, total_steps, title):
    """Print a section header with the current step."""
    print("\n" + "="*80)
    print(f"[{time.strftime('%H:%M:%S')}] STEP {current_step}/{total_steps}: {title}")
    print("-"*80)
    sys.stdout.flush()

def run_test(model_path: str, verbose: bool = False):
    """Run the tensor parallel test with real weights."""
    # Setup
    print("\n" + "#"*80)
    print("# QWEN2.5-7B TENSOR PARALLELISM DEBUG TEST - WITH FULL MODEL")
    print("#"*80)
    print(f"Model path: {model_path}")
    
    # Print JAX version and device info
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    
    # Try to get memory stats
    try:
        mem_stats = jax.devices()[0].memory_stats()
        if mem_stats and 'bytes_available' in mem_stats:
            mem_gb = mem_stats['bytes_available'] / (1024**3)
            print(f"Available memory: {mem_gb:.2f} GB")
        else:
            print("Memory stats not available")
    except Exception as e:
        print("Memory stats not available")
    
    # Define steps
    total_steps = 6
    current_step = 0
    
    # STEP 1: Import all necessary modules and verify availability
    current_step += 1
    print_step(current_step, total_steps, "Importing required components")
    
    try:
        # Basic sanity checks already passed if we got this far
        print("✅ All required modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing required modules: {e}")
        return False
    
    # STEP 2: Load model configuration
    current_step += 1
    print_step(current_step, total_steps, "Loading model configuration")
    
    try:
        print(f"Loading configuration from {model_path}...")
        config = load_qwen_config(model_path)
        
        # Use the full configuration instead of reducing layers
        print("✅ Configuration loaded successfully")
        print(f"• Hidden size: {config['hidden_size']}")
        print(f"• Layers: {config['num_hidden_layers']}")
        print(f"• Attention heads: {config['num_attention_heads']}")
        print(f"• KV heads: {config['num_key_value_heads']}")
        print(f"• Vocabulary size: {config['vocab_size']}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # STEP 3: Create device mesh
    current_step += 1
    print_step(current_step, total_steps, "Creating device mesh (1x8)")
    
    try:
        mesh_shape = (1, 8)  # 1x8 mesh (use 8 devices)
        print(f"Creating device mesh with shape {mesh_shape}...")
        
        mesh = create_device_mesh(mesh_shape)
        print(f"✅ Device mesh created successfully")
    except Exception as e:
        print(f"❌ Error creating device mesh: {e}")
        return False
    
    # STEP 4: Initialize model
    current_step += 1
    print_step(current_step, total_steps, "Initializing model parameters")
    
    try:
        print("Creating model instance...")
        model = TensorParallelQwen2ForCausalLM(
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        
        print("Initializing default parameters (this may take a moment)...")
        start_time = time.time()
        
        # Create input for initialization
        batch_size = 1
        seq_length = 16
        input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        
        # Initialize with random parameters
        with mesh:
            rng = jax.random.PRNGKey(0)
            init_params = model.init(rng, input_ids)
        
        elapsed = time.time() - start_time
        print(f"✅ Model initialized in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # STEP 5: Load model weights
    current_step += 1
    print_step(current_step, total_steps, "Loading model weights")
    
    try:
        print("Loading weights from disk (this may take several minutes)...")
        print("Please be patient as large weights are being loaded...")
        
        # Set up progress reporting
        start_time = time.time()
        last_report = start_time
        
        def progress_callback(current, total):
            nonlocal last_report
            now = time.time()
            if now - last_report >= 5:  # Report every 5 seconds
                elapsed = now - start_time
                percent = (current / total) * 100 if total > 0 else 0
                print(f"[{time.strftime('%H:%M:%S')}] Loading weights: {current}/{total} parameters ({percent:.1f}%) - {elapsed:.1f}s elapsed")
                last_report = now
                sys.stdout.flush()
        
        # Load weights with progress reporting
        print(f"Using real weights from: {model_path}")
        weights = load_qwen_weights(
            model_path=model_path,
            config=config,
            mesh=mesh,
            param_dtype=jnp.bfloat16,
            debug=verbose,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Weights loaded successfully in {elapsed:.2f} seconds")
        
        # Debug weight structure
        print("\nDebugging weight structure:")
        
        # Check top-level keys
        top_level_keys = list(weights.keys())
        print(f"Top-level keys: {top_level_keys}")
        
        # Create an empty variable for the model parameters
        model_params = {}
        
        # Look for the 'params' key that Flax expects
        if 'params' not in top_level_keys:
            print("Warning: 'params' key missing from weights! Restructuring...")
            
            # Create proper Flax structure (params/{model,lm_head}/...)
            restructured_weights = {'params': {'model': {}, 'lm_head': {}}}
            
            # Get list of embeddings, layers, and norm
            embed_tokens = [k for k in top_level_keys if k.startswith('embed_tokens')]
            layers = [k for k in top_level_keys if k.startswith('layers_')]
            norm = [k for k in top_level_keys if k == 'norm']
            lm_head = [k for k in top_level_keys if k == 'lm_head']
            
            print(f"Found: {len(embed_tokens)} embedding, {len(layers)} layers, {len(norm)} norm, {len(lm_head)} lm_head")
            
            # Process embedding: needs 'weight' -> 'embedding' rename and transpose
            if embed_tokens:
                print("Processing embedding")
                embed_data = weights['embed_tokens']
                if 'weight' in embed_data:
                    # HF format: embed_tokens: { weight: tensor }
                    restructured_weights['params']['model']['embed_tokens'] = {
                        'embedding': embed_data['weight']  # No transpose for embeddings
                    }
                else:
                    # Direct tensor format
                    restructured_weights['params']['model']['embed_tokens'] = {
                        'embedding': embed_data
                    }
            
            # Process all layers (not just the first two)
            for layer_name in layers:
                print(f"Processing {layer_name}")
                layer_idx = layer_name.split('_')[1]  # Extract number from layers_X
                layer_data = weights[layer_name]
                restructured_layer = {}
                
                # Process attention blocks
                if 'self_attn' in layer_data:
                    print(f"  Processing self_attn in {layer_name}")
                    self_attn = layer_data['self_attn']
                    restructured_self_attn = {}
                    
                    # Process q_proj, k_proj, v_proj, o_proj
                    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        if proj in self_attn:
                            print(f"    Processing {proj}")
                            proj_data = self_attn[proj]
                            
                            # Create entry for this projection
                            restructured_self_attn[proj] = {}
                            
                            # Add weight (with transpose)
                            if 'weight' in proj_data:
                                restructured_self_attn[proj]['kernel'] = proj_data['weight'].T
                            elif hasattr(proj_data, 'T'):  # Check if it's a tensor
                                restructured_self_attn[proj]['kernel'] = proj_data.T
                            
                            # Add bias if present
                            if 'bias' in proj_data:
                                restructured_self_attn[proj]['bias'] = proj_data['bias']
                    
                    restructured_layer['self_attn'] = restructured_self_attn
                
                # Process MLP blocks
                if 'mlp' in layer_data:
                    print(f"  Processing mlp in {layer_name}")
                    mlp = layer_data['mlp']
                    restructured_mlp = {}
                    
                    # Process gate_proj, up_proj, down_proj
                    for proj in ['gate_proj', 'up_proj', 'down_proj']:
                        if proj in mlp:
                            print(f"    Processing {proj}")
                            proj_data = mlp[proj]
                            
                            # Create entry for this projection
                            restructured_mlp[proj] = {}
                            
                            # Add weight (with transpose)
                            if 'weight' in proj_data:
                                restructured_mlp[proj]['kernel'] = proj_data['weight'].T
                            elif hasattr(proj_data, 'T'):  # Check if it's a tensor
                                restructured_mlp[proj]['kernel'] = proj_data.T
                            
                            # Add bias if present
                            if 'bias' in proj_data:
                                restructured_mlp[proj]['bias'] = proj_data['bias']
                    
                    restructured_layer['mlp'] = restructured_mlp
                
                # Process normalization layers
                for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                    if norm_name in layer_data:
                        print(f"  Processing {norm_name}")
                        norm_data = layer_data[norm_name]
                        restructured_layer[norm_name] = {}
                        
                        # Add weight (no transpose for 1D tensors)
                        if 'weight' in norm_data:
                            restructured_layer[norm_name]['weight'] = norm_data['weight']
                        else:
                            restructured_layer[norm_name]['weight'] = norm_data
                
                # Add processed layer to model
                restructured_weights['params']['model'][layer_name] = restructured_layer
            
            # Process final norm layer
            if norm and 'norm' in weights:
                print("Processing final norm layer")
                norm_data = weights['norm']
                restructured_weights['params']['model']['norm'] = {}
                
                # Add weight (no transpose for 1D tensors)
                if 'weight' in norm_data:
                    restructured_weights['params']['model']['norm']['weight'] = norm_data['weight']
                else:
                    restructured_weights['params']['model']['norm']['weight'] = norm_data
            
            # Process LM head
            if lm_head and 'lm_head' in weights:
                print("Processing lm_head")
                lm_head_data = weights['lm_head']
                
                # Add weight (with transpose)
                if 'weight' in lm_head_data:
                    restructured_weights['params']['lm_head']['kernel'] = lm_head_data['weight'].T
                else:
                    restructured_weights['params']['lm_head']['kernel'] = lm_head_data.T
            
            # Replace weights with restructured version
            weights = restructured_weights
            print(f"New top-level keys: {list(weights.keys())}")
            
            # Check if restructuring worked
            if 'params' in weights and 'model' in weights['params']:
                print(f"Successfully restructured weights with {len(weights['params']['model'])} model components")
                
                # Check for embed_tokens
                if 'embed_tokens' in weights['params']['model']:
                    print("  - Found embed_tokens")
                    embed_keys = list(weights['params']['model']['embed_tokens'].keys())
                    print(f"    Keys: {embed_keys}")
                
                # Check for layers
                layers = [k for k in weights['params']['model'].keys() if k.startswith('layers_')]
                if layers:
                    first_layer = layers[0]
                    print(f"  - Found {len(layers)} layers")
                    print(f"    First layer ({first_layer}) keys: {list(weights['params']['model'][first_layer].keys())}")
                    
                    # Check for attention components
                    if 'self_attn' in weights['params']['model'][first_layer]:
                        attn = weights['params']['model'][first_layer]['self_attn']
                        print(f"    Self-attention keys: {list(attn.keys())}")
                        
                        # Check for projections
                        if 'q_proj' in attn:
                            q_proj = attn['q_proj']
                            print(f"    q_proj keys: {list(q_proj.keys())}")
                
                # Check for norm
                if 'norm' in weights['params']['model']:
                    print("  - Found final norm")
                    norm_keys = list(weights['params']['model']['norm'].keys())
                    print(f"    Keys: {norm_keys}")
                
                # Check for lm_head
                if 'lm_head' in weights['params']:
                    print("  - Found lm_head")
                    lm_head_keys = list(weights['params']['lm_head'].keys())
                    print(f"    Keys: {lm_head_keys}")
            else:
                print("ERROR: Failed to restructure weights properly")
        
        # Optional: verify parameter shapes
        if verbose:
            print("\nVerifying parameter shapes...")
            flat_params = jax.tree_util.tree_leaves(weights)
            print(f"Total number of parameter tensors: {len(flat_params)}")
            
            # Print shapes of a few parameters
            sample_params = list(jax.tree_util.tree_leaves_with_path(weights))[:5]
            for path, param in sample_params:
                path_str = '.'.join(str(p) for p in path)
                print(f"• {path_str}: shape={param.shape}, dtype={param.dtype}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # STEP 6: Run simple inference
    current_step += 1
    print_step(current_step, total_steps, "Running inference")
    
    try:
        print("Loading tokenizer...")
        tokenizer = setup_tokenizer(model_path)
        if tokenizer is None:
            print("❌ Failed to load tokenizer")
            return False
        
        print("✅ Tokenizer loaded")
        
        # Prepare input
        prompt = "What is the capital of France?"
        print(f"\nRunning inference with prompt: '{prompt}'")
        
        # Format prompt
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="np")
        input_ids = jnp.array(input_ids)
        
        # Apply input sharding
        input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
        sharded_input = jax.device_put(input_ids, input_sharding)
        
        print(f"Input shape: {input_ids.shape}")
        print("Running forward pass...")
        
        # Run inference - with error handling
        try:
            with mesh:
                start_time = time.time()
                outputs = model.apply(weights, sharded_input)
                elapsed = time.time() - start_time
                
            logits = outputs[0]
            print(f"✅ Forward pass completed in {elapsed:.2f} seconds")
            print(f"Output shape: {logits.shape}")
            
            # Get top tokens
            last_token_logits = logits[0, -1, :]
            top_tokens = jnp.argsort(last_token_logits, axis=-1)[-5:][::-1]
            
            print(f"\nTop predicted tokens:")
            for i, token_id in enumerate(top_tokens):
                token = tokenizer.decode([token_id])
                print(f"{i+1}. Token {token_id}: '{token}' ({last_token_logits[token_id]:.2f})")
            
            # Generate a short response
            print(f"\nGenerating a short response (20 tokens)...")
            gen_input_ids = input_ids
            
            for i in range(20):  # Generate 20 tokens to see a proper answer
                print(f"  Generating token {i+1}/20...")
                
                # Run model
                with mesh:
                    current_input = jax.device_put(gen_input_ids, input_sharding)
                    start_time = time.time()
                    outputs = model.apply(weights, current_input)
                    elapsed = time.time() - start_time
                
                # Get next token
                next_token_logits = outputs[0][0, -1, :]
                next_token = jnp.argmax(next_token_logits)
                
                # Add token to sequence
                next_token_array = jnp.array([[next_token.item()]])
                gen_input_ids = jnp.concatenate([gen_input_ids, next_token_array], axis=1)
                
                # Print token
                token_text = tokenizer.decode([next_token.item()])
                print(f"    Token: '{token_text}' (id={next_token.item()}) - {elapsed:.2f}s")
                
                # Check for end of sequence token
                if token_text == tokenizer.eos_token:
                    break
            
            # Decode full response
            generated_text = tokenizer.decode(gen_input_ids[0])
            print(f"\nGenerated text:\n{generated_text}")
            
            print("\n✅ Debug inference test completed successfully!")
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Analyze weights and model expectations to help debug
            print("\nAnalyzing model and weights structure for debugging:")
            
            # Get expected param tree def
            expected_params = jax.eval_shape(model.init, jax.random.PRNGKey(0), input_ids)
            print("Expected parameter structure:")
            expected_flat = jax.tree_util.tree_leaves_with_path(expected_params)
            print(f"Number of expected parameters: {len(jax.tree_util.tree_leaves(expected_params))}")
            
            # Print first few expected param paths
            for i, (path, param) in enumerate(expected_flat[:5]):
                path_str = '/'.join(str(p) for p in path)
                print(f"• Expected: {path_str}: shape={param.shape}, dtype={param.dtype}")
            
            # Check actual params
            print("\nActual parameter structure:")
            actual_flat = jax.tree_util.tree_leaves_with_path(weights)
            print(f"Number of actual parameters: {len(jax.tree_util.tree_leaves(weights))}")
            
            # Print first few actual param paths
            for i, (path, param) in enumerate(actual_flat[:5]):
                path_str = '/'.join(str(p) for p in path)
                print(f"• Actual: {path_str}: shape={param.shape}, dtype={param.dtype}")
            
            return False
    except Exception as e:
        print(f"❌ Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print(f"\n{'#'*80}")
    print(f"# TENSOR PARALLELISM DEBUG TEST COMPLETED SUCCESSFULLY")
    print(f"{'#'*80}")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run minimal tensor parallelism test with real weights"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/lu/Documents/tt-bounty-1/qwen2.5-7b",
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return 1
    
    # Run the test
    success = run_test(args.model_path, args.verbose)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 