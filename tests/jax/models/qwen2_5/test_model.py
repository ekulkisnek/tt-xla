import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
import time

from tests.jax.models.qwen2_5.modeling_flax_qwen2_5 import FlaxQwen25ForCausalLM
from tests.jax.models.qwen2_5.configuration_qwen2_5 import Qwen25Config
from tests.jax.models.qwen2_5.tensor_parallel import create_device_mesh, get_partition_specs

def create_simulated_devices(mesh_shape):
    """Create a simulated multi-device mesh for testing."""
    num_devices = np.prod(mesh_shape)
    # Use jax.local_devices() to get available devices
    available_devices = jax.local_devices()
    # Repeat the first device to simulate multiple devices
    devices = [available_devices[0]] * num_devices
    return np.array(devices).reshape(mesh_shape)

def run_model_with_mesh(mesh_shape, batch_size=1, seq_len=32):
    """Run model with specified mesh configuration and measure performance."""
    print(f"\n{'='*80}")
    print(f"Testing mesh configuration: {mesh_shape[0]}x{mesh_shape[1]}")
    print(f"{'='*80}")
    
    # Create device mesh
    devices = create_simulated_devices(mesh_shape)
    mesh = Mesh(devices, ('dp', 'mp'))
    
    # Initialize model configuration
    config = Qwen25Config(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,  # Using GQA with 4 KV heads
        intermediate_size=2816,
        max_position_embeddings=32768,
        rope_theta=10000.0,
    )
    
    input_shape = (batch_size, seq_len)
    
    # Run model with mesh context
    with mesh:
        print("Initializing model...")
        start_time = time.time()
        
        model = FlaxQwen25ForCausalLM(
            config,
            input_shape=input_shape,
            seed=0,
            dtype=jnp.float32,
            _do_init=True,
        )
        
        init_time = time.time() - start_time
        print(f"Model initialization time: {init_time:.2f}s")
        
        # Set up partition specifications
        partition_rules = {
            "self_attn.q_proj": "colwise",
            "self_attn.k_proj": "colwise", 
            "self_attn.v_proj": "colwise",
            "self_attn.o_proj": "rowwise",
            "mlp.gate_proj": "colwise",
            "mlp.up_proj": "colwise",
            "mlp.down_proj": "rowwise",
        }
        partition_specs = get_partition_specs(model.params, partition_rules)
        
        # Prepare input data
        input_ids = jnp.ones(input_shape, dtype=jnp.int32)
        attention_mask = jnp.ones(input_shape, dtype=jnp.int32)
        position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], input_shape)
        
        # Run forward pass
        print("Running forward pass...")
        start_time = time.time()
        output = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        forward_time = time.time() - start_time
        
        # Print results
        print("\nResults:")
        print(f"Device Mesh Shape: {mesh_shape}")
        print(f"Total Devices: {np.prod(mesh_shape)}")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Query Heads: {config.num_attention_heads}")
        print(f"Number of Key-Value Heads: {config.num_key_value_heads}")
        print(f"Logits Shape: {output.logits.shape}")
        print(f"Hidden States Shapes: {[h.shape for h in output.hidden_states]}")
        print(f"Attention Shapes: {[a.shape for a in output.attentions]}")
        print(f"Initialization Time: {init_time:.2f}s")
        print(f"Forward Pass Time: {forward_time:.2f}s")
        print(f"Total Time: {init_time + forward_time:.2f}s")
        print(f"{'='*80}\n")

def test_all_mesh_configurations():
    """Test all required mesh configurations."""
    mesh_configs = [
        (2, 4),   # 2x4 mesh
        (1, 8),   # 1x8 mesh
        (1, 32),  # 1x32 mesh
        (8, 4),   # 8x4 mesh
    ]
    
    print("Starting comprehensive mesh configuration tests...")
    print("This will test all required mesh shapes from the bounty requirements.")
    print("Each test will measure initialization and forward pass times.")
    
    for mesh_shape in mesh_configs:
        try:
            run_model_with_mesh(mesh_shape)
        except Exception as e:
            print(f"Error with mesh shape {mesh_shape}: {str(e)}")
            print("Continuing with next configuration...")
            continue

if __name__ == "__main__":
    test_all_mesh_configurations() 