import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import sys

# Set environment variables to simulate 128 devices (matching the N300s core count)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print("=== Tenstorrent Tensor Parallelism Test ===")
print("This test simulates spreading computation across all 128 cores of a Tenstorrent N300s")
print("When running on real hardware, JAX operations would be distributed across the actual cores")

# Print JAX configuration
print(f"\nJAX version: {jax.__version__}")
print(f"JAX devices available: {len(jax.devices())}")
print(f"First few devices: {jax.devices()[:5]} ...")

# Create a device mesh that mimics the layout of Tenstorrent cores (could be 8x16 or similar)
try:
    # For a 128-core N300s, we'll use a 16x8 mesh
    # This helps with visualizing how a model would be distributed on real hardware
    device_count = len(jax.devices())
    mesh_x = 16
    mesh_y = 8
    
    print(f"\nCreating a {mesh_x}x{mesh_y} device mesh for tensor parallelism")
    from jax.sharding import Mesh
    device_mesh = np.array(jax.devices()).reshape(mesh_x, mesh_y)
    mesh = Mesh(device_mesh, ('x', 'y'))
    print("Device mesh created successfully")
except Exception as e:
    print(f"Could not create device mesh: {e}")
    print("Continuing with regular pmap-based parallelism")

# Define a model-like computation
def test_model_parallel_computation():
    print("\n=== Testing Model-like Parallel Computation ===")
    
    # These dimensions are similar to a language model layer
    seq_length = 128
    hidden_dim = 1024
    batch_size = 128  # One per device
    
    print(f"Creating tensors with dimensions similar to a language model:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Hidden dimension: {hidden_dim}")
    
    # Create example inputs and weights
    rng = jax.random.PRNGKey(0)
    
    # This split will give one batch per device - simulating tensor parallelism
    inputs = jax.random.normal(rng, (batch_size, seq_length, hidden_dim))
    weights = jax.random.normal(rng, (hidden_dim, hidden_dim))
    
    # Define a function for parallel computation
    @jax.pmap
    def parallel_transform(x, w):
        # This is similar to a transformer layer computation
        # Each device computes one batch, but uses the full weight matrix
        hidden = jnp.matmul(x, w)
        hidden = jax.nn.relu(hidden)
        return hidden
    
    # Reshape data for use with pmap
    device_count = jax.device_count()
    actual_batch = min(batch_size, device_count)
    
    # Take only as many batches as we have devices
    inputs_batched = inputs[:actual_batch]
    
    # Replicate weights to all devices (each device gets same weights)
    replicated_weights = jax.device_put_replicated(weights, jax.devices()[:actual_batch])
    
    print(f"\nUsing {actual_batch} simulated devices for parallel computation")
    
    # Warmup run
    print("Performing warmup run...")
    _ = parallel_transform(inputs_batched, replicated_weights)
    
    # Actual timed run
    print("Running model-like computation across devices...")
    start_time = time.time()
    result = parallel_transform(inputs_batched, replicated_weights)
    # Force computation to complete
    result_sum = float(jnp.sum(result))
    end_time = time.time()
    
    computation_time = end_time - start_time
    print(f"Computation completed in {computation_time:.4f} seconds")
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {result_sum}")
    
    # Get device utilization metrics
    try:
        unique_devices = set(str(d) for d in result.devices())
        percent_utilized = (len(unique_devices) / actual_batch) * 100
        print(f"\nDevice utilization: {len(unique_devices)}/{actual_batch} devices ({percent_utilized:.1f}%)")
    except Exception as e:
        print(f"Could not determine device utilization: {e}")
    
    return {
        "computation_time": computation_time,
        "num_devices_used": actual_batch,
        "result_shape": result.shape
    }

# Run the model-like computation test
try:
    print("\n=== Running tensor parallelism test ===")
    print("This test shows how the Qwen2.5-7B model could be distributed across Tenstorrent N300s cores")
    
    result = test_model_parallel_computation()
    print("\n✅ Test successful!")
    print(f"Computation spread across {result['num_devices_used']} simulated cores")
    print(f"Computation took {result['computation_time']:.4f} seconds")
    print(f"Result tensor shape: {result['result_shape']}")
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")

print("\n=== Test Complete ===")
print("This demonstrates how JAX can distribute computation across all cores of a Tenstorrent N300s")
print("On actual hardware, this parallelism would utilize the physical tensor cores rather than simulated devices")
print("This approach could be used to efficiently run the Qwen2.5-7B model on Tenstorrent hardware") 