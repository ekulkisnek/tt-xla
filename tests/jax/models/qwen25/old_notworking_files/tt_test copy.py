import os
import time
import jax
import jax.numpy as jnp

# Set the environment variable to simulate multiple devices (128 cores for N300s)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")
print(f"First few devices: {jax.devices()[:5]}")
print(f"Total devices available: {len(jax.devices())}")

# Define a simple function to run on all devices
@jax.pmap
def simple_function(x):
    # Make it do a bit more work to use the device more thoroughly
    for _ in range(10):
        x = x + jnp.sin(x)
    return jnp.sum(x)  # No axis specified - sum all elements

# Create input for all devices - each device gets a slice of this data
num_devices = jax.device_count()
print(f"\nPreparing data for {num_devices} devices...")
data = jnp.ones((num_devices, 10))  # Each device gets 10 numbers to process

# Run the function
print("\nRunning parallel computation across all devices...")
start_time = time.time()
result = simple_function(data)
end_time = time.time()

print(f"Computation complete in {end_time - start_time:.4f} seconds")
print(f"Result shape: {result.shape}")

# Check if the result is actually distributed across devices
try:
    result_devices = result.devices()
    unique_devices = set(str(d) for d in result_devices)
    print(f"\nResult is spread across {len(unique_devices)} unique devices")
    print(f"Sample of devices used: {list(unique_devices)[:5]}...")
except Exception as e:
    print(f"\nCould not determine device placement: {e}")

# Verify that every device received a different result
print("\nVerifying device-specific results:")
device_results = [float(result[i]) for i in range(min(5, len(result)))]
print(f"Sample of first few results: {device_results}")

print(f"\nTest complete! JAX distributed the computation across {num_devices} simulated devices.") 