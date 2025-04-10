import os
import jax
import jax.numpy as jnp

# Set the environment variable to simulate 8 JAX devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

# Verify the number of devices
num_devices = jax.device_count()
print(f"Number of devices available: {num_devices}")

# Create a simple input tensor to distribute across devices
input_data = jnp.ones((num_devices, 64, 64, 1))  # 8 batches of 64x64 matrices with an additional dimension

# Define a simple function to run on all devices
@jax.pmap
def simple_function(x):
    return jnp.sum(x, axis=(1, 2))  # Sum across the last two dimensions

# Run the function
print("\nRunning computation across all devices...")
result = simple_function(input_data)

# Check the result
print(f"Result shape: {result.shape}")
print(f"Result values: {result}")
