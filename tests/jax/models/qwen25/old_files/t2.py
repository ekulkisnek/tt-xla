import os
import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding

# Set the environment variable to simulate 8 JAX devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

# Verify the number of devices
num_devices = jax.device_count()
print(f"Number of devices available: {num_devices}")

# Create a simple input tensor to distribute across devices
# Adjusting the shape to be compatible with the sharding
input_data = jnp.ones((8, 64, 64, 1))  # Fixed to have a shape compatible with 8 devices

# Ensure the input data is compatible with the sharding strategy
if input_data.ndim != 4:
    raise ValueError(f"Expected input_data to have 4 dimensions, but got {input_data.ndim} dimensions.")

# Check the shape of the input data
print(f"Input data shape: {input_data.shape}")

# Define a sharding strategy for Tenstorrent N300s
sharding = PositionalSharding(jax.devices()[:8])  # Fixed to use 8 devices directly

# Shard the input tensor across the Tenstorrent devices
sharded_input_data = jax.device_put(input_data, sharding)

# Define a simple function to run on all devices
@jax.pmap
def simple_function(x):
    return jnp.sum(x, axis=(1, 2))  # Sum across the last two dimensions

# Run the function
print("\nRunning computation across all devices...")
result = simple_function(sharded_input_data)

# Check the result
print(f"Result shape: {result.shape}")
print(f"Result values: {result}")
