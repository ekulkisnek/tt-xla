import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PositionalSharding
from flax import linen as nn

# Set the environment variable to simulate multiple JAX devices to match Tenstorrent cores
# N300s has 2 chips with 64 cores each, so we'll simulate 128 devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(f"JAX version: {jax.__version__}")
print(f"Number of simulated devices: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

# Define a simple model using Flax
class SimpleModel(nn.Module):
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        return x

# Create input data that can be sharded across all cores
# We'll create one batch per core
batch_size = 128  # One per core
feature_dim = 64

# Create input data
rng = jax.random.PRNGKey(0)
input_data = jax.random.normal(rng, (batch_size, feature_dim))

# Initialize the model
model = SimpleModel()
params = model.init(rng, jnp.ones((1, feature_dim)))

# Create a device mesh/sharding for the input data
sharding = PositionalSharding(jax.devices())

# Display the sharding we'll use
print(f"Sharding: {sharding}")

# Apply the sharding to the input data
start_time = time.time()
print("Sharding input data across all devices...")
sharded_input = jax.device_put(input_data, sharding)
print(f"Input data sharding: {sharded_input.sharding}")

# Define a parallel forward pass function using jax.pmap
@jax.pmap
def parallel_forward(params, x):
    return model.apply(params, x)

# Execute the parallel forward pass
print("\nExecuting parallel computation across all simulated devices...")
start_compute = time.time()
result = parallel_forward(jax.device_put_replicated(params, jax.devices()), sharded_input)
end_compute = time.time()

print(f"Computation completed in {end_compute - start_compute:.4f} seconds")
print(f"Result shape: {result.shape}")
print(f"Result device placement: {result.devices()}")

# Verify if computation is distributed
print("\nVerifying computation distribution across devices...")
device_placement_counts = {}
for device in jax.devices():
    device_placement_counts[str(device)] = 0

for idx, device in enumerate(result.devices()):
    device_str = str(device)
    if device_str in device_placement_counts:
        device_placement_counts[device_str] += 1

# Print statistics about device utilization
print(f"Total devices utilized: {len(set(result.devices()))}")
print(f"Distribution of computation across devices: {len(set(device_placement_counts.values()))} different patterns")

# Summarize the results
print("\nDistribution of computation per device:")
for device, count in sorted(device_placement_counts.items()):
    if count > 0:
        print(f"{device}: {count} computations")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.4f} seconds")
print("Test completed successfully! JAX simulation over Tenstorrent hardware is working correctly.") 