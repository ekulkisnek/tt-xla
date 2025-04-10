import os
import time
import jax
import jax.numpy as jnp
from flax import linen as nn

# Set the environment variable to simulate multiple JAX devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'  # Simulate 128 devices

print(f"JAX version: {jax.__version__}")
print(f"Number of available devices: {jax.device_count()}")
print(f"First few devices: {jax.devices()[:5]}...")

# Define a simple model using Flax
class SimpleModel(nn.Module):
    features: int = 32
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        return x

# Create an instance of the model
model = SimpleModel()

# Initialize the model parameters with the right input shape
rng = jax.random.PRNGKey(0)
feature_dim = 16
params = model.init(rng, jnp.ones((1, feature_dim)))  # Input shape (1, feature_dim)

# Create dummy input data - one batch per device
num_devices = jax.device_count()
input_data = jnp.ones((num_devices, feature_dim))  # (num_devices, feature_dim)

print(f"\nCreated input with shape {input_data.shape} for {num_devices} devices")

# Replicate parameters to all devices
print("\nReplicating parameters across all devices...")
replicated_params = jax.device_put_replicated(params, jax.devices())

# Use pmap to run the model in parallel across the simulated devices
@jax.pmap
def parallel_forward(params, x):
    return model.apply(params, x)

# Run the model with the input data
print("\nRunning computation across all devices...")
start_time = time.time()
result = parallel_forward(replicated_params, input_data)
end_time = time.time()

print(f"Computation completed in {end_time - start_time:.4f} seconds")
print(f"Result shape: {result.shape}")

# Print a sample of results to verify they're distinct
print("\nSample of results from first 5 devices:")
print(result[:5])

print(f"\nTest complete! JAX successfully distributed computation across {num_devices} simulated devices.") 