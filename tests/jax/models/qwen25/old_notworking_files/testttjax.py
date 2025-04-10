import os
import jax
import jax.numpy as jnp
from flax import linen as nn

# Set the environment variable to simulate multiple JAX devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'  # Simulate 8 devices

# Define a simple model using Flax
class SimpleModel(nn.Module):
    def setup(self):
        self.dense = nn.Dense(features=64)

    def __call__(self, x):
        return self.dense(x)

# Create an instance of the model
model = SimpleModel()

# Initialize the model parameters
rng = jax.random.PRNGKey(0)
params = model.init(rng, jax.random.normal(rng, (1, 64)))  # Input shape (1, 64)

# Create dummy input data
input_data = jnp.ones((8, 64))  # 8 batches of 64 features

# Use jax.device_put to ensure the input data is on the correct devices
input_data = jax.device_put(input_data)

# Use pmap to run the model in parallel across the simulated devices
@jax.pmap
def parallel_forward(params, x):
    return model.apply(params, x)

# Run the model with the input data
result = parallel_forward(params, input_data)

# Print the result
print("Output from the model:", result)