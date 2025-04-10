import os
import time
import jax
import jax.numpy as jnp
from flax import linen as nn

# Set the environment variable to create virtual JAX devices to match Tenstorrent cores (128)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")
print(f"Device list: {jax.devices()[:5]}... (total: {len(jax.devices())})")

# Capture Tenstorrent device info
try:
    import subprocess
    tt_smi_output = subprocess.check_output("tt-smi", shell=True).decode()
    print("\nTenstorrent device info:")
    print(tt_smi_output)
except Exception as e:
    print(f"Error getting Tenstorrent device info: {e}")

# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        return x

# Initialize model
model = SimpleModel()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 16)))

# Define computation function for pmap
@jax.pmap
def forward(params, x):
    return model.apply(params, x)

# Create input data properly shaped for pmap (device_count, ...)
batch_per_device = 1
feature_dim = 16
num_devices = jax.device_count()
input_data = jnp.ones((num_devices, batch_per_device, feature_dim))

# Replicate params to all devices
print("\nDistributing parameters to all devices...")
replicated_params = jax.device_put_replicated(params, jax.devices())

print(f"Input shape: {input_data.shape}")
print(f"Replicated params devices: {type(replicated_params)}")

# Run computation across devices
print("\nRunning computation on all devices...")
start_time = time.time()
result = forward(replicated_params, input_data)
end_time = time.time()

print(f"Computation complete in {end_time - start_time:.4f} seconds")
print(f"Result shape: {result.shape}")

# Verify device mapping
print("\nVerifying device distribution:")
try:
    result_devices = result.devices()
    unique_devices = set(str(d) for d in result_devices)
    print(f"Result is distributed across {len(unique_devices)} unique devices")
    print(f"First few devices: {list(unique_devices)[:5]}")
except Exception as e:
    print(f"Could not determine device placement: {e}")

# Summary
print("\nTest Summary:")
print(f"- Successfully simulated {jax.device_count()} JAX devices")
print(f"- Performed computation across these devices using pmap")
print(f"- Computation time: {end_time - start_time:.4f} seconds")
print("- Test complete!") 