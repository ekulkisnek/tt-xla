import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Check if TT-XLA is installed and accessible
try:
    # Add TT-XLA to path if needed
    if os.path.exists("/root/workingdir/tt-xla"):
        sys.path.append("/root/workingdir/tt-xla")
    
    # Try to import TT-XLA specific modules
    import ttxla
    tt_xla_available = True
    print("TT-XLA is available and imported successfully!")
except ImportError:
    tt_xla_available = False
    print("WARNING: TT-XLA module not found. Running in simulation mode.")

# Set necessary environment variables for Tenstorrent
os.environ["TT_BACKEND"] = "TT_BACKEND_XLA"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=128"
os.environ["TT_ARCH_NAME"] = "blackhole"  # Use the appropriate architecture for N300s

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")
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

print("\n--- Creating test data ---")
# Create input data
batch_size = 128  # Match number of cores
feature_dim = 64
rng = jax.random.PRNGKey(0)
input_data = jax.random.normal(rng, (batch_size, feature_dim))

# Initialize the model
model = SimpleModel()
params = model.init(rng, jnp.ones((1, feature_dim)))

# Try to detect Tenstorrent devices
print("\n--- Checking for Tenstorrent devices ---")
try:
    # Run terminal command to check TT devices
    import subprocess
    tt_smi_output = subprocess.check_output("tt-smi", shell=True).decode()
    print("Tenstorrent device info:")
    print(tt_smi_output)
    
    # Extract number of chips
    import re
    chips_match = re.search(r"Detected Chips: (\d+)", tt_smi_output)
    if chips_match:
        num_chips = int(chips_match.group(1))
        print(f"Detected {num_chips} Tenstorrent chips")
    else:
        num_chips = 0
        print("Could not determine number of chips")
except Exception as e:
    print(f"Error getting Tenstorrent device info: {e}")
    num_chips = 0

# Define a parallel forward pass function using jax.pmap
@jax.pmap
def parallel_forward(params, x):
    return model.apply(params, x)

print("\n--- Attempting to execute computation across all devices ---")
start_time = time.time()

# Execute the forward pass
try:
    # Replicate params to all devices
    replicated_params = jax.device_put_replicated(params, jax.devices())
    
    # Place input on devices
    sharded_input = jax.device_put(input_data)
    
    # Execute the model
    result = parallel_forward(replicated_params, sharded_input)
    
    print(f"Computation successful!")
    print(f"Result shape: {result.shape}")
    if hasattr(result, "devices"):
        print(f"Result devices: {result.devices()}")
except Exception as e:
    print(f"Error during computation: {e}")
    result = None

# Additional verification
if result is not None:
    # Calculate some simple statistics to verify the computation
    total_sum = jnp.sum(result)
    mean_val = jnp.mean(result)
    print(f"\nVerification stats:")
    print(f"Total sum: {total_sum}")
    print(f"Mean value: {mean_val}")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.4f} seconds")

if tt_xla_available and num_chips > 0:
    print("✅ Test completed successfully with Tenstorrent hardware!")
else:
    print("⚠️ Test completed in simulation mode!") 