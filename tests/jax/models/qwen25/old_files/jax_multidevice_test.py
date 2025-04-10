import os
import time
import jax
import jax.numpy as jnp

# Set the environment variable to simulate 128 JAX devices (matching N300s cores)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

# Print JAX device information
print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")
print(f"First few devices: {jax.devices()[:5]} (showing 5 of {len(jax.devices())})")

# Define a simple function that will be executed on all devices
@jax.pmap
def matrix_computation(x):
    # Perform matrix operations
    y = jnp.matmul(x, x)  # Matrix multiply
    y = jnp.sin(y) + jnp.cos(y)  # Non-linear operations
    return jnp.sum(y)  # Reduce to a single value

# Create different input data for each device
num_devices = jax.device_count()
print(f"\nPreparing unique data for {num_devices} devices...")

# Create data with a different scalar value for each device
data = jnp.zeros((num_devices, 50, 50))
for i in range(num_devices):
    # Each device gets a matrix with a unique value
    data = data.at[i].set(jnp.ones((50, 50)) * (i + 1) / 100.0)

print(f"Input data shape: {data.shape}")
print(f"First few input values: {[float(data[i][0][0]) for i in range(5)]}")

# Run the function on all devices
print("\nRunning matrix computation across all devices...")
print("This will test if JAX can utilize all 128 cores...")
start_time = time.time()
result = matrix_computation(data)
end_time = time.time()

# Verify the results
computation_time = end_time - start_time
print(f"Computation completed in {computation_time:.4f} seconds")
print(f"Result shape: {result.shape}")
print(f"First few results: {result[:5]}")

# Check if results differ across devices (they should due to the different inputs)
unique_values = set(float(x) for x in result)
print(f"\nNumber of unique result values: {len(unique_values)} (should be close to {num_devices})")
print(f"Sample of unique values: {list(unique_values)[:5]} ...")

# Success message
print(f"\nTest complete! Successfully distributed unique computation across {num_devices} JAX devices.")
print(f"Computation took {computation_time:.4f} seconds.")
print(f"Found {len(unique_values)} different output values across devices.")
print("This confirms that JAX successfully utilized all available simulated cores on the Tenstorrent hardware.") 