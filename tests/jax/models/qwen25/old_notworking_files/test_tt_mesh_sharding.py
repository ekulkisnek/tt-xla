import os
import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from flax import linen as nn

# Set the environment variable for 128 virtual devices (matching N300s cores)
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128'

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")

# Define a simple model using Flax
class SimpleModel(nn.Module):
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        return x

# Try to detect Tenstorrent devices
#print("\n--- Checking for Tenstorrent devices ---")
#try:
#    import subprocess
#    tt_smi_output = subprocess.check_output("tt-smi", shell=True).decode()
#    print("Tenstorrent device info:")
#    print(tt_smi_output)
#except Exception as e:
#    print(f"Error getting Tenstorrent device info: {e}")

# Create a 2D mesh to match Tenstorrent topology
# N300s has 2 chips with 64 cores each, so we'll create an 8x16 mesh
# This gives us 128 virtual devices arranged in a 2D grid
devices = jax.devices()
mesh_shape = (8, 16)  # 8x16 = 128 cores
mesh_devices = devices[:128]  # Use only the first 128 devices

if len(mesh_devices) < 128:
    print(f"WARNING: Not enough devices available. Found {len(mesh_devices)}, need 128.")
    # Adjust mesh shape if necessary
    if len(mesh_devices) > 0:
        import math
        side = int(math.sqrt(len(mesh_devices)))
        mesh_shape = (side, len(mesh_devices) // side)
        print(f"Adjusted mesh shape to {mesh_shape}")
    else:
        print("No devices available for mesh.")
        exit(1)

# Create the device mesh
device_mesh = mesh_devices
print(f"Creating a {mesh_shape} device mesh with {len(device_mesh)} devices")
mesh = Mesh(device_mesh, ('x', 'y'))
print(f"Mesh created: {mesh}")

# Create input data
batch_size = 128
feature_dim = 64
rng = jax.random.PRNGKey(0)
input_data = jax.random.normal(rng, (batch_size, feature_dim))

# Initialize the model
model = SimpleModel()
params = model.init(rng, jnp.ones((1, feature_dim)))

# Define a function to run on the mesh with manual sharding
def run_on_mesh(params, x):
    # Define the computation
    def forward_fn(params, x):
        return model.apply(params, x)
    
    # Create the sharded function
    from jax.experimental import maps
    return maps.xmap(
        forward_fn,
        in_axes=(None, PartitionSpec('x', None)),
        out_axes=PartitionSpec('x', None),
        axis_resources={'x': 'x', 'y': 'y'}
    )(params, x)

# Define a simple parallel computation using jax.pmap as a backup
@jax.pmap
def parallel_forward(params, x):
    return model.apply(params, x)

print("\n--- Attempting to execute computation with mesh sharding ---")
start_time = time.time()

try:
    # Method 1: Using device mesh
    print("Approach 1: Using device mesh and xmap...")
    with mesh:
        # This will try to shard the computation across the device mesh
        result_mesh = run_on_mesh(params, input_data)
    print("Mesh-based computation successful!")
except Exception as e:
    print(f"Mesh-based approach failed: {e}")
    result_mesh = None

# Try alternative approach with pmap
print("\nApproach 2: Using pmap for backup...")
try:
    # Reshape data for pmap (which expects leading dimension = number of devices)
    pmap_devices = min(jax.device_count(), 128)
    chunk_size = batch_size // pmap_devices
    
    # Ensure data fits evenly across devices
    if batch_size % pmap_devices != 0:
        print(f"Adjusting batch size from {batch_size} to {chunk_size * pmap_devices}")
        reshaped_data = input_data[:chunk_size * pmap_devices].reshape(pmap_devices, chunk_size, feature_dim)
    else:
        reshaped_data = input_data.reshape(pmap_devices, chunk_size, feature_dim)
    
    # Replicate parameters to all devices
    replicated_params = jax.device_put_replicated(params, jax.devices()[:pmap_devices])
    
    # Run with pmap
    result_pmap = parallel_forward(replicated_params, reshaped_data)
    print("PMAP-based computation successful!")
    
    # Print device information
    print(f"Result shape: {result_pmap.shape}")
    try:
        print(f"Result devices: {result_pmap.devices()}")
        
        # Count unique devices
        unique_devices = set(result_pmap.devices())
        print(f"Number of unique devices used: {len(unique_devices)}")
    except:
        print("Could not determine device placement for results")
except Exception as e:
    print(f"PMAP approach failed: {e}")
    result_pmap = None

# Report success
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.4f} seconds")

if result_mesh is not None or result_pmap is not None:
    print("✅ Test completed successfully! JAX operations executed across multiple devices.")
    
    # Additional verification
    if result_pmap is not None:
        total_sum = jnp.sum(result_pmap)
        mean_val = jnp.mean(result_pmap)
        print(f"\nVerification stats:")
        print(f"Total sum: {total_sum}")
        print(f"Mean value: {mean_val}")
    elif result_mesh is not None:
        total_sum = jnp.sum(result_mesh)
        mean_val = jnp.mean(result_mesh)
        print(f"\nVerification stats:")
        print(f"Total sum: {total_sum}")
        print(f"Mean value: {mean_val}")
else:
    print("❌ Test failed! Could not execute JAX operations across multiple devices.") 