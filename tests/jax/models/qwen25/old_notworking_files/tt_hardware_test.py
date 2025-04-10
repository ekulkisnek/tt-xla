import os
import time
import jax
import jax.numpy as jnp
import subprocess
import sys

# Set necessary environment variables for Tenstorrent 
# Assuming N300s is a Blackhole architecture
print("=== Setting Tenstorrent environment variables ===")
os.environ["TT_BACKEND"] = "TT_BACKEND_XLA"
os.environ["TT_ARCH_NAME"] = "blackhole"
os.environ["XLA_FLAGS"] = "--xla_tenstorrent_rpc_debug=true --xla_tenstorrent_use_tensix=true"

# Attempt to check for Tenstorrent hardware, but don't exit if not found
print("=== Checking for Tenstorrent hardware ===")
try:
    tt_smi_output = subprocess.check_output("tt-smi", shell=True, stderr=subprocess.PIPE).decode()
    if "Detected Chips:" in tt_smi_output:
        print("✅ Tenstorrent hardware detected!")
        print(tt_smi_output.strip())
    else:
        print("⚠️ Could not detect Tenstorrent hardware through tt-smi")
        print("Continuing anyway as hardware may still be available to JAX")
except Exception as e:
    print(f"⚠️ Error running tt-smi: {e}")
    print("Continuing anyway as hardware may still be available to JAX")

# Attempt to load the TT-XLA module
try:
    # Add tt-xla to path if needed
    tt_xla_path = "/root/workingdir/tt-xla"
    if os.path.exists(tt_xla_path):
        if tt_xla_path not in sys.path:
            sys.path.append(tt_xla_path)
    
    # Try to import TT-XLA specific modules if available
    import ttxla
    print("✅ TT-XLA module imported successfully!")
except ImportError as e:
    print(f"⚠️ Could not import TT-XLA module: {e}")
    print("Running with standard JAX...")

# Print JAX configuration
print("\n=== JAX Configuration ===")
print(f"JAX version: {jax.__version__}")
try:
    # This will show if Tenstorrent devices are available to JAX
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Number of devices: {len(devices)}")
    
    # Check if any are Tenstorrent devices
    tt_devices = [d for d in devices if "TT" in str(d) or "tenstor" in str(d).lower()]
    if tt_devices:
        print(f"Found {len(tt_devices)} Tenstorrent device(s): {tt_devices}")
    else:
        print("No explicit Tenstorrent JAX devices found, using default devices")
        tt_devices = []
except Exception as e:
    print(f"Error checking JAX devices: {e}")
    tt_devices = []

# Define a matrix computation to run on the hardware
def matrix_computation(size=100):
    print(f"\n=== Running {size}x{size} matrix computation ===")
    
    # Create input matrices
    matrix_a = jnp.ones((size, size))
    matrix_b = jnp.ones((size, size)) * 2.0
    
    # Time execution
    start_time = time.time()
    
    # Try to explicitly place on TT hardware if detected
    if tt_devices:
        print(f"Placing computation on Tenstorrent device: {tt_devices[0]}")
        matrix_a = jax.device_put(matrix_a, tt_devices[0])
        matrix_b = jax.device_put(matrix_b, tt_devices[0])
    
    # JIT-compile the function for best performance
    @jax.jit
    def matmul(a, b):
        return jnp.matmul(a, b)
    
    # Warmup run
    print("Performing warmup run...")
    _ = matmul(matrix_a, matrix_b)
    
    # Actual computation
    print("Running matrix multiplication...")
    try:
        # Perform matrix multiplication
        result = matmul(matrix_a, matrix_b)
        
        # Force computation to complete
        result_sum = float(jnp.sum(result))
        
    except Exception as e:
        print(f"Error during computation: {e}")
        return None
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    print(f"Computation completed in {computation_time:.4f} seconds")
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {result_sum}")
    
    return {
        "computation_time": computation_time,
        "result_sum": result_sum,
        "result_shape": result.shape
    }

# Run increasingly demanding computations to test hardware throughput
print("\n=== Starting matrix computation tests ===")
print("These tests will determine if JAX is using Tenstorrent hardware")
print("If the tests run significantly faster than on CPU, then it's using Tenstorrent")

for size in [100, 500, 1000, 2000]:
    try:
        result = matrix_computation(size)
        if result:
            print(f"✅ {size}x{size} matrix computation successful")
        else:
            print(f"❌ {size}x{size} matrix computation failed")
    except Exception as e:
        print(f"❌ {size}x{size} matrix computation failed with error: {e}")

print("\n=== Test Complete ===")
print("If computations completed successfully with good performance, JAX is using Tenstorrent hardware.")
print("Compare these run times with CPU-only performance to confirm hardware acceleration.") 