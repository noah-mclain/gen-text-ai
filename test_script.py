#!/usr/bin/env python3
"""
Test script to verify proper shell variable quoting
"""

# Simulate the recommended vars including the problematic ones
recommended_vars = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
    "TORCH_NVCC_FLAGS": "-Xfatbin -compress-all",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
}

# Create a test script with the environment variables
with open("test_env_setup.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Test environment variables\n\n")
    
    for key, value in recommended_vars.items():
        # Quote values containing special characters
        if any(c in value for c in ';- '):
            f.write(f"export {key}=\"{value}\"\n")
        else:
            f.write(f"export {key}={value}\n")

print("Created test_env_setup.sh")
print("Run `source test_env_setup.sh` to test") 