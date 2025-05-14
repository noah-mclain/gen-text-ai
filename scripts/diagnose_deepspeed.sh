#!/bin/bash
# DeepSpeed Diagnostic Script
# Run this script if you encounter DeepSpeed-related issues during training

echo "===== DEEPSPEED DIAGNOSTIC TOOL ====="
echo "This script helps diagnose DeepSpeed configuration issues"
echo

# 1. Check if DeepSpeed is installed
echo "Checking DeepSpeed installation..."
if python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')" 2>/dev/null; then
    echo "✅ DeepSpeed is installed"
else
    echo "❌ DeepSpeed is not installed or has errors. Try reinstalling with:"
    echo "    pip install deepspeed --no-cache-dir"
    echo
fi

# 2. Check environment variables
echo -e "\nChecking environment variables..."
ENV_VARS=(
    "ACCELERATE_USE_DEEPSPEED"
    "ACCELERATE_DEEPSPEED_CONFIG_FILE"
    "ACCELERATE_DEEPSPEED_PLUGIN_TYPE"
    "HF_DS_CONFIG"
    "DS_ACCELERATOR"
)

for var in "${ENV_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo "✅ $var = ${!var}"
    else
        echo "❌ $var is not set"
    fi
done

# 3. Check config file existence
echo -e "\nChecking config files..."
CONFIG_PATHS=(
    "${ACCELERATE_DEEPSPEED_CONFIG_FILE}"
    "${HF_DS_CONFIG}"
    "$(pwd)/ds_config_a6000.json"
    "/notebooks/ds_config_a6000.json"
)

for path in "${CONFIG_PATHS[@]}"; do
    if [ -n "$path" ] && [ -f "$path" ]; then
        echo "✅ Config file exists: $path"
        echo "   Content preview:"
        head -n 10 "$path" | sed 's/^/   /'
        echo "   ..."
    elif [ -n "$path" ]; then
        echo "❌ Config file does not exist: $path" 
    fi
done

# 4. Check for common issues in DeepSpeed config 
echo -e "\nChecking DeepSpeed config for common issues..."
if [ -n "$ACCELERATE_DEEPSPEED_CONFIG_FILE" ] && [ -f "$ACCELERATE_DEEPSPEED_CONFIG_FILE" ]; then
    # Check for zero optimization
    if grep -q "zero_optimization" "$ACCELERATE_DEEPSPEED_CONFIG_FILE"; then
        echo "✅ zero_optimization section found"
        
        # Check ZeRO stage
        ZERO_STAGE=$(grep -o '"stage": [0-9]' "$ACCELERATE_DEEPSPEED_CONFIG_FILE" | grep -o '[0-9]')
        echo "   ZeRO stage: $ZERO_STAGE"

        # Check batch sizes
        MICRO_BATCH=$(grep -o '"train_micro_batch_size_per_gpu": [0-9]*' "$ACCELERATE_DEEPSPEED_CONFIG_FILE" | grep -o '[0-9]*')
        echo "   Micro batch size: $MICRO_BATCH"
    else
        echo "❌ Missing zero_optimization section"
    fi
else
    echo "❌ Cannot check config file - not specified or doesn't exist"
fi

# 5. Verify transformers integration
echo -e "\nChecking transformers DeepSpeed integration..."
python -c "
import sys
try:
    import transformers
    import deepspeed
    print(f'✅ Transformers version: {transformers.__version__}')
    print(f'✅ DeepSpeed version: {deepspeed.__version__}')
    
    # Check for plugin type support
    if hasattr(transformers.deepspeed, 'HfDeepSpeedConfig'):
        print('✅ HfDeepSpeedConfig is available')
        
        # Try to initialize config
        try:
            import os
            config_path = os.environ.get('HF_DS_CONFIG')
            if config_path and os.path.exists(config_path):
                ds_config = transformers.deepspeed.HfDeepSpeedConfig(config_path)
                print('✅ Successfully loaded HfDeepSpeedConfig')
            else:
                print('❌ HF_DS_CONFIG not set or file not found')
        except Exception as e:
            print(f'❌ Error loading DeepSpeed config: {str(e)}')
    else:
        print('❌ HfDeepSpeedConfig not found in transformers.deepspeed')
        
except ImportError as e:
    print(f'❌ Import error: {str(e)}')
except Exception as e:
    print(f'❌ Error: {str(e)}')
"

# 6. Run fix_deepspeed.py script if available
echo -e "\nRunning fix_deepspeed.py to repair configuration..."
if [ -f "scripts/fix_deepspeed.py" ]; then
    python scripts/fix_deepspeed.py
else
    echo "❌ scripts/fix_deepspeed.py not found"
fi

# 7. Run test_deepspeed_config.py if available
echo -e "\nRunning DeepSpeed configuration test..."
if [ -f "scripts/test_deepspeed_config.py" ]; then
    python scripts/test_deepspeed_config.py
else
    echo "❌ scripts/test_deepspeed_config.py not found"
fi

echo -e "\n===== DIAGNOSTIC COMPLETE ====="
echo "If issues persist, consider:"
echo "1. Check GPU memory usage with 'nvidia-smi'"
echo "2. Re-install DeepSpeed with 'pip install --force-reinstall deepspeed'"
echo "3. Try with ZeRO stage 1 instead of 2 or 3"
echo "4. Reduce batch size"
echo "5. Re-run this script with verbose output: bash scripts/diagnose_deepspeed.sh 2>&1 | tee deepspeed_diagnostic_log.txt" 