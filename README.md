# AI Model Fine-Tuning Pipeline

This repository contains a comprehensive pipeline for fine-tuning various AI models:

1. DeepSeek-Coder model for code generation
2. FLAN-UL2 model for text and story generation

The pipeline includes data preprocessing, training with PEFT and optimization techniques, advanced evaluation, and results visualization.

## Features

- **Code Generation (DeepSeek-Coder)**:
  - Preprocessing of multiple code datasets (CodeSearchNet, CodeAlpaca, MBPP, etc.)
  - Dataset deduplication to remove redundant prompt+completion pairs
  - Consistent formatting for fine-tuning
  - **Multi-language support** with automatic processing of all programming languages
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Optimization with Unsloth for faster training
- 4-bit quantization for memory efficiency
- Comprehensive evaluation on HumanEval and MBPP benchmarks
- Advanced code quality metrics (complexity, linting, semantic similarity)
- Runtime performance metrics (execution time, memory usage)
- Visualization of training metrics and evaluation results

- **Text Generation (FLAN-UL2)**:

  - Preprocessing of text generation datasets (OpenAssistant, GPTeacher, Pile, etc.)
  - Support for creative writing and story generation (WritingPrompts)
  - Persona chat support for character-based text generation
  - Multi-language support where available

- **Common Features**:
  - Parameter-Efficient Fine-Tuning (PEFT) with LoRA
  - Optimization with Unsloth for faster training
  - 4-bit quantization for memory efficiency
  - Comprehensive evaluation metrics
  - Google Drive integration for model checkpoints
  - DeepSpeed ZeRO-3 optimization for large models

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
.
├── config/                       # Configuration files
│   ├── dataset_config.json       # Code dataset configuration
│   ├── dataset_config_text.json  # Text dataset configuration
│   ├── training_config.json      # Code training configuration
│   └── training_config_text.json # Text training configuration
├── data/                         # Data directory
│   ├── processed/                # Processed datasets
│   └── raw/                      # Raw datasets
├── models/                       # Fine-tuned models
├── results/                      # Evaluation results
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   └── processors/           # Dataset processors
│   │       └── text_processors.py# Text dataset processors
│   ├── training/                 # Training modules
│   │   └── text_trainer.py       # FLAN-UL2 trainer
│   ├── evaluation/               # Evaluation modules
│   └── utils/                    # Utility modules
├── visualizations/               # Visualization outputs
├── main.py                       # Code generation pipeline script
├── train_text_flan.py            # Text generation pipeline script
├── train_text_flan.sh            # Text fine-tuning shell script
└── requirements.txt              # Dependencies
```

## Updates and Fixes

### Recent Improvements (2024)

- **New: FLAN-UL2 Text Generation Fine-Tuning**:

  - Added support for fine-tuning Google's FLAN-UL2 20B model for text generation
  - Implemented specialized dataset processors for text generation datasets
  - Memory-optimized training with DeepSpeed ZeRO-3 and CPU offloading
  - Support for story generation, instruction following, and chat
  - Integration with Google Drive for storing model checkpoints

- **Enhanced Google Drive Integration**:

  - Fixed authorization issues related to missing redirect_uri parameter
  - Added setup script for easier credential configuration
  - Improved authentication flow to handle headless environments
  - Added test script to verify Drive integration is working properly
  - Comprehensive documentation in [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md)
  - Fixed duplicate directory creation in Google Drive
  - Added automatic overwriting of existing files and folders to prevent duplicates
  - Improved file ID caching and clearing to ensure latest versions are accessed
  - Added proper error handling for 404 errors when files don't exist during deletion attempts
  - Improved logging for better troubleshooting
  - Added proper error handling for authentication and file operations
  - **NEW: Added dataset checking to avoid redundant processing by detecting pre-processed datasets on Drive**
  - **NEW: Optimized training pipeline to only process datasets not available locally or on Drive**

- **Enhanced Preprocessing Pipeline**:

  - Implemented deduplication of prompt+completion pairs to remove redundant examples
  - Added robust whitespace handling (strip leading/trailing spaces and newlines)
  - Improved filtering of null, empty, or too-short samples
  - Ensured consistent UTF-8 encoding for all datasets
  - Added multi-language support for CodeSearchNet and The Stack datasets (Python, Java, JavaScript, PHP, Ruby, Go, C++, C, C#)
  - **NEW: Enhanced language-specific processing that preserves language information in processed datasets**
  - **NEW: Optimized processing of large multi-language datasets with robust error handling**
  - **NEW: Improved memory management during multi-language dataset processing**
  - **NEW: Smart dataset detection to skip already processed datasets**

- **Improved Training Process**:

  - Added explicit dataset shuffling during training for better model convergence
  - Fixed dataset splitting to ensure proper train/validation/test distribution
  - Implemented proper data formatting with User/Assistant pattern for DeepSeek-Coder
  - **NEW: Added language tracking during training to ensure balanced representation across languages**

- **Fixed dataset preprocessing**:

  - Corrected dataset paths for CodeSearchNet (`code-search-net/code_search_net`), HumanEval (`openai/openai_humaneval`), and other datasets
  - Added robust error handling for missing or empty datasets
  - Added 'enabled' flag support in dataset configuration to selectively process datasets
  - Fixed the configuration for The Stack dataset by using correct data_dir parameter instead of name parameter
  - Improved language-specific handling in dataset processors
  - **NEW: Added parallel language-specific processing for CodeSearchNet to ensure all languages are processed correctly**
  - **NEW: Memory-optimized language processing with incremental saving to prevent OOM errors**

- **Memory Efficiency Improvements**:
  - Fixed batch processing to handle empty datasets
  - Protected against division by zero in batch size calculation
  - Added streaming mode for large datasets
  - **NEW: Implemented memory-efficient multi-language processing with incremental saving**
  - **NEW: Added language distribution tracking to monitor balance across programming languages**

## Usage

### Google Drive Integration

This project includes a robust Google Drive integration system using OAuth2 authentication that works in headless environments like Paperspace notebooks:

```bash
# Set up Google Drive authentication
./setup_google_drive.py

# For headless environments like Paperspace notebooks:
./setup_google_drive.py --headless

# Test the integration
python -m tests.test_google_drive --headless
```

The integration supports:

- **OAuth Authentication** with Out-of-Band (OOB) flow for headless environments
- **Automatic Directory Structure** creation on Google Drive
- **File and Folder Synchronization** between local storage and Drive
- **Transparent Integration** with training scripts via simple command-line flags

For detailed Google Drive setup instructions, see [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md).

### Text Generation with FLAN-UL2

To fine-tune the FLAN-UL2 model for text generation:

```bash
# Run the full fine-tuning pipeline
./train_text_flan.sh
```

Process text datasets only:

```bash
python train_text_flan.py --process_only --data_dir data/processed
```

Train with Google Drive integration:

```bash
python train_text_flan.py --use_drive --drive_base_dir "FlanUL2Text" --data_dir data/processed
```

Push to Hugging Face Hub after training:

```bash
python train_text_flan.py --push_to_hub --hub_model_id your-username/flan-ul2-ft
```

### Code Generation with DeepSeek-Coder

### Complete Pipeline

To run the entire pipeline (preprocessing, training, evaluation, visualization):

```bash
python main.py --mode all
```

### Individual Steps

#### Data Preprocessing

Process all datasets:

```bash
python main.py --mode process
```

Process specific datasets:

```bash
python main.py --mode process --datasets codesearchnet mbpp humaneval
```

Process CodeSearchNet with all languages:

```bash
python main.py --mode process --datasets codesearchnet_all
```

Process specific languages from CodeSearchNet:

```bash
python main.py --mode process --datasets codesearchnet_python codesearchnet_java codesearchnet_javascript
```

For memory-efficient processing:

```bash
python main.py --mode process --streaming --no_cache
```

#### Training

Train the model with the configuration in `config/training_config.json`:

```bash
python main.py --mode train
```

Train with all CodeSearchNet languages (recommended approach):

```bash
python main.py --mode train --datasets codesearchnet_all code_alpaca humaneval
```

Enable Google Drive integration for larger datasets:

```bash
python main.py --mode train --use_drive --datasets codesearchnet_all code_alpaca mbpp
```

Push to Hugging Face Hub after training:

```bash
python main.py --mode train --push_to_hub --hub_model_id your-username/model-name
```

#### Fast Training with The Stack

For efficient direct training using The Stack dataset with language filters (time-constrained):

```bash
# NEW: Integrated quick-stack mode (recommended approach)
python main_api.py --mode quick-stack --auto-time --use_drive_api --credentials_path credentials.json --headless
```

This single command:

1. Automatically calculates time until midnight
2. Optimizes training parameters for your time constraint
3. Filters The Stack for specific languages (python, java, javascript, c, c++, c#, typescript, html, sql, tex, dockerfile)
4. Only uses content with English and Arabic natural language in comments
5. Uses mixed precision (FP16) and gradient accumulation for speed

Advanced options for quick-stack mode:

```bash
# Set specific time limit (in hours)
python main_api.py --mode quick-stack --max-hours 3 --headless

# Skip preprocessing step (use direct loading)
python main_api.py --mode quick-stack --skip-preprocessing --auto-time --headless

# Add specific datasets
python main_api.py --mode quick-stack --datasets the_stack_filtered mbpp code_alpaca --auto-time --headless

# Train with all CodeSearchNet languages and The Stack
python main_api.py --mode quick-stack --datasets the_stack_filtered codesearchnet_all code_alpaca --auto-time --headless
```

For more details, see [DIRECT_STACK_GUIDE.md](DIRECT_STACK_GUIDE.md).

#### Evaluation

Evaluate a fine-tuned model:

```bash
python main.py --mode evaluate --model_path models/deepseek-coder-finetune --base_model deepseek-ai/deepseek-coder-6.7b-base
```

Run specific evaluations:

```bash
python -m src.evaluation.evaluate \
  --model_path models/deepseek-coder-finetune \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --eval_humaneval \
  --eval_mbpp \
  --eval_ds1000 \
  --eval_code_quality \
  --eval_runtime \
  --eval_semantic
```

Evaluate with DS-1000 data science benchmark on specific libraries:

```bash
python -m src.evaluation.evaluate \
  --model_path models/deepseek-coder-finetune \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --eval_ds1000 \
  --ds1000_libraries numpy pandas sklearn
```

Evaluate with custom prompts:

```bash
python -m src.evaluation.evaluate \
  --model_path models/deepseek-coder-finetune \
  --custom_prompts prompts.json
```

#### Visualization

Visualize training metrics and evaluation results:

```bash
python main.py --mode visualize
```

## Configuration

### Dataset Configuration

Edit `config/dataset_config.json` to configure datasets for preprocessing. Example:

```json
{
  "codesearchnet_python": {
    "path": "code-search-net/code_search_net",
    "processor": "codesearchnet",
    "split": "train",
    "language": "python"
  },
  "codesearchnet_all": {
    "path": "code-search-net/code_search_net",
    "processor": "codesearchnet",
    "split": "train",
    "languages": ["python", "java", "javascript", "php", "ruby", "go"],
    "enabled": true
  },
  "mbpp": {
    "path": "mbpp",
    "processor": "mbpp",
    "split": "train"
  },
  "the_stack_filtered": {
    "path": "bigcode/the-stack",
    "processor": "the_stack",
    "split": "train",
    "languages": [
      "python",
      "java",
      "javascript",
      "c",
      "cpp",
      "c-sharp",
      "typescript",
      "html",
      "sql",
      "tex",
      "dockerfile"
    ],
    "natural_languages": ["en", "ar"],
    "sampling_ratio": 0.05,
    "max_samples": 30000,
    "max_processing_time": 180,
    "enabled": true
  }
}
```

### Memory-Optimized Training Commands

For training with large multi-language datasets, use these memory-optimized commands:

#### Low-Memory Processing (8GB+ RAM)

```bash
# Process CodeSearchNet with all languages
python main.py --mode process --datasets codesearchnet_all --streaming --no_cache --max_samples 5000
```

#### Medium-Memory Processing (16GB+ RAM)

```bash
# Process CodeSearchNet with all languages and higher sample count
python main.py --mode process --datasets codesearchnet_all humaneval mbpp --streaming --max_samples 10000
```

#### High-Memory Processing (32GB+ RAM)

```bash
# Process all datasets with optimal settings
python main.py --mode process --streaming --max_samples 20000
```

### Multi-Language Fine-Tuning Commands

For fine-tuning with specific language combinations:

```bash
# Fine-tune on Python and JavaScript only
python main.py --mode train --datasets codesearchnet_python codesearchnet_javascript code_alpaca

# Fine-tune on all supported languages
python main.py --mode train --datasets codesearchnet_all code_alpaca humaneval

# Fine-tune with The Stack filtered dataset plus CodeSearchNet
python main_api.py --mode quick-stack --datasets the_stack_filtered codesearchnet_all --auto-time
```

## Advanced Evaluation Metrics

Our pipeline supports a comprehensive set of evaluation metrics for code generation:

### Functional Correctness

- **Pass@k**: Standard pass rate metrics for HumanEval and MBPP (pass@1, pass@5, pass@10)
- **Tests Passed Rate**: Proportion of test cases passed

### Data Science Benchmarking

- **DS-1000**: Natural and reliable benchmark for evaluating data science code generation across 7 libraries:
  - NumPy, Pandas, TensorFlow, PyTorch, Matplotlib, SciPy, and scikit-learn
  - Evaluates model's ability to solve real-world data science tasks
  - Measures both correctness and execution efficiency

### Code Quality Metrics

- **CodeBLEU**: BLEU-based metric adapted for code similarity measurement
- **Semantic Similarity**: Embedding-based similarity using sentence-transformers
- **Cyclomatic Complexity**: Code complexity measurement
- **Maintainability Index**: Measure of code maintainability
- **Linting Score**: Code quality score using pylint

### Runtime Performance

- **Execution Success Rate**: Percentage of generated code that executes without errors
- **Execution Time**: Average time to execute generated code
- **Memory Usage**: Memory consumption during execution

### Generation Metrics

- **Generation Time**: Average time to generate code
- **Token Statistics**: Length comparison between generated and reference code

## Running on Paperspace Gradient

This pipeline has two options for running on Paperspace Gradient:

### Option 1: Standard Approach (Local Storage)

For basic usage with local storage on Paperspace:

1. Create a new notebook with PyTorch runtime
2. Select an A4000, A5000, or A6000 GPU instance (48GB+ VRAM recommended)
3. Clone this repository:

```bash
git clone https://github.com/yourusername/gen-text-ai.git
cd gen-text-ai
pip install -r requirements.txt
```

### Option 2: Google Drive API Integration (Recommended)

Since Paperspace doesn't support FUSE mounting, we provide a Google Drive API implementation for persistent storage:

```bash
# Install Google Drive API dependencies
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Run the memory-efficient pipeline with Drive API integration
python main_api.py --mode all --streaming --no_cache --use_drive_api --credentials_path credentials.json
```

For detailed instructions on setting up and using the Google Drive API approach, see [PAPERSPACE.md](PAPERSPACE.md).

### Paperspace-Specific Configuration

For optimal performance on Paperspace Gradient:

1. Enable streaming dataset loading to save memory:

```bash
python main.py --mode process --streaming --no_cache
```

2. Adjust the batch size in `config/training_config.json` based on the GPU:

   - A4000 (16GB): per_device_train_batch_size = 2
   - A5000 (24GB): per_device_train_batch_size = 4
   - A6000 (48GB): per_device_train_batch_size = 8

3. Set environment variables for Hugging Face access:

```bash
export HF_TOKEN=your_huggingface_token
```

### Persistent Storage on Paperspace

To save your models and data between Gradient notebook sessions:

1. Use the `--use_drive` flag to save to Google Drive, or
2. Create a persistent Gradient storage and mount it

## Monitoring Training

Training progress can be monitored with Weights & Biases or TensorBoard:

```bash
pip install wandb
wandb login
```

Or for TensorBoard:

```bash
tensorboard --logdir models/deepseek-coder-finetune
```

## Results Visualization

After evaluation, view the generated HTML report in the `visualizations` directory.

## License

[MIT License](LICENSE)
