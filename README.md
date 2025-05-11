# DeepSeek-Coder Fine-Tuning Pipeline

This repository contains a comprehensive pipeline for fine-tuning the DeepSeek-Coder model on various code datasets. The pipeline includes data preprocessing, training with PEFT and Unsloth optimization, advanced evaluation, and results visualization.

## Features

- Preprocessing of multiple code datasets (CodeSearchNet, CodeAlpaca, MBPP, etc.)
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Optimization with Unsloth for faster training
- 4-bit quantization for memory efficiency
- Comprehensive evaluation on HumanEval and MBPP benchmarks
- Advanced code quality metrics (complexity, linting, semantic similarity)
- Runtime performance metrics (execution time, memory usage)
- Visualization of training metrics and evaluation results

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config/                  # Configuration files
│   ├── dataset_config.json  # Dataset configuration
│   └── training_config.json # Training configuration
├── data/                    # Data directory
│   ├── processed/           # Processed datasets
│   └── raw/                 # Raw datasets
├── models/                  # Fine-tuned models
├── results/                 # Evaluation results
├── src/                     # Source code
│   ├── data/                # Data processing modules
│   ├── training/            # Training modules
│   ├── evaluation/          # Evaluation modules
│   └── utils/               # Utility modules
├── visualizations/          # Visualization outputs
├── main.py                  # Main pipeline script
└── requirements.txt         # Dependencies
```

## Updates and Fixes

### Recent Improvements (2024)

- **Fixed dataset preprocessing**:

  - Corrected dataset paths for CodeSearchNet (`code-search-net/code_search_net`), HumanEval (`openai/openai_humaneval`), and other datasets
  - Added robust error handling for missing or empty datasets
  - Added 'enabled' flag support in dataset configuration to selectively process datasets

- **Enhanced Google Drive Integration**:

  - Fixed duplicate directory creation in Google Drive
  - Added automatic overwriting of existing files and folders to prevent duplicates
  - Improved logging for better troubleshooting
  - Added proper error handling for authentication and file operations

- **Memory Efficiency Improvements**:
  - Fixed batch processing to handle empty datasets
  - Protected against division by zero in batch size calculation
  - Added streaming mode for large datasets

## Usage

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

For memory-efficient processing:

```bash
python main.py --mode process --streaming --no_cache
```

#### Training

Train the model with the configuration in `config/training_config.json`:

```bash
python main.py --mode train
```

Enable Google Drive integration for larger datasets:

```bash
python main.py --mode train --use_drive
```

Push to Hugging Face Hub after training:

```bash
python main.py --mode train --push_to_hub --hub_model_id your-username/model-name
```

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
  --eval_code_quality \
  --eval_runtime \
  --eval_semantic
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
  "codesearchnet": {
    "path": "code-search-net",
    "processor": "codesearchnet",
    "split": "train"
  },
  "mbpp": {
    "path": "mbpp",
    "processor": "mbpp",
    "split": "train"
  }
}
```

### Training Configuration

Edit `config/training_config.json` to configure the training process. Example:

```json
{
  "model": {
    "base_model": "deepseek-ai/deepseek-coder-6.7b-base",
    "use_unsloth": true,
    "use_4bit": true
  },
  "peft": {
    "lora_alpha": 16,
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "training": {
    "output_dir": "models/deepseek-coder-finetune",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2
  }
}
```

## Advanced Evaluation Metrics

Our pipeline supports a comprehensive set of evaluation metrics for code generation:

### Functional Correctness

- **Pass@k**: Standard pass rate metrics for HumanEval and MBPP (pass@1, pass@5, pass@10)
- **Tests Passed Rate**: Proportion of test cases passed

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
