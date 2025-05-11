# FLAN-UL2 Text Generation Fine-Tuning Guide

This guide explains how to fine-tune the FLAN-UL2 20B model for text and story generation using our optimized pipeline.

## Model Overview

[FLAN-UL2](https://huggingface.co/google/flan-ul2) is a 20 billion parameter language model from Google that has been instruction-tuned on a variety of tasks. It's an excellent foundation model for text generation, creative writing, and instruction following. Our fine-tuning pipeline adapts this model for specialized text generation tasks.

## Datasets

Our pipeline supports the following text generation datasets:

1. **OpenAssistant (OASST1)** - A high-quality assistant-style dataset with conversations
2. **GPTeacher-General-Instruct** - Instruction-tuning dataset covering general knowledge topics
3. **The Pile** - A large, diverse dataset of text for general knowledge
4. **Synthetic-Persona-Chat** - Character-based conversations for persona-based generation
5. **WritingPrompts** - Creative writing dataset for stories and narratives

## Hardware Requirements

- GPU: NVIDIA A6000 or better (48GB+ VRAM recommended)
- CPU: 8+ cores
- RAM: 32GB+ recommended
- Storage: 100GB+ available disk space
- CUDA 11.8+

## Memory Optimization Techniques

The pipeline uses several memory optimization techniques:

1. **4-bit Quantization** - Reduces model size by a factor of 8
2. **LoRA Fine-Tuning** - Parameter-efficient training with adapter modules
3. **DeepSpeed ZeRO-3** - Shards model states across CPU and GPU
4. **Gradient Checkpointing** - Trades compute for memory by recomputing activations
5. **Flash Attention 2** - Efficient attention algorithm for faster training
6. **Lazy Loading** - Processes datasets as needed instead of loading everything at once

## Quick Start

### 1. Install Required Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/gen-text-ai.git
cd gen-text-ai

# Install dependencies
pip install -r requirements.txt

# Install unsloth for optimization (optional but recommended)
pip install unsloth
```

### 2. Set Up HuggingFace Token

Some datasets require authentication:

```bash
export HF_TOKEN=your_huggingface_token
```

### 3. Run the Training Pipeline

```bash
# Process datasets and start training
./train_text_flan.sh
```

Or separately:

```bash
# Process datasets only
python train_text_flan.py --process_only

# Train the model after processing
python train_text_flan.py
```

## Advanced Configuration

### Customizing Datasets

Edit `config/dataset_config_text.json` to enable/disable specific datasets or change parameters:

```json
{
  "openassistant": {
    "enabled": true,  # Set to false to disable this dataset
    "path": "agie-ai/OpenAssistant-oasst1",
    "processor": "openassistant",
    "split": "train"
  },
  // Additional datasets...
}
```

### Training Configuration

Edit `config/training_config_text.json` to adjust training parameters:

```json
{
  "model": {
    "base_model": "google/flan-ul2",
    "use_4bit": true,  # Set to false for higher precision
    // Additional model settings...
  },
  "training": {
    "num_train_epochs": 3,  # Adjust epochs
    "learning_rate": 1e-4,  # Adjust learning rate
    // Additional training settings...
  },
  // Additional settings...
}
```

### Google Drive Integration

To save models and datasets to Google Drive (useful for Colab):

```bash
python train_text_flan.py --use_drive --drive_base_dir "MyTextModels"
```

## Post-Training Usage

### Loading the Fine-Tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("google/flan-ul2", device_map="auto",
                                                 load_in_4bit=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

# Load the LoRA adapters
model = PeftModel.from_pretrained(base_model, "path/to/fine-tuned/model")

# Generate text
input_text = "Write a short story about a robot who learns to paint:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.9)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Evaluation

Our pipeline includes evaluation capabilities for:

1. **Text Quality** - Perplexity, fluency, coherence
2. **Creative Writing** - Narrative structure, creativity, adherence to prompts
3. **Instruction Following** - Accuracy in following complex instructions

## Advanced Topics

### Extending to New Datasets

To add support for a new dataset:

1. Create a processor function in `src/data/processors/text_processors.py`
2. Add the dataset to `config/dataset_config_text.json`
3. Update the processor map in `src/data/processors/text_processors.py`

### Multi-GPU Training

The pipeline supports multi-GPU training using DeepSpeed. Update the configuration:

```json
{
  "training": {
    "deepspeed": {
      "zero_stage": 3,
      "offload_optimizer": {
        "device": "cpu"
      },
      "offload_param": {
        "device": "cpu"
      }
    }
  }
}
```

And run with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_text_flan.py
```

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size, increase gradient accumulation steps, or enable more aggressive memory optimization in the config.
- **Dataset Access Issues**: Ensure your HuggingFace token is set and has proper permissions.
- **Training Instability**: Try reducing the learning rate or increasing warmup steps.

## References

- [FLAN-UL2 Model Card](https://huggingface.co/google/flan-ul2)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
