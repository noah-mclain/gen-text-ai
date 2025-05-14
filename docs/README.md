# Gen-Text-AI Project Documentation

This directory contains documentation for the Gen-Text-AI project.

## Project Structure

The project is organized into the following key directories:

### Core Directories

- `src/` - Source code for the project

  - `src/data/` - Data processing and loading code
  - `src/training/` - Model training code
  - `src/evaluation/` - Model evaluation code
  - `src/utils/` - Utility functions and helper modules

- `scripts/` - Utility scripts for various tasks

  - See `scripts/README.md` for details on individual scripts

- `config/` - Configuration files for datasets, models, and training

- `data/` - Data directory

  - `data/raw/` - Raw, unprocessed datasets
  - `data/processed/` - Processed datasets ready for training
  - `data/processed/features/` - Extracted features from processed datasets

- `models/` - Trained models and model checkpoints

  - `models/deepseek-coder-finetune-cpu/` - CPU-compatible trained models

- `tests/` - Test code and utilities

### Additional Directories

- `docs/` - Project documentation
- `examples/` - Example usage and demos
- `notebooks/` - Jupyter notebooks for experiments
- `logs/` - Training and execution logs
- `results/` - Evaluation results
- `visualizations/` - Visualizations of training and results
- `archived/` - Archived code and files for reference

## Key Documentation Files

- `GOOGLE_DRIVE_SETUP.md` - Instructions for setting up Google Drive integration
- `README.md` (root) - Main project documentation and getting started guide

## Workflow Overview

1. **Dataset Processing**:

   - Raw datasets are downloaded to `data/raw/`
   - Datasets are processed using scripts in `src/data/`
   - Processed datasets are stored in `data/processed/`
   - Features are extracted to `data/processed/features/`

2. **Model Training**:

   - Training is configured via files in `config/`
   - Training scripts from `scripts/` directory are used
   - Models are trained using code in `src/training/`
   - Trained models are saved to `models/`
   - Logs are written to `logs/`

3. **Model Evaluation**:
   - Models are evaluated using code in `src/evaluation/`
   - Results are stored in `results/`
   - Visualizations are generated in `visualizations/`

## Configuration

The project uses JSON configuration files in the `config/` directory:

- `dataset_config.json` - Dataset configuration
- `training_config.json` - Training configuration

These files control which datasets are used, how they're processed, and how models are trained.

## Google Drive Integration

The project supports using Google Drive for storing and retrieving datasets and models. This is particularly useful for:

1. Sharing preprocessed datasets between team members
2. Persisting trained models across different compute environments
3. Backing up important training results

See `GOOGLE_DRIVE_SETUP.md` for detailed setup instructions.

## Getting Started

Refer to the root `README.md` file for detailed getting started instructions.
