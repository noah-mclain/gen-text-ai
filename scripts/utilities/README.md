# Utilities for Gen-Text-AI

This directory contains utility scripts for the Gen-Text-AI project.

## Hugging Face Token Utility

The `set_hf_token.py` file in this directory is a **redirect** to the main implementation in:

```
src/utils/set_hf_token.py
```

This is the authoritative implementation and should be the only one maintained.

## Directory Structure

The project maintains utilities in the following locations:

- **Main Utilities**: `src/utils/` - This is the primary location for utility modules
- **Script Utilities**: `scripts/utilities/` - Contains standalone utility scripts

## Best Practices

Always import directly from the main utils package:

```python
from src.utils.set_hf_token import set_hf_token
```

This directory exists only for backward compatibility with existing imports.

## Prompt Enhancer

The `prompt_enhancer.py` script improves text and code prompts by fixing common issues:

- Corrects spelling mistakes
- Fixes spacing around punctuation
- Adds missing words to incomplete sentences
- Formats and corrects code blocks
- Provides side-by-side comparison of original vs enhanced prompts

### Usage

```bash
# Enhance a prompt directly from command line
python -m scripts.utilities.prompt_enhancer --prompt "Your prompt text with code and mispelled words"

# Process a prompt from a file
python -m scripts.utilities.prompt_enhancer --file path/to/your/prompt.txt

# Save the enhanced prompt to an output file
python -m scripts.utilities.prompt_enhancer --prompt "Your prompt" --output enhanced_prompt.txt

# Output in different formats
python -m scripts.utilities.prompt_enhancer --prompt "Your prompt" --format json
python -m scripts.utilities.prompt_enhancer --prompt "Your prompt" --format diff
python -m scripts.utilities.prompt_enhancer --prompt "Your prompt" --format text
```

### Options

- `--prompt TEXT` - Provide the prompt text directly
- `--file PATH` - Read prompt from a file
- `--output PATH` - Save enhanced prompt to file
- `--format {text,json,diff,side-by-side}` - Output format (default: side-by-side)
- `--aggressive` - Enable more aggressive enhancement
- `--no-code-fix` - Disable code block formatting
- `--no-add-words` - Disable missing word detection
- `--width N` - Width for side-by-side display (default: 80)

## Other Utilities

This directory contains utilities files for the Gen-Text-AI project.
