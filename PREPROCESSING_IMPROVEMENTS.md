# Preprocessing Improvements

This document summarizes the improvements made to the data preprocessing workflow in the `src/data/preprocessing.py` file.

## Major Improvements

1. **Enhanced Error Handling**

   - Added comprehensive error handling with detailed logging
   - Added traceback output for better debugging
   - Improved exception recovery to continue processing other datasets

2. **Resource Management**

   - Added memory monitoring using `psutil`
   - Implemented proactive garbage collection when memory usage is high
   - Added tokenizer cache clearing to reduce memory pressure

3. **Progress Tracking**

   - Added tqdm progress bars for better visibility during processing
   - Improved logging with timestamps and log levels
   - Added periodic status updates and memory usage reporting

4. **Unified Field Extraction**

   - Added a utility method `_extract_fields` to standardize how fields are extracted across datasets
   - Implemented field name mappings for different dataset structures
   - Improved handling of diverse input formats

5. **Safe Dataset Iteration**

   - Added the `_safe_dataset_iterator` method for reliable dataset iteration
   - Implemented proper limits and error counting to avoid infinite loops
   - Added protection against malformed data

6. **Input Validation**

   - Added comprehensive input validation before processing
   - Improved handling of missing or malformed fields
   - Added type checking and conversion for robust processing

7. **Streaming Mode Improvements**
   - Enhanced handling of streaming datasets with better batching
   - Fixed the "object of type 'int' has no len()" error in streaming mode
   - Improved memory efficiency for large datasets

## Dataset-Specific Improvements

1. **CodeSearchNet**

   - Fixed language handling and improved docstring/code extraction
   - Added support for diverse dataset structures

2. **HumanEval**

   - Improved field extraction and prompt generation
   - Added special handling for entry_point and test fields

3. **MBPP**

   - Enhanced field extraction with more flexible field mapping
   - Fixed batching issues in streaming mode

4. **InstructCode (Magicoder)**

   - Added support for conversation-style formats
   - Fixed field extraction for problem/solution structure
   - Added progress logging during processing

5. **The Stack**
   - Improved permissive license filtering
   - Added better language-specific prompt generation

## Testing

Added a test script (`test_preprocessing.py`) that:

- Tests different preprocessing methods
- Verifies streaming mode functionality
- Handles dataset-specific configurations
- Validates output format and content

## Usage

The improved preprocessing workflow can be tested using:

```bash
python test_preprocessing.py --dataset [dataset_name] --processor [processor_name]
```

Where:

- `dataset_name`: humaneval, mbpp, codesearchnet, instruct_code, the_stack
- `processor_name`: humaneval, mbpp, codesearchnet, instruct_code, the_stack

Example:

```bash
python test_preprocessing.py --dataset humaneval --processor humaneval
```
