# Preprocessing Improvements

This document summarizes the improvements made to the data preprocessing workflow in the `src/data/preprocessing.py` file.

## Major Improvements

1. **Enhanced Error Handling**

   - Added comprehensive error handling with detailed logging
   - Added traceback output for better debugging
   - Improved exception recovery to continue processing other datasets
   - **NEW**: Added timeout mechanism to prevent hanging on slow datasets

2. **Resource Management**

   - Added memory monitoring using `psutil`
   - Implemented proactive garbage collection when memory usage is high
   - Added tokenizer cache clearing to reduce memory pressure
   - **NEW**: Added GPU VRAM monitoring and optimization using PyTorch when available
   - **NEW**: Added incremental saving to minimize memory usage for large datasets

3. **Progress Tracking**

   - Added tqdm progress bars for better visibility during processing
   - Improved logging with timestamps and log levels
   - Added periodic status updates and memory usage reporting
   - **NEW**: Reduced terminal output by consolidating progress information
   - **NEW**: Single-line updates for progress bars to avoid terminal flooding
   - **NEW**: Added more frequent progress updates for slower datasets like The Stack

4. **Unified Field Extraction**

   - Added a utility method `_extract_fields` to standardize how fields are extracted across datasets
   - Implemented field name mappings for different dataset structures
   - Improved handling of diverse input formats

5. **Safe Dataset Iteration**

   - Added the `_safe_dataset_iterator` method for reliable dataset iteration
   - Implemented proper limits and error counting to avoid infinite loops
   - Added protection against malformed data
   - **NEW**: Optimized to minimize memory footprint during iteration
   - **NEW**: Added stall detection to identify datasets that are not making progress

6. **Input Validation**

   - Added comprehensive input validation before processing
   - Improved handling of missing or malformed fields
   - Added type checking and conversion for robust processing

7. **Streaming Mode Improvements**
   - Enhanced handling of streaming datasets with better batching
   - Fixed the "object of type 'int' has no len()" error in streaming mode
   - Improved memory efficiency for large datasets
   - **NEW**: Forced streaming mode to minimize RAM/VRAM usage
   - **NEW**: Split large datasets into smaller chunks to avoid OOM errors

## Timeout Mechanisms

1. **Thread-based Timeouts**

   - Added timeout mechanisms using Python's threading to avoid indefinite hanging
   - Implemented configurable timeout durations for each dataset
   - Added automatic recovery when processing exceeds the timeout

2. **Stall Detection**

   - Added detection of stalled processing where no progress is being made
   - Implemented warning messages when no progress is detected for a period
   - Added rate tracking to estimate completion time

3. **Dataset-Specific Timeouts**

   - Added special handling for The Stack dataset which is known to be slow
   - Implemented separate timeout configurations for different dataset types
   - Added progress reporting based on processing rate (examples/second)

## VRAM Optimization

1. **GPU Memory Management**

   - Added detection of GPU availability via PyTorch
   - Implemented monitoring of allocated and reserved VRAM
   - Added automatic `torch.cuda.empty_cache()` calls to free VRAM when needed
   - Segregated system RAM and GPU VRAM monitoring for better resource allocation

2. **Minimized Memory Footprint**

   - Implemented incremental processing with periodic garbage collection
   - Added chunked dataset saving to avoid loading entire datasets into memory
   - Optimized tokenization to process in smaller batches
   - Limited the number of examples held in memory at any time

3. **Minimal Caching**
   - Disabled dataset caching when possible to reduce disk and memory usage
   - Implemented direct streaming from source to avoid redundant data loading
   - Used minimal intermediate storage during processing

## Reduced Terminal Output

1. **Consolidated Progress Information**

   - Used single-line progress updates instead of multiple progress bars
   - Reduced frequency of log messages during normal operation
   - Implemented filtered error logging to avoid terminal flooding
   - Added cleaner timestamps and more concise formats

2. **Focused Reporting**
   - Only reported critical memory usage information
   - Provided summary statistics instead of verbose processing details
   - Added quiet mode for minimal terminal output
   - Reduced duplicate error messages

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
   - **NEW**: Added special timeout handling for The Stack dataset
   - **NEW**: Implemented more frequent progress updates
   - **NEW**: Added processing rate tracking (examples/second)

## Testing

Added a test script (`test_preprocessing.py`) that:

- Tests different preprocessing methods
- Verifies streaming mode functionality
- Handles dataset-specific configurations
- Validates output format and content
- **NEW**: Includes a quiet mode to reduce terminal output
- **NEW**: Uses streaming mode by default to minimize memory usage
- **NEW**: Added configurable timeouts to prevent tests from hanging

## Usage

The improved preprocessing workflow can be tested using:

```bash
python test_preprocessing.py --dataset [dataset_name] --processor [processor_name] [--quiet] [--timeout SECONDS]
```

Where:

- `dataset_name`: humaneval, mbpp, codesearchnet, instruct_code, the_stack
- `processor_name`: humaneval, mbpp, codesearchnet, instruct_code, the_stack
- `--quiet`: Optional flag to reduce terminal output
- `--timeout`: Timeout in seconds (default: 300, set to 0 to disable)

Example:

```bash
python test_preprocessing.py --dataset humaneval --processor humaneval --quiet --timeout 60
```
