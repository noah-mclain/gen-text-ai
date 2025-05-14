import os
import logging
import json
from typing import Dict, List, Any, Union, Optional
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def openassistant_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dataset:
    """
    Process the OpenAssistant dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading OpenAssistant dataset from {dataset_path}")
    
    try:
        # Load the dataset with appropriate split
        dataset = load_dataset(
            dataset_path, 
            split=split, 
            streaming=streaming,
            token=os.environ.get("HF_TOKEN")
        )
        
        def process_example(example):
            """Process a single example from the OpenAssistant dataset"""
            # Check if it's a prompt (role is "prompter") or response (role is "assistant")
            if example.get('role') == 'prompter':
                instruction = example.get('text', '')
                response = ""
            elif example.get('role') == 'assistant':
                instruction = ""
                response = example.get('text', '')
            else:
                instruction = ""
                response = ""
                
            # Extract the language if available
            lang = example.get('lang', 'en')
            
            return {
                "instruction": instruction,
                "response": response,
                "language": lang,
                "text": f"User: {instruction}\nAssistant: {response}",
                "source": "openassistant"
            }
            
        # Process the dataset
        logger.info("Processing OpenAssistant dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: 
            x["instruction"] != "" or x["response"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "openassistant_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing OpenAssistant dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

def gpteacher_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dataset:
    """
    Process the GPTeacher dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading GPTeacher dataset from {dataset_path}")
    
    try:
        # Load the dataset with appropriate split
        dataset = load_dataset(
            dataset_path, 
            split=split, 
            streaming=streaming,
            token=os.environ.get("HF_TOKEN")
        )
        
        def process_example(example):
            """Process a single example from the GPTeacher dataset"""
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            context = example.get('context', '')
            
            if context:
                full_instruction = f"{context}\n\n{instruction}"
            else:
                full_instruction = instruction
                
            return {
                "instruction": full_instruction,
                "response": response,
                "text": f"User: {full_instruction}\nAssistant: {response}",
                "language": "en",
                "source": "gpteacher"
            }
            
        # Process the dataset
        logger.info("Processing GPTeacher dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: 
            x["instruction"] != "" and x["response"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "gpteacher_general_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing GPTeacher dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

def pile_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dataset:
    """
    Process the Pile dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading Pile dataset from {dataset_path}")
    
    try:
        # Verify zstandard is available
        try:
            import zstandard
            logger.info("zstandard package is available for Pile dataset")
        except ImportError:
            logger.error("Missing required dependency 'zstandard' for Pile dataset")
            logger.error("Install with: pip install zstandard")
            return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})
            
        # Load the dataset with appropriate split
        dataset = load_dataset(
            dataset_path, 
            split=split, 
            streaming=streaming, 
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        def process_example(example):
            """Process a single example from the Pile dataset"""
            text = example.get('text', '')
            
            # For Pile, we'll just use the text directly
            # Since it's not instruction-based, we'll format it as an instruction pair
            return {
                "instruction": "Write a continuation of the following text:",
                "response": text,
                "text": f"User: Write a continuation of the following text:\nAssistant: {text}",
                "language": "en",  # Assuming English, but Pile has mixed languages
                "source": "pile"
            }
            
        # Process the dataset
        logger.info("Processing Pile dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: x["text"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "pile_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing Pile dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

def persona_chat_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dataset:
    """
    Process the Synthetic-Persona-Chat dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading Synthetic-Persona-Chat dataset from {dataset_path}")
    
    try:
        # Load the dataset with appropriate split
        dataset = load_dataset(
            dataset_path, 
            split=split, 
            streaming=streaming,
            token=os.environ.get("HF_TOKEN")
        )
        
        def process_example(example):
            """Process a single example from the Synthetic-Persona-Chat dataset"""
            # Look for different field names - personas might be in different formats
            persona_fields = ['personas', 'persona', 'personality']
            dialogue_fields = ['dialogue', 'dialog', 'conversation', 'utterances']
            
            # Get persona
            persona = None
            for field in persona_fields:
                if field in example and example[field]:
                    persona = example[field]
                    break
                    
            # Get dialogue
            dialogues = None
            for field in dialogue_fields:
                if field in example and example[field]:
                    dialogues = example[field]
                    break
            
            if not persona or not dialogues:
                return {
                    "instruction": "",
                    "response": "",
                    "text": "",
                    "language": "en",
                    "source": "persona_chat"
                }
                
            # Format persona as a string
            if isinstance(persona, list):
                persona_str = "\n".join([f"- {p}" for p in persona if p])
            else:
                persona_str = str(persona)
                
            # Extract conversation turns
            instruction = f"Persona:\n{persona_str}\n\nConversation:\n"
            response = ""
            
            # Extract the conversation
            if dialogues:
                if isinstance(dialogues, list):
                    # Format dialogue as instruction/response pairs
                    conversation = []
                    for i, turn in enumerate(dialogues):
                        if i % 2 == 0:  # User turn (even indices)
                            conversation.append(f"User: {turn}")
                        else:  # Assistant turn (odd indices)
                            conversation.append(f"Assistant: {turn}")
                            
                    instruction += "\n".join(conversation)
                    
                    # Last assistant response (if exists) becomes the target response
                    if len(dialogues) > 1:
                        response = dialogues[-1] if len(dialogues) % 2 == 0 else ""
                else:
                    instruction += str(dialogues)
                
            return {
                "instruction": instruction,
                "response": response,
                "text": f"User: {instruction}\nAssistant: {response}",
                "language": "en",
                "source": "persona_chat"
            }
            
        # Process the dataset
        logger.info("Processing Synthetic-Persona-Chat dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: x["text"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "synthetic_persona_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing Synthetic-Persona-Chat dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

def writingprompts_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    **kwargs
) -> Dataset:
    """
    Process the WritingPrompts dataset for fine-tuning.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading WritingPrompts dataset from {dataset_path}")
    
    try:
        # Load the dataset with appropriate split
        dataset = load_dataset(
            dataset_path, 
            split=split, 
            streaming=streaming,
            token=os.environ.get("HF_TOKEN")
        )
        
        def process_example(example):
            """Process a single example from the WritingPrompts dataset"""
            prompt = example.get('prompt', '')
            story = example.get('story', '')
            
            # Format as instruction/response where the prompt is the instruction
            # and the story is the response
            return {
                "instruction": f"Write a story for the following prompt: {prompt}",
                "response": story,
                "text": f"User: Write a story for the following prompt: {prompt}\nAssistant: {story}",
                "language": "en",
                "source": "writingprompts"
            }
            
        # Process the dataset
        logger.info("Processing WritingPrompts dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: 
            x["instruction"] != "" and x["response"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "writingprompts_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing WritingPrompts dataset: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

def standardize_text_processor(
    dataset_path: str,
    output_dir: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_cache: bool = True,
    force_reprocess: bool = False,
    **kwargs
) -> Dataset:
    """
    Generic processor to standardize any text dataset.
    
    Args:
        dataset_path: Path to the dataset on HuggingFace
        output_dir: Directory to save the processed dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        streaming: Whether to stream the dataset to save memory
        use_cache: Whether to use caching for the dataset
        force_reprocess: Whether to force reprocessing even if the dataset exists
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading dataset from {dataset_path} with standardize_text_processor")
    dataset_name = os.path.basename(dataset_path).replace("/", "_")
    output_path = os.path.join(output_dir, f"{dataset_name}_processed")
    
    # Check if the processed dataset already exists and we're not forcing reprocessing
    if os.path.exists(output_path) and not force_reprocess and not streaming:
        logger.info(f"Processed dataset already exists at {output_path}, loading it")
        try:
            from datasets import load_from_disk
            return load_from_disk(output_path)
        except Exception as e:
            logger.warning(f"Error loading existing dataset: {e}, will reprocess")
    
    try:
        # Attempt to handle various dataset formats
        try:
            # Try loading as a HuggingFace dataset
            dataset = load_dataset(
                dataset_path, 
                split=split, 
                streaming=streaming,
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=True
            )
        except Exception as e1:
            logger.warning(f"Error loading as HuggingFace dataset: {e1}")
            # Try loading as a local path
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(dataset_path)
                if split in dataset:
                    dataset = dataset[split]
            except Exception as e2:
                logger.error(f"Error loading as local dataset: {e2}")
                raise ValueError(f"Could not load dataset from {dataset_path}: {e1}; {e2}")
        
        def process_example(example):
            """Process a single example by standardizing its format"""
            # Try to extract the text field from various possible field names
            text = None
            
            # Check for common field names containing text content
            for field in ['text', 'content', 'code', 'prompt', 'input', 'source']:
                if field in example:
                    text = example[field]
                    break
            
            # If no text field found, try to join all string fields
            if text is None:
                text_pieces = []
                for k, v in example.items():
                    if isinstance(v, str) and len(v.strip()) > 0:
                        text_pieces.append(v)
                text = "\n".join(text_pieces)
            
            # If still no text, use the first field
            if not text and len(example) > 0:
                first_field = list(example.keys())[0]
                text = str(example[first_field])
            
            # Default if all else fails
            if not text:
                text = ""
                
            # Get text length
            length = len(text)
            
            # Determine the language if possible (default to English)
            language = example.get('language', 'en')
            
            # For instruction/response format datasets
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            
            # If instruction and response are available, format accordingly
            if instruction and response:
                processed_text = f"User: {instruction}\nAssistant: {response}"
            else:
                processed_text = text
                
            return {
                "processed_text": processed_text,
                "text": text,
                "length": length,
                "language": language,
                "instruction": instruction,
                "response": response,
                "source": kwargs.get('language', 'generic')
            }
            
        # Process the dataset
        logger.info("Standardizing dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: len(x["processed_text"]) > 0)
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            os.makedirs(output_path, exist_ok=True)
            processed_dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error in standardize_text_processor: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"processed_text": [], "length": [], "language": []})

# Map of dataset names to processor functions
PROCESSOR_MAP = {
    'openassistant': openassistant_processor,
    'gpteacher': gpteacher_processor,
    'pile': pile_processor,
    'persona_chat': persona_chat_processor,
    'writingprompts': writingprompts_processor,
    'standardize_text': standardize_text_processor
} 

def process_datasets(
    dataset_paths: Dict[str, str],
    output_dir: str,
    split: str = 'train',
    streaming: bool = False,
    config: Dict[str, Any] = {},
    failed_count: int = 0,
    successful_count: int = 0,
    processed_datasets: Dict[str, Dataset] = {}
) -> Dict[str, Dataset]:
    """
    Process multiple datasets and save them to disk.
    
    Args:
        dataset_paths: Dictionary of dataset names to dataset paths
        output_dir: Directory to save the processed datasets
        split: Dataset split to use
        streaming: Whether to stream the datasets to save memory
        config: Configuration for processing datasets
        failed_count: Counter for failed processing attempts
        successful_count: Counter for successfully processed datasets
        processed_datasets: Dictionary to store processed datasets
        
    Returns:
        Dictionary of processed datasets
    """
    for dataset_name, dataset_path in dataset_paths.items():
        try:
            # Process with a large timeout for larger datasets
            with tqdm(total=1, desc=f"Processing {dataset_name}", leave=True) as pbar:
                # Check if we have the necessary dependencies before processing
                if dataset_name == 'pile':
                    try:
                        import zstandard
                        logger.info("zstandard package is available for Pile dataset")
                    except ImportError:
                        logger.error("Missing required dependency 'zstandard' for Pile dataset")
                        logger.error("Install with: pip install zstandard")
                        failed_count += 1
                        continue
                
                # Process the dataset
                processor_func = PROCESSOR_MAP[dataset_name]
                processed_dataset = processor_func(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    split=split,
                    streaming=streaming,
                    use_cache=config.get("use_cache", True),
                    max_samples=config.get("max_samples", None)
                )
                
                # Store the processed dataset
                processed_datasets[dataset_name] = processed_dataset
                pbar.update(1)
                successful_count += 1
                logger.info(f"Successfully processed {dataset_name}")
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            logger.error(traceback.format_exc())
            failed_count += 1
    
    return processed_datasets 