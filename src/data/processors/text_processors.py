import os
import logging
import json
from typing import Dict, List, Any, Union, Optional
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

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
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        
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
                "text": f"Instruction: {instruction}\nResponse: {response}",
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
            processed_dataset.save_to_disk(output_path)
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing OpenAssistant dataset: {str(e)}")
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
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        
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
                "text": f"Instruction: {full_instruction}\nResponse: {response}",
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
            processed_dataset.save_to_disk(output_path)
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing GPTeacher dataset: {str(e)}")
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
        # Load the dataset with appropriate split
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        
        def process_example(example):
            """Process a single example from the Pile dataset"""
            text = example.get('text', '')
            
            # For Pile, we'll just use the text directly
            # Since it's not instruction-based, we'll leave instruction/response empty
            return {
                "instruction": "",
                "response": "",
                "text": text,
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
            processed_dataset.save_to_disk(output_path)
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing Pile dataset: {str(e)}")
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
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        
        def process_example(example):
            """Process a single example from the Synthetic-Persona-Chat dataset"""
            instruction = example.get('human', '')
            response = example.get('assistant', '')
            persona = example.get('persona', '')
            
            if persona:
                full_instruction = f"Persona: {persona}\n\nUser: {instruction}"
            else:
                full_instruction = f"User: {instruction}"
                
            return {
                "instruction": full_instruction,
                "response": response,
                "text": f"{full_instruction}\n\nAssistant: {response}",
                "language": "en",
                "source": "synthetic_persona"
            }
            
        # Process the dataset
        logger.info("Processing Synthetic-Persona-Chat dataset examples")
        processed_dataset = dataset.map(process_example)
        
        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: 
            x["instruction"] != "" and x["response"] != "")
        
        # Limit the number of samples if specified
        if max_samples and not streaming:
            processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, "synthetic_persona_processed")
        if not streaming:
            logger.info(f"Saving processed dataset to {output_path}")
            processed_dataset.save_to_disk(output_path)
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing Synthetic-Persona-Chat dataset: {str(e)}")
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
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        
        def process_example(example):
            """Process a single example from the WritingPrompts dataset"""
            prompt = example.get('prompt', '')
            story = example.get('story', '')
            
            return {
                "instruction": f"Writing Prompt: {prompt}\n\nWrite a story based on this prompt:",
                "response": story,
                "text": f"Writing Prompt: {prompt}\n\nWrite a story based on this prompt:\n\n{story}",
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
            processed_dataset.save_to_disk(output_path)
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing WritingPrompts dataset: {str(e)}")
        # Return an empty dataset as a fallback
        return Dataset.from_dict({"instruction": [], "response": [], "text": [], "language": [], "source": []})

# Map of dataset names to processor functions
PROCESSOR_MAP = {
    'openassistant': openassistant_processor,
    'gpteacher': gpteacher_processor,
    'pile': pile_processor,
    'persona_chat': persona_chat_processor,
    'writingprompts': writingprompts_processor
} 