import os
import json
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset
import time
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DS1000Evaluator:
    """
    Evaluator for the DS-1000 benchmark (Data Science tasks).
    DS-1000 is a natural and reliable benchmark for evaluating code generation
    models on data science tasks across 7 Python libraries.
    """
    
    def __init__(self, 
                 model=None, 
                 tokenizer=None, 
                 max_new_tokens=512, 
                 temperature=0.2,
                 top_p=0.95,
                 libraries=None,
                 execution_timeout=30):
        """
        Initialize the DS-1000 evaluator.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            libraries: List of libraries to evaluate (default: all)
                Options: ['numpy', 'pandas', 'tensorflow', 'pytorch', 'matplotlib', 'scipy', 'sklearn']
            execution_timeout: Timeout for code execution in seconds
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.execution_timeout = execution_timeout
        
        # DS-1000 supported libraries
        self.all_libraries = ['numpy', 'pandas', 'tensorflow', 'pytorch', 'matplotlib', 'scipy', 'sklearn']
        self.libraries = libraries if libraries else self.all_libraries
        
        # Validate libraries
        for lib in self.libraries:
            if lib not in self.all_libraries:
                logger.warning(f"Library {lib} is not supported by DS-1000. Skipping.")
                self.libraries.remove(lib)
        
        # Dictionary to store results
        self.results = {}
        
    def load_dataset(self) -> dict:
        """
        Load the DS-1000 dataset from Hugging Face.
        
        Returns:
            Dictionary mapping library names to problem sets
        """
        try:
            logger.info("Loading DS-1000 benchmark dataset...")
            # Load from Hugging Face
            ds1000_dataset = load_dataset("Salesforce/DS-1000")
            
            # Organize dataset by library
            problems_by_library = {}
            for lib in self.libraries:
                if lib in ds1000_dataset:
                    problems_by_library[lib] = ds1000_dataset[lib]
                    logger.info(f"Loaded {len(problems_by_library[lib])} problems for library: {lib}")
                else:
                    logger.warning(f"Library {lib} not found in DS-1000 dataset")
            
            return problems_by_library
        except Exception as e:
            logger.error(f"Error loading DS-1000 dataset: {e}")
            return {}
    
    def generate_solution(self, prompt: str) -> str:
        """
        Generate a solution using the model.
        
        Args:
            prompt: The problem prompt
            
        Returns:
            Generated solution
        """
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not initialized")
            return ""
        
        try:
            # Format input with appropriate prompt
            formatted_prompt = f"Write a Python function to solve the following data science problem:\n\n{prompt}\n\n"
            
            # Tokenize input
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
            if hasattr(self.model, "device"):
                input_ids = input_ids.to(self.model.device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            
            # Decode the output, skipping the input prompt
            prompt_length = input_ids.shape[1]
            generated_ids = output[0][prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.debug(f"Generated solution in {generation_time:.2f} seconds")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return ""
    
    def evaluate_solution(self, problem: dict, solution: str) -> dict:
        """
        Evaluate a solution using the DS-1000 evaluation framework.
        
        Args:
            problem: Problem dictionary from the dataset
            solution: Generated solution
            
        Returns:
            Evaluation results
        """
        # This is a simplified implementation
        # In a real implementation, we would execute the solution
        # against the test cases in the DS-1000 benchmark
        
        result = {
            "problem_id": problem.get("id", "unknown"),
            "library": problem.get("library", "unknown"),
            "pass": False,
            "error": None,
            "execution_time": None
        }
        
        try:
            # Create a temporary file for the solution
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(solution)
                solution_file = f.name
            
            # Execute the solution in a controlled environment
            # This would need to be replaced with proper execution against DS-1000 framework
            import subprocess
            start_time = time.time()
            process = subprocess.run(
                ["python", solution_file], 
                capture_output=True,
                text=True,
                timeout=self.execution_timeout
            )
            execution_time = time.time() - start_time
            
            result["execution_time"] = execution_time
            
            # Check if the solution runs without errors
            if process.returncode == 0:
                # This is just checking if it runs without error
                # A real implementation would check if it passes the DS-1000 tests
                result["pass"] = True
            else:
                result["error"] = process.stderr
            
            # Clean up
            os.unlink(solution_file)
            
        except subprocess.TimeoutExpired:
            result["error"] = f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the DS-1000 benchmark on the model.
        
        Returns:
            Dictionary with benchmark results
        """
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not initialized")
            return {"error": "Model or tokenizer not initialized"}
        
        # Load dataset
        problems_by_library = self.load_dataset()
        
        # Dictionary to store results
        results = {
            "overall": {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_execution_time": 0.0,
                "libraries": {}
            },
            "problems": []
        }
        
        total_time = 0.0
        problem_count = 0
        
        # Initialize library results
        for lib in self.libraries:
            results["overall"]["libraries"][lib] = {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0
            }
        
        # Evaluate each problem
        for lib, problems in problems_by_library.items():
            logger.info(f"Evaluating {len(problems)} problems for library: {lib}")
            
            for i, problem in enumerate(problems):
                logger.info(f"Evaluating problem {i+1}/{len(problems)} for {lib}")
                
                # Generate solution
                solution = self.generate_solution(problem["prompt"])
                
                # Evaluate solution
                eval_result = self.evaluate_solution(problem, solution)
                
                # Save result
                results["problems"].append({
                    "id": problem.get("id", f"{lib}_{i}"),
                    "library": lib,
                    "passed": eval_result["pass"],
                    "execution_time": eval_result["execution_time"],
                    "error": eval_result["error"]
                })
                
                # Update statistics
                results["overall"]["total"] += 1
                results["overall"]["libraries"][lib]["total"] += 1
                
                if eval_result["pass"]:
                    results["overall"]["passed"] += 1
                    results["overall"]["libraries"][lib]["passed"] += 1
                
                if eval_result["execution_time"]:
                    total_time += eval_result["execution_time"]
                    problem_count += 1
        
        # Calculate pass rates
        for lib in self.libraries:
            lib_total = results["overall"]["libraries"][lib]["total"]
            lib_passed = results["overall"]["libraries"][lib]["passed"]
            
            if lib_total > 0:
                results["overall"]["libraries"][lib]["pass_rate"] = lib_passed / lib_total
        
        # Calculate overall pass rate
        if results["overall"]["total"] > 0:
            results["overall"]["pass_rate"] = results["overall"]["passed"] / results["overall"]["total"]
        
        # Calculate average execution time
        if problem_count > 0:
            results["overall"]["avg_execution_time"] = total_time / problem_count
        
        self.results = results
        return results
    
    def save_results(self, output_dir: str) -> str:
        """
        Save benchmark results to a file.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to the results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"ds1000_results_{timestamp}.json")
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"DS-1000 benchmark results saved to {output_file}")
        return output_file

def evaluate_model_on_ds1000(model, tokenizer, output_dir="results/ds1000", **kwargs):
    """
    Evaluate a model on the DS-1000 benchmark.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        output_dir: Directory to save results
        **kwargs: Additional arguments for the DS1000Evaluator
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        # Import torch here to avoid dependency if not needed
        import torch
        
        # Initialize evaluator
        evaluator = DS1000Evaluator(model=model, tokenizer=tokenizer, **kwargs)
        
        # Run benchmark
        results = evaluator.run_benchmark()
        
        # Save results
        evaluator.save_results(output_dir)
        
        return results
    except Exception as e:
        logger.error(f"Error evaluating model on DS-1000: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)} 