import os
import json
import logging
import sys
import time
import re
import tempfile
import subprocess
import traceback
import resource
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import radon.complexity as radon_complexity
import radon.metrics as radon_metrics
from pylint.lint import Run
from pylint.reporters.text import TextReporter

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK downloads are done
try:
    nltk.download('punkt', quiet=True)
except:
    logger.warning("NLTK download failed, but proceeding anyway")

class ModelEvaluator:
    def __init__(self, model_path: str, base_model_path: Optional[str] = None, use_unsloth: bool = False):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model_path: Path to the base model (required if model_path is a PEFT adapter)
            use_unsloth: Whether to use Unsloth for faster inference
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.use_unsloth = use_unsloth and UNSLOTH_AVAILABLE
        self.begin_token = "<｜begin of sentence｜>"
        
        # Load tokenizer and model
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # First, try loading as a regular model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            if self.use_unsloth:
                self.model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    load_in_4bit=True,
                    token=os.environ.get("HF_TOKEN")
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN")
                )
                
        except:
            # If that fails, try loading as a PEFT adapter
            if not self.base_model_path:
                raise ValueError("Base model path must be provided for PEFT adapters")
            
            logger.info(f"Loading as a PEFT adapter with base model {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            
            if self.use_unsloth:
                self.model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.base_model_path,
                    max_seq_length=2048,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    load_in_4bit=True,
                    token=os.environ.get("HF_TOKEN")
                )
                self.model = FastLanguageModel.get_peft_model(
                    self.model, 
                    peft_path=self.model_path
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN")
                )
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()  # Set model to evaluation mode
        logger.info("Model and tokenizer loaded successfully")
        
        # Load sentence-transformers for semantic similarity if available
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_similarity_model = True
        except ImportError:
            logger.warning("sentence-transformers not available, semantic similarity evaluation will be skipped")
            self.has_similarity_model = False
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512, 
                          temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            Generated response
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Set generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Measure generation time
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)
        
        generation_time = time.time() - start_time
        
        # Decode and return generated text (excluding the prompt)
        generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text, generation_time
    
    def _compute_code_bleu(self, reference: str, candidate: str) -> float:
        """
        Compute a simplified version of CodeBLEU score between reference and candidate code.
        
        Args:
            reference: Reference code
            candidate: Generated code
            
        Returns:
            CodeBLEU score
        """
        # Tokenize code using Python tokenizer
        def tokenize_code(code):
            # Handle empty or invalid code
            if not code or not isinstance(code, str):
                return []
            
            # Simple tokenization by splitting on whitespace and preserving 
            # operators and punctuation
            tokens = []
            # Split by spaces first
            for token in re.findall(r'\S+|\s+', code):
                # Further split by operators and punctuation
                for subtok in re.findall(r'[A-Za-z0-9_]+|[^A-Za-z0-9_\s]', token.strip()):
                    if subtok:  # Skip empty tokens
                        tokens.append(subtok)
            return tokens
        
        # Tokenize
        ref_tokens = tokenize_code(reference)
        cand_tokens = tokenize_code(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Calculate BLEU with smoothing
        weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-4 grams
        smoothing = SmoothingFunction().method1
        
        try:
            bleu_score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=smoothing)
            return bleu_score
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {str(e)}")
            return 0.0
    
    def _compute_semantic_similarity(self, reference: str, candidate: str) -> float:
        """
        Compute semantic similarity between reference and candidate code using embeddings.
        
        Args:
            reference: Reference code
            candidate: Generated code
            
        Returns:
            Semantic similarity score (cosine similarity)
        """
        if not self.has_similarity_model:
            return 0.0
            
        try:
            # Generate embeddings
            ref_embedding = self.similarity_model.encode([reference])[0]
            cand_embedding = self.similarity_model.encode([candidate])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([ref_embedding], [cand_embedding])[0][0]
            return similarity
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def _measure_execution_metrics(self, code: str, test_cases: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Measure execution time and memory usage of code.
        
        Args:
            code: Python code to execute
            test_cases: Optional list of test cases to run
            
        Returns:
            Dictionary with execution metrics
        """
        metrics = {
            "execution_success": False,
            "execution_time": 0.0,
            "memory_usage_mb": 0.0,
            "error": None
        }
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Basic code execution without test cases
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Set a timeout of 5 seconds
            stdout, stderr = process.communicate(timeout=5)
            
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["execution_success"] = process.returncode == 0
            
            # Get maximum memory usage (this is approximate)
            metrics["memory_usage_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
            
            # Run test cases if provided
            if test_cases and metrics["execution_success"]:
                test_results = []
                for i, test_case in enumerate(test_cases):
                    try:
                        # Create a temporary file for the test case
                        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as test_file:
                            # Import the code module and run the test
                            module_name = os.path.basename(temp_file_path).replace('.py', '')
                            test_code = f"import sys\nsys.path.append('{os.path.dirname(temp_file_path)}')\n"
                            test_code += f"import {module_name}\n{test_case}"
                            test_file.write(test_code)
                            test_file_path = test_file.name
                        
                        # Run the test
                        test_process = subprocess.run(
                            [sys.executable, test_file_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=5
                        )
                        test_results.append(test_process.returncode == 0)
                        os.unlink(test_file_path)
                    except Exception as e:
                        test_results.append(False)
                
                metrics["test_results"] = test_results
                metrics["tests_passed"] = sum(test_results)
                metrics["tests_total"] = len(test_results)
        
        except subprocess.TimeoutExpired:
            metrics["error"] = "Execution timed out"
        except Exception as e:
            metrics["error"] = str(e)
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return metrics
    
    def _compute_code_complexity(self, code: str) -> Dict[str, Any]:
        """
        Compute code complexity metrics using Radon.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        metrics = {
            "cyclomatic_complexity": 0,
            "maintained_index": 0,
            "halstead_metrics": {}
        }
        
        try:
            # Compute cyclomatic complexity
            try:
                cc_results = radon_complexity.cc_visit(code)
                if cc_results:
                    metrics["cyclomatic_complexity"] = sum(result.complexity for result in cc_results)
                    metrics["avg_cyclomatic_complexity"] = metrics["cyclomatic_complexity"] / len(cc_results) if cc_results else 0
            except:
                pass
            
            # Compute maintainability index
            try:
                mi_result = radon_metrics.mi_visit(code, multi=True)
                if mi_result:
                    metrics["maintained_index"] = mi_result
            except:
                pass
            
            # Compute Halstead metrics
            try:
                h_metrics = radon_metrics.h_visit(code)
                if h_metrics:
                    metrics["halstead_metrics"] = {
                        "h1": h_metrics.h1,
                        "h2": h_metrics.h2,
                        "N1": h_metrics.N1,
                        "N2": h_metrics.N2,
                        "vocabulary": h_metrics.vocabulary,
                        "length": h_metrics.length,
                        "volume": h_metrics.volume,
                        "difficulty": h_metrics.difficulty,
                        "effort": h_metrics.effort,
                        "time": h_metrics.time,
                        "bugs": h_metrics.bugs
                    }
            except:
                pass
        
        except Exception as e:
            logger.warning(f"Error computing code complexity: {str(e)}")
        
        return metrics
    
    def _compute_linting_score(self, code: str) -> Dict[str, Any]:
        """
        Compute linting score using Pylint.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with linting metrics
        """
        metrics = {
            "pylint_score": 0.0,
            "pylint_messages": []
        }
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Capture pylint output
            from io import StringIO
            pylint_output = StringIO()
            reporter = TextReporter(pylint_output)
            
            # Run pylint with reduced checks to focus on important issues
            Run([
                '--disable=all',
                '--enable=E,F,undefined-variable,unused-variable,unused-import,redefined-outer-name',
                '--score=y',
                temp_file_path
            ], reporter=reporter, exit=False)
            
            # Parse output to get score
            output_text = pylint_output.getvalue()
            score_match = re.search(r'Your code has been rated at ([-\d.]+)', output_text)
            if score_match:
                metrics["pylint_score"] = float(score_match.group(1).split('/')[0])
            
            # Parse output to get error messages
            message_lines = [line.strip() for line in output_text.split('\n') if re.match(r'^[A-Z]:.*', line)]
            metrics["pylint_messages"] = message_lines
            metrics["pylint_issues_count"] = len(message_lines)
            
            # Clean up
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.warning(f"Error computing linting score: {str(e)}")
        
        return metrics
    
    def _compute_extended_metrics(self, reference: str, candidate: str, 
                                test_cases: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute extended evaluation metrics for a generated solution.
        
        Args:
            reference: Reference solution
            candidate: Generated solution
            test_cases: Optional list of test cases to run
            
        Returns:
            Dictionary with extended metrics
        """
        metrics = {}
        
        # Text similarity metrics
        metrics["code_bleu"] = self._compute_code_bleu(reference, candidate)
        metrics["semantic_similarity"] = self._compute_semantic_similarity(reference, candidate)
        
        # Code quality metrics
        metrics["complexity"] = self._compute_code_complexity(candidate)
        metrics["linting"] = self._compute_linting_score(candidate)
        
        # Execution metrics
        metrics["execution"] = self._measure_execution_metrics(candidate, test_cases)
        
        # Token-level statistics
        ref_tokens = len(self.tokenizer.encode(reference))
        cand_tokens = len(self.tokenizer.encode(candidate))
        metrics["token_stats"] = {
            "reference_length": ref_tokens,
            "candidate_length": cand_tokens,
            "length_ratio": cand_tokens / ref_tokens if ref_tokens > 0 else 0
        }
        
        return metrics
    
    def evaluate_humaneval(self, output_path: str = "results/humaneval_results.json"):
        """
        Evaluate the model on the HumanEval benchmark.
        
        Args:
            output_path: Path to save evaluation results
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Ensure human_eval is installed
            import human_eval
        except ImportError:
            logger.error("human_eval package not installed. Install with 'pip install human-eval'")
            return {"error": "human_eval package not installed"}
        
        from human_eval.data import write_jsonl, read_problems
        from human_eval.evaluation import evaluate_functional_correctness
        
        logger.info("Evaluating on HumanEval benchmark")
        
        # Load HumanEval problems
        problems = read_problems()
        
        # Generate solutions
        samples = []
        generation_metrics = []
        extended_metrics = {}
        
        for task_id, problem in tqdm(problems.items(), desc="Generating solutions"):
            prompt = problem["prompt"]
            formatted_prompt = f"{self.begin_token}User: {prompt}\nAssistant:"
            
            try:
                # Generate solution with timing
                solution, generation_time = self.generate_response(formatted_prompt, max_new_tokens=1024, temperature=0.1)
                
                # Extract test cases
                test_code = problem.get("test", "")
                test_cases = re.findall(r'assert[^;]*;?\n', test_code)
                
                # Calculate extended metrics
                canonical_solution = problem.get("canonical_solution", "")
                sample_metrics = self._compute_extended_metrics(canonical_solution, solution, test_cases)
                extended_metrics[task_id] = sample_metrics
                
                # Record metrics
                generation_metrics.append({
                    "task_id": task_id,
                    "generation_time": generation_time,
                    "length": len(solution)
                })
                
                # Add to samples
                samples.append({
                    "task_id": task_id,
                    "completion": solution
                })
            except Exception as e:
                logger.error(f"Error generating solution for {task_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save generated solutions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_jsonl(output_path, samples)
        
        # Evaluate solutions for pass@k
        results = evaluate_functional_correctness(output_path)
        
        # Add extended metrics
        avg_generation_time = np.mean([m["generation_time"] for m in generation_metrics])
        avg_code_bleu = np.mean([m["code_bleu"] for m in extended_metrics.values()])
        avg_semantic_similarity = np.mean([m["semantic_similarity"] for m in extended_metrics.values()])
        avg_cc = np.mean([m["complexity"].get("cyclomatic_complexity", 0) for m in extended_metrics.values()])
        avg_linting_score = np.mean([m["linting"].get("pylint_score", 0) for m in extended_metrics.values()])
        
        # Add aggregated metrics to results
        results.update({
            "avg_generation_time": float(avg_generation_time),
            "avg_code_bleu": float(avg_code_bleu),
            "avg_semantic_similarity": float(avg_semantic_similarity),
            "avg_cyclomatic_complexity": float(avg_cc),
            "avg_linting_score": float(avg_linting_score),
            "detailed_metrics": extended_metrics
        })
        
        # Save results
        with open(output_path.replace(".json", "_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_mbpp(self, output_path: str = "results/mbpp_results.json"):
        """
        Evaluate the model on the MBPP benchmark.
        
        Args:
            output_path: Path to save evaluation results
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating on MBPP benchmark")
        
        try:
            # Load MBPP dataset
            mbpp_dataset = load_dataset("mbpp", split="test")
        except Exception as e:
            logger.error(f"Error loading MBPP dataset: {str(e)}")
            return {"error": f"Failed to load MBPP dataset: {str(e)}"}
        
        # Generate solutions
        samples = []
        generation_metrics = []
        extended_metrics = {}
        
        for i, example in enumerate(tqdm(mbpp_dataset, desc="Generating solutions")):
            task_id = str(example["task_id"])
            prompt = example["text"]
            
            # Get test cases and ground truth
            test_cases = example.get("test_list", [])
            canonical_solution = example.get("code", "")
            
            formatted_prompt = f"{self.begin_token}User: {prompt}\nAssistant:"
            
            try:
                # Generate solution with timing
                solution, generation_time = self.generate_response(formatted_prompt, max_new_tokens=1024, temperature=0.1)
                
                # Calculate extended metrics
                sample_metrics = self._compute_extended_metrics(canonical_solution, solution, test_cases)
                extended_metrics[task_id] = sample_metrics
                
                # Record metrics
                generation_metrics.append({
                    "task_id": task_id,
                    "generation_time": generation_time,
                    "length": len(solution)
                })
                
                # Add to samples
                samples.append({
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": solution,
                    "test_cases": test_cases,
                    "canonical_solution": canonical_solution,
                    "metrics": {
                        "generation_time": generation_time,
                        "execution_success": sample_metrics["execution"]["execution_success"],
                        "execution_time": sample_metrics["execution"]["execution_time"],
                        "tests_passed": sample_metrics["execution"].get("tests_passed", 0),
                        "tests_total": sample_metrics["execution"].get("tests_total", len(test_cases)),
                        "code_bleu": sample_metrics["code_bleu"],
                        "semantic_similarity": sample_metrics["semantic_similarity"]
                    }
                })
            except Exception as e:
                logger.error(f"Error generating solution for {task_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save generated solutions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"samples": samples}, f, indent=2)
        
        # Calculate pass@k metrics similar to HumanEval
        # MBPP typically uses a simpler evaluation where we just check if the tests pass
        correct = [s["metrics"]["tests_passed"] == s["metrics"]["tests_total"] 
                  for s in samples if s["metrics"]["tests_total"] > 0]
        
        total = len(correct)
        k_values = [1, 5, 10]
        pass_at_k = {}
        
        for k in k_values:
            if k == 1:
                pass_at_k[f"pass@{k}"] = float(np.mean(correct)) if correct else 0.0
            else:
                # For pass@k where k > 1, we need to adjust for the fact that we only generate
                # one sample per problem with temperature=0.1
                n = 1  # number of samples
                c = sum(correct)
                pass_at_k[f"pass@{k}"] = 1.0 - (1.0 - c / total) ** min(k, n)
        
        # Add extended metrics
        avg_generation_time = np.mean([m["generation_time"] for m in generation_metrics])
        avg_code_bleu = np.mean([m["code_bleu"] for m in extended_metrics.values()])
        avg_semantic_similarity = np.mean([m["semantic_similarity"] for m in extended_metrics.values()])
        tests_passed_rate = np.mean([s["metrics"]["tests_passed"] / max(1, s["metrics"]["tests_total"]) for s in samples])
        avg_cc = np.mean([m["complexity"].get("cyclomatic_complexity", 0) for m in extended_metrics.values()])
        avg_linting_score = np.mean([m["linting"].get("pylint_score", 0) for m in extended_metrics.values()])
        
        # Compilation rate (code that executes without error)
        execution_success_rate = np.mean([1 if m["execution"]["execution_success"] else 0 for m in extended_metrics.values()])
        
        results = {
            **pass_at_k,
            "avg_generation_time": float(avg_generation_time),
            "avg_code_bleu": float(avg_code_bleu),
            "avg_semantic_similarity": float(avg_semantic_similarity),
            "tests_passed_rate": float(tests_passed_rate),
            "execution_success_rate": float(execution_success_rate),
            "avg_cyclomatic_complexity": float(avg_cc),
            "avg_linting_score": float(avg_linting_score),
            "detailed_metrics": extended_metrics
        }
        
        # Save results
        with open(output_path.replace(".json", "_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_code_generation(self, prompts: List[str], output_path: str = "results/code_generation_results.json"):
        """
        Evaluate the model on custom code generation prompts.
        
        Args:
            prompts: List of prompts or dictionary mapping prompt_id to prompt text
            output_path: Path to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating code generation on {len(prompts)} custom prompts")
        
        samples = []
        generation_metrics = []
        extended_metrics = {}
        
        # Convert dictionary to list if needed
        if isinstance(prompts, dict):
            prompt_items = [(id, prompt) for id, prompt in prompts.items()]
        else:
            prompt_items = [(i, prompt) for i, prompt in enumerate(prompts)]
        
        for prompt_id, prompt in tqdm(prompt_items, desc="Generating code"):
            formatted_prompt = f"{self.begin_token}User: {prompt}\nAssistant:"
            
            try:
                # Generate solution with timing
                solution, generation_time = self.generate_response(formatted_prompt, max_new_tokens=1024, temperature=0.7)
                
                # Calculate extended metrics (without reference or test cases)
                sample_metrics = self._compute_extended_metrics("", solution, None)
                extended_metrics[str(prompt_id)] = sample_metrics
                
                # Record metrics
                generation_metrics.append({
                    "prompt_id": prompt_id,
                    "generation_time": generation_time,
                    "length": len(solution)
                })
                
                # Add to samples
                samples.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "completion": solution,
                    "metrics": {
                        "generation_time": generation_time,
                        "execution_success": sample_metrics["execution"]["execution_success"],
                        "execution_time": sample_metrics["execution"]["execution_time"],
                        "cyclomatic_complexity": sample_metrics["complexity"].get("cyclomatic_complexity", 0),
                        "pylint_score": sample_metrics["linting"].get("pylint_score", 0)
                    }
                })
            except Exception as e:
                logger.error(f"Error generating solution for prompt {prompt_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save generated solutions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"samples": samples}, f, indent=2)
        
        # Compile overall metrics
        avg_generation_time = np.mean([m["generation_time"] for m in generation_metrics])
        execution_success_rate = np.mean([1 if m["execution"]["execution_success"] else 0 for m in extended_metrics.values()])
        avg_cc = np.mean([m["complexity"].get("cyclomatic_complexity", 0) for m in extended_metrics.values()])
        avg_linting_score = np.mean([m["linting"].get("pylint_score", 0) for m in extended_metrics.values()])
        avg_execution_time = np.mean([m["execution"]["execution_time"] for m in extended_metrics.values() 
                                    if m["execution"]["execution_success"]])
        avg_memory_usage = np.mean([m["execution"]["memory_usage_mb"] for m in extended_metrics.values() 
                                  if m["execution"]["execution_success"]])
        
        results = {
            "avg_generation_time": float(avg_generation_time),
            "execution_success_rate": float(execution_success_rate),
            "avg_execution_time": float(avg_execution_time),
            "avg_memory_usage_mb": float(avg_memory_usage),
            "avg_cyclomatic_complexity": float(avg_cc),
            "avg_linting_score": float(avg_linting_score),
            "total_samples": len(samples)
        }
        
        # Save results
        with open(output_path.replace(".json", "_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results 