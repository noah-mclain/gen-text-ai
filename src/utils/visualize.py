#!/usr/bin/env python3
import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

# Import drive utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_utils import mount_google_drive, setup_drive_directories, get_drive_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsVisualizer:
    def __init__(self, output_dir: str):
        """
        Initialize the MetricsVisualizer.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """Load metrics from a JSON file."""
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def plot_training_loss(self, log_file: str, save_path: Optional[str] = None):
        """
        Plot training loss from a log file.
        
        Args:
            log_file: Path to the training log file (usually trainer_state.json)
            save_path: Path to save the plot (if None, will use output_dir)
        """
        try:
            # Load training log
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Extract loss data
            steps = []
            train_loss = []
            eval_loss = []
            
            for entry in log_data.get("log_history", []):
                if "loss" in entry:
                    steps.append(entry.get("step", 0))
                    train_loss.append(entry.get("loss", 0))
                elif "eval_loss" in entry:
                    eval_loss.append(entry.get("eval_loss", 0))
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps, train_loss, label="Training Loss")
            
            if eval_loss:
                # Interpolate evaluation steps
                eval_steps = np.linspace(min(steps), max(steps), len(eval_loss))
                plt.plot(eval_steps, eval_loss, label="Evaluation Loss", linestyle="--")
            
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title("Training and Evaluation Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.output_dir, "training_loss.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Training loss plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training loss: {str(e)}")
    
    def plot_benchmark_comparison(self, results_files: Dict[str, str], 
                                 benchmark: str, metric: str = "pass@1",
                                 save_path: Optional[str] = None):
        """
        Plot benchmark comparison between different models.
        
        Args:
            results_files: Dictionary mapping model names to result files
            benchmark: Name of the benchmark (e.g., "humaneval", "mbpp")
            metric: Metric to compare (e.g., "pass@1")
            save_path: Path to save the plot (if None, will use output_dir)
        """
        try:
            # Load results
            models = []
            scores = []
            
            for model_name, result_file in results_files.items():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if benchmark in data:
                    if metric in data[benchmark]:
                        models.append(model_name)
                        scores.append(data[benchmark][metric])
            
            if not models:
                logger.warning(f"No data found for benchmark {benchmark} and metric {metric}")
                return
            
            # Plot
            plt.figure(figsize=(10, 6))
            
            # Create bar chart
            sns.barplot(x=models, y=scores)
            
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.title(f"{benchmark.upper()} Benchmark - {metric}")
            plt.ylim(0, max(scores) * 1.2)
            
            # Add value labels
            for i, score in enumerate(scores):
                plt.text(i, score + max(scores) * 0.05, f"{score:.3f}", 
                         ha="center", va="bottom", fontweight="bold")
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.output_dir, f"{benchmark}_{metric}_comparison.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Benchmark comparison plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting benchmark comparison: {str(e)}")
    
    def plot_generation_time(self, results_file: str, save_path: Optional[str] = None):
        """
        Plot code generation time for custom prompts.
        
        Args:
            results_file: Path to the custom evaluation results
            save_path: Path to save the plot (if None, will use output_dir)
        """
        try:
            # Load results
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            samples = data.get("samples", [])
            if not samples:
                logger.warning("No samples found in results file")
                return
            
            # Extract generation times
            prompt_ids = []
            generation_times = []
            
            for sample in samples:
                prompt_ids.append(sample.get("prompt_id", 0))
                generation_times.append(sample.get("generation_time", 0))
            
            # Plot
            plt.figure(figsize=(12, 6))
            
            # Create bar chart
            sns.barplot(x=prompt_ids, y=generation_times)
            
            plt.xlabel("Prompt ID")
            plt.ylabel("Generation Time (seconds)")
            plt.title("Code Generation Time by Prompt")
            
            # Add average line
            avg_time = data.get("results", {}).get("avg_generation_time", 0)
            plt.axhline(y=avg_time, color='r', linestyle='--', 
                       label=f"Average: {avg_time:.2f}s")
            plt.legend()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.output_dir, "generation_time.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Generation time plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting generation time: {str(e)}")
    
    def generate_html_report(self, training_log: str, results_files: List[str], 
                            output_path: Optional[str] = None):
        """
        Generate an HTML report with all metrics and plots.
        
        Args:
            training_log: Path to the training log file
            results_files: List of paths to result files
            output_path: Path to save the HTML report (if None, will use output_dir)
        """
        try:
            import base64
            from io import BytesIO
            
            # Generate all plots
            plot_paths = []
            
            # Training loss plot
            loss_plot_path = os.path.join(self.output_dir, "training_loss.png")
            self.plot_training_loss(training_log, loss_plot_path)
            plot_paths.append(("Training Loss", loss_plot_path))
            
            # Results plots
            for result_file in results_files:
                name = os.path.basename(result_file).replace("_results.json", "")
                
                # Try to detect benchmark type
                if "humaneval" in name.lower():
                    benchmark_plot_path = os.path.join(self.output_dir, f"{name}_pass_at_1.png")
                    self.plot_benchmark_comparison(
                        {name: result_file}, "humaneval", "pass@1", benchmark_plot_path
                    )
                    plot_paths.append((f"{name.upper()} Benchmark", benchmark_plot_path))
                elif "mbpp" in name.lower():
                    benchmark_plot_path = os.path.join(self.output_dir, f"{name}_pass_at_1.png")
                    self.plot_benchmark_comparison(
                        {name: result_file}, "mbpp", "pass@1", benchmark_plot_path
                    )
                    plot_paths.append((f"{name.upper()} Benchmark", benchmark_plot_path))
                elif "custom" in name.lower():
                    gen_time_plot_path = os.path.join(self.output_dir, f"{name}_gen_time.png")
                    self.plot_generation_time(result_file, gen_time_plot_path)
                    plot_paths.append((f"{name.upper()} Generation Time", gen_time_plot_path))
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DeepSeek-Coder Fine-Tuning Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    .plot-container {{ margin: 20px 0; }}
                    .plot-image {{ max-width: 100%; border: 1px solid #ddd; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>DeepSeek-Coder Fine-Tuning Report</h1>
            """
            
            # Add plots
            for title, path in plot_paths:
                try:
                    with open(path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                        
                    html_content += f"""
                    <h2>{title}</h2>
                    <div class="plot-container">
                        <img class="plot-image" src="data:image/png;base64,{encoded_image}" alt="{title}">
                    </div>
                    """
                except Exception as e:
                    logger.error(f"Error embedding image {path}: {str(e)}")
            
            # Add results tables
            html_content += "<h2>Evaluation Results</h2>"
            
            for result_file in results_files:
                name = os.path.basename(result_file).replace("_results.json", "")
                
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    html_content += f"<h3>{name.upper()} Results</h3>"
                    html_content += "<table>"
                    html_content += "<tr><th>Metric</th><th>Value</th></tr>"
                    
                    # Extract results based on structure
                    if isinstance(data, dict):
                        if "results" in data:
                            results = data["results"]
                        else:
                            results = data
                            
                        for metric, value in results.items():
                            if isinstance(value, (int, float)):
                                html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
                    
                    html_content += "</table>"
                except Exception as e:
                    logger.error(f"Error processing results file {result_file}: {str(e)}")
            
            # Close HTML
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            if output_path is None:
                output_path = os.path.join(self.output_dir, "report.html")
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics and evaluation results")
    parser.add_argument("--training_log", type=str, required=True,
                        help="Path to the training log file (trainer_state.json)")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing evaluation result files")
    parser.add_argument("--output_dir", type=str, default="../../visualizations",
                        help="Directory to save visualization plots")
    parser.add_argument("--use_drive", action="store_true", 
                        help="Use Google Drive for storage")
    parser.add_argument("--drive_base_dir", type=str, default="DeepseekCoder",
                        help="Base directory on Google Drive (if using Drive)")
    
    args = parser.parse_args()
    
    # Setup Google Drive if requested
    if args.use_drive:
        logger.info("Attempting to mount Google Drive...")
        if mount_google_drive():
            logger.info(f"Setting up directories in Google Drive under {args.drive_base_dir}")
            drive_base = os.path.join("/content/drive/MyDrive", args.drive_base_dir)
            drive_paths = setup_drive_directories(drive_base)
            
            # Update paths to use Google Drive
            args.output_dir = get_drive_path(args.output_dir, drive_paths["visualizations"], args.output_dir)
            args.results_dir = get_drive_path(args.results_dir, drive_paths["results"], args.results_dir)
            args.training_log = get_drive_path(args.training_log, drive_paths["logs"], args.training_log)
            
            logger.info(f"Visualizations will be saved to {args.output_dir}")
        else:
            logger.warning("Failed to mount Google Drive. Using local storage instead.")
            args.use_drive = False
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = MetricsVisualizer(args.output_dir)
    
    # Get all result files
    result_files = []
    for file in os.listdir(args.results_dir):
        if file.endswith("_results.json") or file == "overall_results.json":
            result_files.append(os.path.join(args.results_dir, file))
    
    # Generate HTML report
    visualizer.generate_html_report(args.training_log, result_files)
    
    logger.info(f"Visualization completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 