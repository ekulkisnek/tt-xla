#!/usr/bin/env python3
"""
Problem detection wrapper for Qwen25 inference.

This script wraps run_inference.py and adds additional problem detection and logging.
It maintains all original functionality while adding warnings for potential issues.

Usage:
    python run_inference_problem_detector.py --model_path /path/to/model/weights --prompt "Hello" --max_tokens 10
"""

import os
import sys
import time
import logging
import argparse
import psutil
import traceback
from typing import Optional, Dict, Any

# Add the directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the original run_inference script
from run_inference import main as original_main, parse_args as original_parse_args

# Configure problem-focused logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qwen25_problem_detector")

# Thresholds for problem detection
THRESHOLDS = {
    "memory_high_gb": 10,
    "model_creation_slow_s": 5,
    "file_processing_slow_s": 10,
    "total_loading_slow_s": 30,
    "warmup_slow_s": 5,
    "generation_slow_s": 10
}

class ProblemDetector:
    """Class to track and detect potential problems during inference."""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = self.get_memory_usage()
        self.problems_found = []
        self.checkpoints = {}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024 * 1024)
        except Exception:
            return 0.0
    
    def checkpoint(self, name: str) -> None:
        """Record a checkpoint with current memory and time."""
        self.checkpoints[name] = {
            'time': time.time() - self.start_time,
            'memory': self.get_memory_usage()
        }
    
    def check_memory(self, label: str) -> None:
        """Check if memory usage is high."""
        current_memory = self.get_memory_usage()
        if current_memory > THRESHOLDS["memory_high_gb"]:
            self.problems_found.append(f"High memory usage at {label}: {current_memory:.2f} GB")
    
    def check_time(self, label: str, start_time: float) -> None:
        """Check if an operation took too long."""
        duration = time.time() - start_time
        threshold = THRESHOLDS.get(f"{label}_slow_s", float('inf'))
        if duration > threshold:
            self.problems_found.append(f"Slow {label}: {duration:.2f}s")
    
    def log_problems(self) -> None:
        """Log all detected problems."""
        if self.problems_found:
            logger.warning("\n==== Detected Problems ====")
            for problem in self.problems_found:
                logger.warning(f"⚠️ {problem}")
            
            # Log memory summary
            final_memory = self.get_memory_usage()
            logger.warning("\n==== Memory Usage Summary ====")
            logger.warning(f"Initial memory: {self.initial_memory:.2f} GB")
            logger.warning(f"Final memory: {final_memory:.2f} GB (+{final_memory - self.initial_memory:.2f} GB)")
            
            # Log checkpoint timings
            logger.warning("\n==== Operation Timings ====")
            for name, data in self.checkpoints.items():
                logger.warning(f"{name}: {data['time']:.2f}s")

def wrap_original_main(args: argparse.Namespace) -> int:
    """Wrap the original main function with problem detection."""
    detector = ProblemDetector()
    
    try:
        # Check initial state
        detector.check_memory("initial")
        
        # Run the original main function
        result = original_main()
        
        # Log any problems found
        detector.log_problems()
        
        return result
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        detector.log_problems()
        return 1

def main():
    """Main entry point with problem detection."""
    # Get the original parser
    original_parser = argparse.ArgumentParser()
    original_parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to directory containing Qwen25 model weights"
    )
    original_parser.add_argument(
        "--prompt", 
        type=str, 
        default="Write a short story about a robot learning to paint:",
        help="Text prompt for generation"
    )
    original_parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=200,
        help="Maximum number of tokens to generate"
    )
    original_parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (lower = more deterministic)"
    )
    original_parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling probability threshold"
    )
    original_parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    original_parser.add_argument(
        "--mesh_shape", 
        type=str, 
        default=None,
        help="Device mesh shape for tensor parallelism (e.g., '1,8')"
    )
    original_parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model parameters"
    )
    original_parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    original_parser.add_argument(
        "--no_stream", 
        action="store_true",
        help="Disable streaming output"
    )
    original_parser.add_argument(
        "--shard",
        action="store_true",
        help="Enable parameter sharding across devices"
    )
    original_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed memory profiling"
    )
    original_parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save generated text to this file (if not specified, a timestamped file will be created)"
    )
    original_parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save generated text to a file"
    )
    
    # Add our problem detection specific argument
    original_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable stricter problem detection thresholds"
    )
    
    # Parse arguments
    args = original_parser.parse_args()
    
    # Update thresholds if strict mode is enabled
    if args.strict:
        for key in THRESHOLDS:
            THRESHOLDS[key] *= 0.8  # Make thresholds 20% stricter
    
    # Run the wrapped main function
    return wrap_original_main(args)

if __name__ == "__main__":
    exit(main()) 