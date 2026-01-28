# quantization/run_quantization.py
import yaml
import os
import argparse
import importlib
import logging
import sys
import shutil
from safetensors import SafetensorError
import torch
import gc
import subprocess

def setup_logging():
    """Set up logging to file and console."""
    log_filename = "quantization.log"
    
    # Use append mode, and let's not clear the log on every single run
    # especially when using subprocesses. The main process can clear it once.
    file_mode = 'w' if '--run-single' not in sys.argv else 'a'
    
    # Remove existing handlers to avoid duplication in subprocesses
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode=file_mode),
            logging.StreamHandler(sys.stdout)
        ]
    )

def quantize_model(repo_id: str, method_name: str, override: bool):
    """
    Quantizes a single model with a single method.
    This function is designed to be called in a subprocess for isolation.
    """
    logging.info(f"--- Starting {method_name} quantization for {repo_id} ---")

    try:
        methods_dir = os.path.join(os.path.dirname(__file__), 'methods')
        available_methods = [f.split('.')[0] for f in os.listdir(methods_dir) if f.endswith('.py') and f != '__init__.py']
    except FileNotFoundError:
        logging.error(f"The 'methods' directory was not found at {methods_dir}.")
        sys.exit(1)

    if method_name not in available_methods:
        logging.warning(f"Method '{method_name}' not found in 'methods' directory. Skipping.")
        return

    base_output_dir = f"../models/{repo_id.replace('/', '_')}"
    output_dir = os.path.join(base_output_dir, method_name)

    if not override and os.path.isdir(output_dir) and os.listdir(output_dir):
        logging.info(f"Skipping {method_name} for {repo_id} as it already exists.")
        return

    method_module = None
    model = None
    tokenizer = None
    try:
        method_module = importlib.import_module(f"methods.{method_name}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            result = method_module.quantize(repo_id, output_dir)
            if isinstance(result, tuple) and len(result) == 2:
                model, tokenizer = result
            else:
                model = None
                tokenizer = None
        except SafetensorError as e:
            logging.warning(f"Safetensor error for {repo_id}: {e}. This may indicate a corrupt model cache. Attempting to clear cache and retry...")
            mangled_repo_id = "models--" + repo_id.replace("/", "--")
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "hub", mangled_repo_id)

            if os.path.exists(cache_dir):
                logging.info(f"Deleting cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                logging.info(f"Cache cleared for {repo_id}. Retrying quantization...")
                result = method_module.quantize(repo_id, output_dir)
                if isinstance(result, tuple) and len(result) == 2:
                    model, tokenizer = result
                else:
                    model = None
                    tokenizer = None
            else:
                logging.warning(f"Could not find cache directory to clear: {cache_dir}. Skipping retry.")
                raise e

        logging.info(f"--- Finished {method_name} quantization for {repo_id} ---")
    except Exception as e:
        logging.error(f"An error occurred during {method_name} quantization for {repo_id}: {e}", exc_info=True)
        sys.exit(1) # Exit subprocess with an error code
    finally:
        logging.info("Cleaning up GPU memory...")
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer

        module_name = f"methods.{method_name}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        if 'method_module' in locals() and method_module is not None:
            del method_module

        gc.collect()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run quantization on a model.")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the quantization config file.")
    parser.add_argument("--run-single", action="store_true", help="Run a single quantization task specified by other arguments.")
    parser.add_argument("--repo-id", type=str, help="Repository ID for single run.")
    parser.add_argument("--method", type=str, help="Quantization method for single run.")
    parser.add_argument("--override-single", action="store_true", help="Override existing model in single run.")
    parser.add_argument("--device-id", type=str, default=None, help="CUDA device ID to use (e.g., '0' or '0,1'). Overrides config.")
    args = parser.parse_args()

    # Setup logging. If it's the main process, it will clear the log.
    # If it's a subprocess, it will append.
    setup_logging()

    if args.run_single:
        if not args.repo_id or not args.method:
            logging.error("--repo-id and --method are required with --run-single")
            sys.exit(1)
        quantize_model(args.repo_id, args.method, args.override_single)
        return

    # Main process logic starts here
    log_filename = "quantization.log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    setup_logging() # Re-init to ensure correct mode after potential deletion

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {args.config}")
        sys.exit(1)

    override = config.get('override', False)
    repo_ids = config.get('repo_ids', [])
    
    # Determine Device ID: CLI arg > Config file > Default "0"
    device_id = args.device_id if args.device_id is not None else config.get('device_id', "0")
    
    if isinstance(device_id, list):
        device_id = ",".join(map(str, device_id))
    else:
        device_id = str(device_id)

    if 'repo_id' in config and config['repo_id'] not in repo_ids:
        repo_ids.insert(0, config['repo_id'])

    if not repo_ids:
        logging.error("Config file must contain either 'repo_id' or 'repo_ids'.")
        sys.exit(1)
    
    try:
        methods_dir = os.path.join(os.path.dirname(__file__), 'methods')
        available_methods = [f.split('.')[0] for f in os.listdir(methods_dir) if f.endswith('.py') and f != '__init__.py']
    except FileNotFoundError:
        logging.error(f"The 'methods' directory was not found at {methods_dir}. Please ensure it exists and contains quantization scripts.")
        sys.exit(1)
        
    methods_to_run = config.get('methods', available_methods)

    for repo_id in repo_ids:
        logging.info(f"===== Starting quantization jobs for model: {repo_id} ====")
        for method_name in methods_to_run:
            command = [
                sys.executable,
                __file__,
                "--run-single",
                "--repo-id", repo_id,
                "--method", method_name,
            ]
            if override:
                command.append("--override-single")
            
            logging.info(f"Spawning subprocess for {repo_id} with method {method_name} on GPU {device_id}")
            
            # Force usage of specific GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device_id
            
            # Using Popen to stream output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

            # Log output line-by-line
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    logging.info(line.strip())
                process.stdout.close()
            
            return_code = process.wait()

            if return_code != 0:
                logging.error(f"Subprocess for {repo_id}/{method_name} failed with exit code {return_code}.")
            else:
                logging.info(f"Subprocess for {repo_id}/{method_name} completed successfully.")
        
        logging.info(f"===== Finished quantization jobs for model: {repo_id} ====")


if __name__ == "__main__":
    main()