# lm_eval/run_evaluation.py
import yaml
import os
import argparse
import subprocess
import json
import logging
import sys
import glob

def setup_logging():
    log_filename = "evaluation.log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_lm_eval(command, model_path):
    try:
        logging.info(f"Running command: {' '.join(command)}")
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation failed for {model_path}")
        logging.error(f"Exit code: {e.returncode}")
        logging.error(f"Stderr:\n{e.stderr}")
        # Check for specific errors
        if "ValueError: The model is quantized" in e.stderr and "passing a dict config" in e.stderr:
             return "RETRY_CONFIG_CONFLICT"
        return False

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness.")
    parser.add_argument("--config", type=str, default="configs/config.yml")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config not found: {args.config}")
        sys.exit(1)

    model_paths = config['model_paths']
    override = config.get('override', False)
    tasks = config.get('tasks', "arc_challenge,hellaswag,mmlu,winogrande")
    if isinstance(tasks, list): tasks = ",".join(tasks)
    gpus = config.get('gpus', [0])

    for i, model_path in enumerate(model_paths):
        logging.info(f"--- Evaluating model: {model_path} ---")

        if not os.path.exists(model_path):
             logging.warning(f"Path {model_path} not found. Skipping.")
             continue

        # Name logic
        model_name_parts = model_path.split('/')
        try:
            if 'models' in model_name_parts:
                idx = model_name_parts.index('models')
                repo = model_name_parts[idx+1]
                method = model_name_parts[idx+2] if len(model_name_parts) > idx+2 else ""
                result_name = f"{repo}_{method}"
            else:
                result_name = model_path.replace('/', '_')
                method = ""
        except:
            result_name = model_path.replace('/', '_')
            method = ""

        output_base = f"../results/{result_name}"
        if not override and glob.glob(f"{output_base}_temp_detail_*.json"):
            logging.info(f"Skipping {model_path}, exists.")
            continue
        
        temp_output = f"{output_base}_temp_detail.json"
        gpu = gpus[i % len(gpus)]

        # Check for existing config
        def has_quantization_config(path):
            try:
                config_path = os.path.join(path, "config.json")
                if not os.path.exists(config_path): return False
                with open(config_path, 'r') as f:
                    return 'quantization_config' in json.load(f)
            except:
                return False

        has_quant_config = has_quantization_config(model_path)

        # Build args
        backend = "hf"
        gguf_file = None
        
        if method == "gguf":
            # GGUF via HF
            files = glob.glob(os.path.join(model_path, "*.gguf"))
            if files:
                gguf_full_path = files[0]
                if os.path.getsize(gguf_full_path) < 1024:
                    logging.warning(f"GGUF file {gguf_full_path} seems to be a placeholder (size < 1KB). Skipping.")
                    continue
                gguf_file = os.path.basename(gguf_full_path)
            else:
                logging.warning("No .gguf file found.")
                continue
        
        # Retry loop
        # Attempt 1: Standard (Trust=True)
        # Attempt 2: Trust=False (Fallback)
        attempts_config = [
            {"trust": True},
            {"trust": False}
        ]
        
        success = False
        for attempt in attempts_config:
            args_list = [f"pretrained={model_path}"]
            
            if attempt["trust"]:
                args_list.append("trust_remote_code=True")
            
            if method == "gguf" and gguf_file:
                args_list.append(f"gguf_file={gguf_file}")
            
            # Only add load_in bits if NOT in config
            if method in ['int4', 'int8'] and not has_quant_config:
                 args_list.append(f"load_in_{method[3:]}bit=True")
            
            # AWQ and FP8 often require explicit dtype=float16 to avoid bf16 mismatch in kernels
            if method in ['awq', 'fp8']:
                args_list.append("dtype=float16")

            # Fix for tokenizer regex issues (e.g. Gemma 3 / Mistral)
            args_list.append("fix_mistral_regex=True")

            # For GPTQ/AWQ, do NOT add disable_exllama by default to avoid conflicts
            
            cmd = [
                sys.executable, "lm_eval_wrapper.py", "--model", backend,
                "--model_args", ",".join(args_list),
                "--tasks", tasks,
                "--output_path", temp_output,
                "--log_samples",
                "--device", f"cuda:{gpu}"
            ]
            
            os.makedirs(os.path.dirname(temp_output), exist_ok=True)
            res = run_lm_eval(cmd, model_path)
            
            if res is True:
                success = True
                break
            if res == "RETRY_CONFIG_CONFLICT":
                logging.info("Retrying due to config conflict...")
                continue
            
        if success:
            logging.info(f"Success: {model_path}")
        else:
            logging.error(f"Failed: {model_path}")

if __name__ == "__main__":
    main()
