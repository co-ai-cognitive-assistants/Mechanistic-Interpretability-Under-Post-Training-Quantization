# quantization/methods/gguf.py
# This script handles GGUF (GPT-Generated Unified Format) conversion and quantization.
# GGUF is a file format for storing models for inference with GGML and llama.cpp.
# It is a successor to GGML, GGJT, and GGMF.
# Source: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import os
import subprocess
import sys
from huggingface_hub import snapshot_download

def quantize(model_name, output_dir):
    """
    Converts a model to GGUF format and quantizes it using llama.cpp.
    The quantization method can be specified via the GGUF_QUANT_METHOD environment variable.
    Defaults to Q4_0.
    """
    quant_method = os.environ.get("GGUF_QUANT_METHOD", "q8_0")
    print(f"Preparing for GGUF conversion for model: {model_name} with method: {quant_method}")

    # GGUF conversion requires the model to be downloaded locally first.
    # snapshot_download will download the repo to the HF cache and return the path.
    try:
        model_path = snapshot_download(repo_id=model_name)
    except Exception as e:
        print(f"Failed to download model '{model_name}'.")
        print(f"Error: {e}")
        raise

    # Ensure output directory exists and paths are absolute
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f"model-{quant_method.lower()}.gguf"
    output_file = os.path.join(output_dir, output_filename)

    llama_cpp_dir = os.path.abspath("../llama.cpp")
    # In llama.cpp, the conversion script for Hugging Face models is convert_hf_to_gguf.py
    convert_script = "convert_hf_to_gguf.py"

    if not os.path.isdir(llama_cpp_dir):
        print(f"Error: 'llama.cpp' directory not found at {llama_cpp_dir}")
        print("Please clone the llama.cpp repository into the project root directory.")
        raise FileNotFoundError(f"'llama.cpp' directory not found at {llama_cpp_dir}")

    requirements_path = os.path.join(llama_cpp_dir, "requirements.txt")
    if not os.path.exists(requirements_path):
        # Fallback for different llama.cpp versions
        requirements_path = os.path.join(llama_cpp_dir, "requirements-hf-to-gguf.txt")

    print(f"Converting and quantizing model to {quant_method} GGUF format...")
    print(f"Model source: {model_path}")
    print(f"Output file: {output_file}")

    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile",
        output_file,
        "--outtype",
        quant_method,
    ]

    try:
        # The conversion script should be run from within the llama.cpp directory.
        # It may have relative path dependencies.
        process = subprocess.run(
            cmd,
            cwd=llama_cpp_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Successfully created GGUF file at: {output_file}")
        print("\n--- llama.cpp conversion script output ---")
        print(process.stdout)
        if process.stderr:
            print("\n--- llama.cpp conversion script warnings ---")
            print(process.stderr)
        print("--- End of script output ---")

    except FileNotFoundError:
        print(f"Error: Conversion script '{convert_script}' not found in '{llama_cpp_dir}'.")
        print("Please ensure you are using a version of llama.cpp that includes this script.")
        raise
    except subprocess.CalledProcessError as e:
        print("\n" + "="*50)
        print("GGUF conversion failed.")
        print(f"Return code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("="*50)
        print(f"\nCommand run in '{llama_cpp_dir}': {' '.join(cmd)}")
        print(f"\nPlease ensure you have the necessary requirements installed from '{requirements_path}' (e.g., pip install -r {requirements_path})")
        raise e