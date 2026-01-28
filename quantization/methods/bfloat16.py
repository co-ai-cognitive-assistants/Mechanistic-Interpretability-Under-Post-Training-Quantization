# quantization/methods/bfloat16.py
# This script converts a model to bfloat16 precision.
# bfloat16 is a 16-bit floating point format that preserves dynamic range
# similar to float32, making it useful for deep learning.
# It is natively supported in modern hardware (e.g., NVIDIA Ampere, Intel Cooper Lake).
# Source: PyTorch documentation and https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import torch

# Monkey-patch for Ministral3 support if missing
if "ministral3" not in CONFIG_MAPPING:
    print("Monkey-patching 'ministral3' into transformers CONFIG_MAPPING...")
    CONFIG_MAPPING["ministral3"] = MistralConfig
    
    # Also patch inside the mistral3 module if possible
    try:
        import transformers.models.mistral3.configuration_mistral3 as m3_config
        if "ministral3" not in m3_config.CONFIG_MAPPING:
             print("Monkey-patching 'ministral3' into transformers.models.mistral3.configuration_mistral3.CONFIG_MAPPING...")
             m3_config.CONFIG_MAPPING["ministral3"] = MistralConfig
    except ImportError:
        pass

def load_tokenizer_safe(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as e:
        if "TokenizersBackend" in str(e):
            print(f"Warning: 'TokenizersBackend' error detected for {model_name}. Falling back to LlamaTokenizerFast.")
            try:
                return LlamaTokenizerFast.from_pretrained(model_name)
            except AttributeError:
                 print(f"Warning: LlamaTokenizerFast failed for {model_name}. Falling back to slow tokenizer.")
                 return AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        raise e

def quantize(model_name, output_dir):
    """
    Quantizes a model to bfloat16 and saves it.
    """
    print(f"Loading model for bfloat16: {model_name}")
    tokenizer = load_tokenizer_safe(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

    print("Converting to bfloat16...")
    model.bfloat16()

    print(f"Saving bfloat16 model to: {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
