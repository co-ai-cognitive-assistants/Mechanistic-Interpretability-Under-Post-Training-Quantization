# quantization/methods/int8.py
# This script implements 8-bit quantization using bitsandbytes.
# This method is based on the LLM.int8() paper.
# Source: `bitsandbytes` library by Tim Dettmers.
# Paper: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
# Link: https://arxiv.org/abs/2208.07339
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, MistralConfig, LlamaTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

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
    Quantizes a model to int8 and saves it.
    """
    print(f"Loading and quantizing model to int8: {model_name}")
    tokenizer = load_tokenizer_safe(model_name)
    
    # Load config first to check for existing quantization config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        print(f"Warning: Model {model_name} already has a quantization_config. Removing it to proceed with int8 quantization.")
        del config.quantization_config

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print(f"Saving int8 model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
