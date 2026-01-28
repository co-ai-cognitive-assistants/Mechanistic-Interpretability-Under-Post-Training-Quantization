import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralConfig, LlamaTokenizerFast, AutoConfig
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
    Quantizes a model using HQQ and saves it.
    Prioritizes transformers native HQQ, falls back to hqq library manual quantization.
    """
    print(f"Loading model for HQQ quantization: {model_name}")
    tokenizer = load_tokenizer_safe(model_name)
    
    # Load config first to check for existing quantization config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        print(f"Warning: Model {model_name} already has a quantization_config. Removing it to proceed with HQQ quantization.")
        del config.quantization_config
    
    # Attempt 1: Transformers Native HQQ (Preferred)
    try:
        from transformers import HqqConfig
        print("Attempting quantization using transformers.HqqConfig...")
        
        quant_config = HqqConfig(nbits=4, group_size=64, axis=1)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        
        print(f"Saving HQQ model (transformers) to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return

    except (ImportError, TypeError, ValueError, AttributeError) as e:
        print(f"Transformers HQQ integration failed or not found: {e}")
        print("Falling back to hqq library manual quantization...")

    # Attempt 2: HQQ Library Manual Quantization
    # We load the model first WITHOUT config to avoid the __init__ error, then quantize.
    try:
        from hqq.engine.hf import HQQModelForCausalLM
        from hqq.core.quantize import BaseQuantizeConfig
        
        # Load model normally
        model = HQQModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Define config
        quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        
        # Quantize in-place
        print("Quantizing model layers...")
        model.quantize_model(quant_config=quant_config)
        
        print(f"Saving HQQ model (hqq-lib) to: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

    except Exception as e:
        print(f"Critical error during HQQ quantization: {e}")
        raise e

