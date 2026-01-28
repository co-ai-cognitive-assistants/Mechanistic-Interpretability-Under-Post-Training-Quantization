# quantization/methods/fp8.py
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config, MistralConfig, LlamaTokenizerFast, AutoConfig
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
            return LlamaTokenizerFast.from_pretrained(model_name)
        raise e

def quantize(model_name, output_dir):
    """
    Quantizes a model to FP8 using FineGrainedFP8Config and saves it.
    Requires a GPU with compute capability >= 9.0 (e.g., H100) and 'accelerate'.
    """
    print(f"Loading and quantizing model to FP8: {model_name}")
    
    try:
        quantization_config = FineGrainedFP8Config()
    except ImportError:
        print("Error: FineGrainedFP8Config not found in transformers. Please ensure you have a recent version installed.")
        # Fallback check for FbgemmFp8Config
        try:
            from transformers import FbgemmFp8Config
            print("Falling back to FbgemmFp8Config...")
            quantization_config = FbgemmFp8Config()
        except ImportError:
            raise ImportError("Neither FineGrainedFP8Config nor FbgemmFp8Config found.")

    try:
        # Load config first to check for existing quantization config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            print(f"Warning: Model {model_name} already has a quantization_config. Removing it to proceed with FP8 quantization.")
            del config.quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = load_tokenizer_safe(model_name)

        print(f"Saving FP8 model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    except (ValueError, ImportError, NotImplementedError, RuntimeError, KeyError) as e:
        print(f"Standard transformers FP8 quantization failed: {e}")
        print("Attempting fallback to llm-compressor (vLLM compatible)...")
        
        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            import torch
            
            # Clean up memory from previous attempt
            if 'model' in locals(): del model
            torch.cuda.empty_cache()

            print(f"Loading model for llm-compressor FP8: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto", 
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = load_tokenizer_safe(model_name)

            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=["lm_head"]
            )

            print("Applying FP8_DYNAMIC quantization...")
            oneshot(
                model=model,
                recipe=recipe,
                tokenizer=tokenizer,
            )

            print(f"Saving llm-compressor FP8 model to: {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
        except ImportError:
            print("Error: llm-compressor not found. Please install it via 'pip install llmcompressor' to use the fallback.")
            raise e
