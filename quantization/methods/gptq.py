# quantization/methods/gptq.py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AutoConfig, MistralConfig, LlamaTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from datasets import load_dataset
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
            return LlamaTokenizerFast.from_pretrained(model_name)
        raise e

def quantize(model_name, output_dir):
    """
    Quantizes a model using GPTQ with Hugging Face transformers and saves it.
    """
    print(f"Loading tokenizer for GPTQ quantization: {model_name}")
    tokenizer = load_tokenizer_safe(model_name)

    # --- Calibration Data ---
    print("Loading and preparing calibration dataset 'HuggingFaceH4/ultrachat_200k'...")
    dataset_id = "HuggingFaceH4/ultrachat_200k"
    dataset_split = "train_sft"
    num_calibration_samples = 512

    ds = load_dataset(dataset_id, split=f"{dataset_split}[:{num_calibration_samples}]")
    ds = ds.shuffle(seed=42)

    # The transformers GPTQ implementation requires a list of strings for the dataset
    def preprocess(example):
        if "messages" in example:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False, # Important for calibration
                )
            }
        return {"text": example["text"]}
    
    ds = ds.map(preprocess, remove_columns=[name for name in ds.column_names if name != 'text'])
    
    # Create a list of strings for the calibration dataset
    calibration_dataset = [d["text"] for d in ds]

    # --- Quantization ---
    print("Creating GPTQ configuration...")
    # For better performance, install auto-gptq with exllama support if available
    # and set use_exllama=True
    gptq_config = GPTQConfig(
        bits=4,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        use_exllama=False,
        model_seqlen=1024
    )

    print(f"Quantizing model: {model_name}")
    
    # Load config first to check for existing quantization config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        print(f"Warning: Model {model_name} already has a quantization_config. Removing it to proceed with GPTQ quantization.")
        del config.quantization_config

    # Ensure use_cache is present in config, as required by some quantizers (e.g. Optimum/GPTQ)
    if not hasattr(config, "use_cache"):
        print(f"Warning: Model config does not have 'use_cache' attribute. Setting it to False for quantization compatibility.")
        config.use_cache = False

    # Determine attention implementation
    # Gemma-3 requires "eager" to avoid SDPA tensor mismatches during quantization.
    # GPT-OSS (and potentially others) fail with "eager" due to RoPE broadcasting issues, so we use default (likely SDPA).
    # Qwen/DeepSeek models often have RoPE issues with SDPA during quantization, so we force eager.
    model_kwargs = {}
    if any(name in model_name.lower() for name in ["gemma", "qwen", "deepseek"]):
        model_kwargs["attn_implementation"] = "eager"
        print(f"Forcing attn_implementation='eager' for {model_name}")

    try:
        # Heuristic: Use llm-compressor for DeepSeek/Qwen to avoid AutoGPTQ instability/crashes
        if any(name in model_name.lower() for name in ["deepseek", "qwen"]):
            print(f"Skipping standard AutoGPTQ for {model_name} to avoid potential CUDA crashes. Jumping to llm-compressor...")
            raise ImportError("Forcing llm-compressor preference")

        # Load and quantize the model
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=gptq_config,
            torch_dtype="auto", # Recommended for GPTQ
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        # --- Save Model and Tokenizer ---
        print(f"Saving quantized model to: {output_dir}")
        quantized_model.save_pretrained(output_dir)
        
        print(f"Saving tokenizer to: {output_dir}")
        tokenizer.save_pretrained(output_dir)

        return quantized_model, tokenizer

    except (AttributeError, ValueError, ImportError, RuntimeError) as e:
        print(f"Standard transformers/AutoGPTQ quantization failed: {e}")
        print("Attempting fallback to llm-compressor (vLLM compatible)...")

        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization.gptq import GPTQModifier
            import torch
            
            # Clean up
            if 'quantized_model' in locals(): del quantized_model
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

            print(f"Loading standard model for llm-compressor GPTQ: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto", 
                device_map="auto",
                trust_remote_code=True
            )

            # Default group size
            q_group_size = 128
            
            # Adjust for Gemma-3 (dimension 4304 is divisible by 16, but not 128)
            if "gemma-3" in model_name.lower():
                 print(f"Detected Gemma-3 model. Adjusting group_size to 16 to fit dimension 4304.")
                 q_group_size = 16

            recipe = GPTQModifier(
                ignore=["lm_head"],
                config_groups={
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": q_group_size},
                        "input_activations": None,
                        "targets": ["Linear"],
                    }
                },
                block_size=128,
                dampening_frac=0.01,
                actorder=None, # Static/None often safer for compatibility than dynamic
            )

            print("Applying GPTQ via llm-compressor...")
            # Use a subset of calibration data for speed in fallback
            from datasets import Dataset
            calibration_ds_subset = Dataset.from_dict({"text": calibration_dataset[:128]})

            oneshot(
                model=model,
                recipe=recipe,
                dataset=calibration_ds_subset,
                tokenizer=tokenizer,
            )

            print(f"Saving llm-compressor GPTQ model to: {output_dir}")
            print("WARNING: To evaluate this model, use the 'vllm' backend in lm_eval (e.g., --model vllm).")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            return model, tokenizer

        except ImportError:
            print("Error: llm-compressor not found. Install with 'pip install llmcompressor'.")
            raise e
