# quantization/methods/awq.py
import torch.nn as nn
import transformers.activations

# Patch PytorchGELUTanh if missing (removed in newer transformers)
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    print("Patching transformers.activations.PytorchGELUTanh...")
    transformers.activations.PytorchGELUTanh = nn.GELU(approximate="tanh")

from transformers import AutoTokenizer, MistralConfig, LlamaTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM # Required for fallback loading

# Monkey-patch for Ministral3 support if missing
if "ministral3" not in CONFIG_MAPPING:
    print("Monkey-patching 'ministral3' into transformers CONFIG_MAPPING...")
    CONFIG_MAPPING["ministral3"] = MistralConfig

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
    Quantizes a model using AWQ with autoawq and saves it.
    """
    # --- Quantization Config ---
    # Default
    q_group_size = 128
    
    # Adjust for Gemma-3 (dimension 4304 is divisible by 16, but not 32, 64, 128)
    if "gemma-3" in model_name.lower():
         print(f"Detected Gemma-3 model. Adjusting group_size to 16 to fit dimension 4304.")
         q_group_size = 16

    # Adjust for Llama-3.1 (avoid packed zero-points decompression error in current autoawq versions)
    use_zero_point = True
    if "llama-3.1" in model_name.lower():
        print(f"Detected Llama-3.1 model. Disabling zero_point (Symmetric Quantization) to avoid decompression errors.")
        use_zero_point = False

    quant_config = {
        "zero_point": use_zero_point, 
        "q_group_size": q_group_size, 
        "w_bit": 4, 
        "version": "GEMM"
    }

    try:
        # Force fallback for Llama-3.1 to avoid AutoAWQ AssertionError regarding missing zeros
        if "llama-3.1" in model_name.lower():
            print("Forcing llm-compressor for Llama-3.1 to avoid AutoAWQ issues...")
            raise ImportError("Force llm-compressor for Llama-3.1")

        print(f"Loading model for AWQ quantization: {model_name}")
        model = AutoAWQForCausalLM.from_pretrained(
            model_name, 
            **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = load_tokenizer_safe(model_name)

        print("Quantizing model with autoawq...")
        model.quantize(tokenizer, quant_config=quant_config)

        # --- Save Model and Tokenizer ---
        print(f"Saving quantized model to: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

    except (TypeError, AttributeError, ImportError, ValueError, KeyError, AssertionError) as e:
        print(f"Standard AutoAWQ quantization failed or skipped: {e}")
        print("Attempting fallback to llm-compressor (vLLM compatible)...")
        
        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.awq import AWQModifier
            from datasets import load_dataset, Dataset
            import torch

            # Clean up
            if 'model' in locals(): del model
            torch.cuda.empty_cache()

            print(f"Loading standard model for llm-compressor AWQ: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            # Re-load tokenizer just in case
            tokenizer = load_tokenizer_safe(model_name)

            # --- Calibration Data (same as GPTQ) ---
            print("Loading and preparing calibration dataset 'HuggingFaceH4/ultrachat_200k'...")
            dataset_id = "HuggingFaceH4/ultrachat_200k"
            dataset_split = "train_sft"
            num_calibration_samples = 512

            ds = load_dataset(dataset_id, split=f"{dataset_split}[:{num_calibration_samples}]")
            ds = ds.shuffle(seed=42)

            def preprocess(example):
                if "messages" in example:
                    return {
                        "text": tokenizer.apply_chat_template(
                            example["messages"],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    }
                return {"text": example["text"]}

            ds = ds.map(preprocess, remove_columns=[name for name in ds.column_names if name != 'text'])
            calibration_data = [d["text"] for d in ds]

            # Configure AWQModifier with the correct group size
            recipe = AWQModifier(
                ignore=["lm_head"],
                config_groups={
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": not use_zero_point, # Use the flag we detected earlier
                            "strategy": "group",
                            "group_size": q_group_size
                        },
                        "targets": ["Linear"]
                    }
                }
            )

            print("Applying AWQ quantization via llm-compressor...")
            # Convert list of strings to a HF Dataset
            calibration_dataset = Dataset.from_dict({"text": calibration_data})
            
            oneshot(
                model=model,
                recipe=recipe,
                dataset=calibration_dataset, 
                tokenizer=tokenizer,
            )

            print(f"Saving llm-compressor AWQ model to: {output_dir}")
            print("WARNING: To evaluate this model, use the 'vllm' backend in lm_eval (e.g., --model vllm).")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        except ImportError:
            print("Error: llm-compressor not found. Install with 'pip install llmcompressor'.")
            raise e