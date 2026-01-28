import os
import yaml
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.pretrained.weight_conversions as wc
from sae_lens import LanguageModelSAETrainingRunner, LanguageModelSAERunnerConfig

# --- Monkey Patch for AutoAWQ / Transformers Compatibility ---
# AutoAWQ imports PytorchGELUTanh from transformers.activations, 
# which was renamed to GELUTanh in newer transformers versions.
import transformers.activations
if not hasattr(transformers.activations, "PytorchGELUTanh") and hasattr(transformers.activations, "GELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh
    print("Monkey-patched transformers.activations.PytorchGELUTanh to fix AutoAWQ import.")
# -----------------------------------------------------------

# Import Matryoshka Config
try:
    from sae_lens import MatryoshkaBatchTopKTrainingSAEConfig
except ImportError:
    from sae_lens.saes.matryoshka_batchtopk_sae import MatryoshkaBatchTopKTrainingSAEConfig

# Import LoggingConfig
try:
    from sae_lens import LoggingConfig
except ImportError:
    from sae_lens.config import LoggingConfig

def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_matryoshka_widths(total_dict_size, group_fractions):
    widths = []
    cumulative_fraction = 0.0
    for fraction in group_fractions:
        cumulative_fraction += fraction
        widths.append(int(round(cumulative_fraction * total_dict_size)))
    return widths

def convert_gemma3_weights(model, cfg: HookedTransformerConfig):
    """
    Converts Gemma 3 weights to HookedTransformer format.
    Gemma 3 nests the text model under `model.language_model`.
    """
    state_dict = {}
    
    # Robustly find the inner transformer model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        gemma = model.model.language_model
    elif hasattr(model, "language_model"):
        gemma = model.language_model
    elif hasattr(model, "model"):
        # Standard HF CausalLM structure (GemmaForCausalLM, LlamaForCausalLM, etc.)
        gemma = model.model
    elif hasattr(model, "transformer"):
        # Some other architectures or older wrappers
        gemma = model.transformer
    else:
        # Fallback/Hope it's the text model itself
        gemma = model
    
    print(f"DEBUG: convert_gemma3_weights - gemma type: {type(gemma)}")
    # Verify we found something with embed_tokens
    if not hasattr(gemma, "embed_tokens"):
        raise AttributeError(f"Could not find 'embed_tokens' in extracted model of type {type(gemma)}. attributes: {list(gemma.__dict__.keys())[:20]}")

    # Embeddings
    state_dict["embed.W_E"] = gemma.embed_tokens.weight
    
    # Layers
    for l in range(cfg.n_layers):
        layer = gemma.layers[l]
        
        # Attention
        # HF: [n_heads * d_head, d_model]
        # TL: [n_heads, d_model, d_head]
        
        # Q
        w_q = layer.self_attn.q_proj.weight
        state_dict[f"blocks.{l}.attn.W_Q"] = w_q.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        # K
        w_k = layer.self_attn.k_proj.weight
        state_dict[f"blocks.{l}.attn.W_K"] = w_k.reshape(cfg.n_key_value_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        # V
        w_v = layer.self_attn.v_proj.weight
        state_dict[f"blocks.{l}.attn.W_V"] = w_v.reshape(cfg.n_key_value_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        # O
        # HF: [d_model, n_heads * d_head]
        # TL: [n_heads, d_head, d_model]
        w_o = layer.self_attn.o_proj.weight
        state_dict[f"blocks.{l}.attn.W_O"] = w_o.reshape(cfg.d_model, cfg.n_heads, cfg.d_head).permute(1, 2, 0)
        
        # MLPs (Gated)
        # Gemma 3 uses gate_proj, up_proj, down_proj
        # HF Linear: [out, in]
        # TL MLP: [in, out] (usually)
        
        # W_gate: HF [d_mlp, d_model] -> TL [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        
        # W_in (up): HF [d_mlp, d_model] -> TL [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        
        # W_out (down): HF [d_model, d_mlp] -> TL [d_mlp, d_model]
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        
        # Norms
        state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight
        state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight
        
    # Final Norm
    state_dict["ln_final.w"] = gemma.norm.weight
    
    # Unembed
    # HF: [vocab, d_model]
    # TL: [d_model, vocab]
    if hasattr(model, "lm_head"):
        state_dict["unembed.W_U"] = model.lm_head.weight.T
    else:
        # Fallback to tied embeddings
        state_dict["unembed.W_U"] = gemma.embed_tokens.weight.T
        
    return state_dict

def get_conversion_function(model_type):
    """Dynamically retrieves the correct weight conversion function."""
    model_type = model_type.lower()
    
    # Map HF model types to TransformerLens converter names
    # This covers the most common cases requested (Llama, Qwen, Gemma)
    mapping = {
        "gemma": "convert_gemma_weights",
        "gemma2": "convert_gemma_weights",
        "gemma3": convert_gemma3_weights, # Use custom function
        "gemma3_text": convert_gemma3_weights, # Map gemma3_text to custom function
        "llama": "convert_llama_weights",
        "mistral": "convert_mistral_weights",
        "qwen2": "convert_qwen2_weights",
        "qwen": "convert_qwen2_weights", # R1/Qwen2 often just identify as qwen2 but handle aliases
    }
    
    val = mapping.get(model_type)
    if val:
        if callable(val):
            print(f"Selected conversion function: {val.__name__}")
            return val
        elif hasattr(wc, val):
            print(f"Selected conversion function: {val}")
            return getattr(wc, val)
    
    # Fallback: Try to guess function name "convert_{type}_weights"
    guessed_name = f"convert_{model_type}_weights"
    if hasattr(wc, guessed_name):
        print(f"Selected conversion function (guessed): {guessed_name}")
        return getattr(wc, guessed_name)
        
    raise ValueError(f"No conversion function found for model type '{model_type}'. Available: {dir(wc)}")

def universal_dequantize_model(module):
    from torch.nn import Linear
    for name, child in module.named_children():
        # Recursive call first
        universal_dequantize_model(child)
        
        # --- Method A: Manual Extraction for FP8 (HuggingFace/Accelerate style) ---
        # FP8Linear often exposes 'weight' (int8/float8) and 'weight_scale_inv'
        if hasattr(child, "weight") and hasattr(child, "weight_scale_inv"):
             print(f"Manual Dequantization (FP8/ScaleInv) for {name}...")
             try:
                 device = child.weight.device
                 target_dtype = torch.bfloat16
                 
                 # 1. Cast to Float32
                 w_float = child.weight.float()
                 s_float = child.weight_scale_inv.float()
                 
                 # 2. Handle Block Expansion (if scale is smaller than weight)
                 # FP8 often uses block sizes (e.g., 128). 
                 # If weight is [1024, 4096] and scale is [1024, 32], we need to expand scale dim 1 by 128.
                 if w_float.shape != s_float.shape:
                     # Iterate over dimensions to check for blocking
                     # We assume scale has same number of dims as weight (usually 2)
                     if w_float.dim() == s_float.dim():
                         for dim in range(w_float.dim()):
                             w_dim = w_float.shape[dim]
                             s_dim = s_float.shape[dim]
                             
                             if w_dim != s_dim:
                                 if w_dim % s_dim == 0:
                                     factor = w_dim // s_dim
                                     # print(f"  Expanding scale dim {dim} by {factor}x ({s_dim} -> {w_dim})")
                                     s_float = s_float.repeat_interleave(factor, dim=dim)
                                 else:
                                     print(f"  Warning: Scale dim {dim} ({s_dim}) not divisible into weight dim {dim} ({w_dim})")
                 
                 # 3. Multiply
                 w_extracted = w_float * s_float
                 
                 # FP8Linear stores weights as [out_features, in_features] (standard), 
                 # BUT sometimes kernels transpose them. We check dimensions.
                 # Standard Linear is (out, in).
                 if w_extracted.shape == (child.out_features, child.in_features):
                     # Already correct shape
                     weight_data = w_extracted.to(target_dtype)
                 elif w_extracted.shape == (child.in_features, child.out_features):
                     # Transposed
                     weight_data = w_extracted.T.to(target_dtype)
                 else:
                     print(f"  Warning: Shape mismatch {w_extracted.shape} for {name} (Expected {child.out_features}, {child.in_features})")
                     # Fallback to generic method if shape is weird
                     raise ValueError("Shape mismatch in manual extraction")
                 
                 # Create Standard Linear
                 new_layer = Linear(child.in_features, child.out_features, 
                                  bias=child.bias is not None, 
                                  device=device, dtype=target_dtype)
                 new_layer.weight.data = weight_data
                 
                 if child.bias is not None:
                     new_layer.bias.data = child.bias.data.to(target_dtype)
                 
                 # Replace
                 setattr(module, name, new_layer)
                 continue # Success, skip generic method
                 
             except Exception as e:
                 print(f"  Manual extraction failed: {e}. Falling back to Identity method...")

        # --- Method B: Generic Identity Matrix Method (For Int4, AWQ, GPTQ) ---
        # Check if it's a linear-like layer that needs dequantization
        # Handle inconsistent naming (in_features vs infeatures for GPTQ/AWQ)
        n_in = getattr(child, "in_features", getattr(child, "infeatures", None))
        n_out = getattr(child, "out_features", getattr(child, "outfeatures", None))
        
        # Criteria: Has dimensions, but is NOT exactly a standard Linear layer
        # Using type(child) is not Linear allows catching subclasses (BNB, CompressedTensors, FP8Linear)
        if n_in is not None and n_out is not None and \
           type(child) is not Linear and \
           not isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList)):
            
            print(f"Dequantizing {name} ({type(child).__name__})...") 
            
            # Robust Retry Loop for Dtypes
            # Some kernels (AWQ/GPTQ) require exact input dtypes (FP16), others work with FP32.
            success = False
            errors = []
            
            for extract_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                try:
                    device = getattr(child, "device", "cuda")
                    if hasattr(child, "weight") and hasattr(child.weight, "device"):
                        device = child.weight.device
                    elif hasattr(child, "qweight") and hasattr(child.qweight, "device"):
                        device = child.qweight.device
                        
                    target_dtype = torch.bfloat16
                    
                    # Create Identity Matrix
                    eye = torch.eye(n_in, device=device, dtype=extract_dtype)
                    
                    with torch.no_grad():
                        # Forward pass: y = xA^T + b
                        output = child(eye)
                    
                    # Handle Bias
                    bias_data = None
                    if hasattr(child, "bias") and child.bias is not None:
                        # Ensure bias is in same dtype for subtraction
                        bias = child.bias.to(extract_dtype)
                        # W^T = output - b
                        weight_T = output - bias
                        bias_data = child.bias.data.to(target_dtype)
                    else:
                        weight_T = output
                    
                    # Convert back to target dtype (BF16)
                    weight = weight_T.T.to(target_dtype)
                    
                    # Create standard Linear layer
                    new_layer = Linear(n_in, n_out, 
                                     bias=bias_data is not None, 
                                     device=device, dtype=target_dtype)
                    new_layer.weight.data = weight
                    if bias_data is not None:
                        new_layer.bias.data = bias_data
                        
                    # Replace in parent
                    setattr(module, name, new_layer)
                    success = True
                    break # Success!
                    
                except Exception as e:
                    errors.append(f"{extract_dtype}: {str(e)}")
                    # Continue to next dtype
            
            if not success:
                print(f"Failed to dequantize {name} with all dtypes. Errors: {errors}")

def main():
    config = load_config()
    
    models = config.get("models", [])
    methods = config.get("methods", [])
    
    sae_params = config["sae"]
    dataset_params = config["dataset"]
    wandb_params = config["wandb"]
    
    group_fractions = sae_params["group_fractions"]
    
    print(f"--- Starting Pipeline ---")
    
    for model_base_path in models:
        for method in methods:
            model_path = os.path.join(model_base_path, method)
            if not os.path.exists(model_path):
                print(f"Model path not found: {model_path}. Skipping.")
                continue
                
            print(f"--- Processing Model: {model_path} ---")
            
            try:
                # 1. Load HF Model
                print("Loading HF Model...")
                load_kwargs = {
                    "device_map": "cuda", # Force single GPU to avoid split-device errors
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16 if method == "bfloat16" else torch.float16,
                }
                
                # Check AutoConfig first to see if quantization config exists
                try:
                    auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    has_quant_config = hasattr(auto_config, "quantization_config") and auto_config.quantization_config is not None
                except Exception as e:
                    print(f"Warning: Could not load AutoConfig: {e}")
                    has_quant_config = False

                # GPTQ/AWQ specific fixes
                if method == "gptq":
                     # Only inject GPTQConfig if the model config DOESN'T already have quantization info
                     if not has_quant_config:
                         print("Injecting GPTQConfig (Legacy/Missing Config Mode)...")
                         from transformers import GPTQConfig
                         load_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)
                     else:
                         print("Detected existing quantization config. Relying on AutoModel to handle it.")
                
                hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                
                # Capture dtype str before deletion
                model_dtype_str = str(hf_model.dtype).split('.')[-1]
                
                # 2. Extract Config
                hf_config = hf_model.config
                model_type = getattr(hf_config, "model_type", "unknown")
                print(f"Detected Model Type: {model_type}")

                # Handle Multimodal Configs (Gemma 3)
                target_config = hf_config
                if hasattr(hf_config, "text_config"):
                    print("Detected Multimodal Config. Using 'text_config' for dimensions.")
                    target_config = hf_config.text_config
                
                # Helper for strict config extraction
                def get_config_attr(cfg, attr):
                    if hasattr(cfg, attr):
                        return getattr(cfg, attr)
                    raise ValueError(f"Missing required config attribute '{attr}' in {cfg}")

                n_layers = get_config_attr(target_config, "num_hidden_layers")
                d_model = get_config_attr(target_config, "hidden_size")
                n_heads = get_config_attr(target_config, "num_attention_heads")
                n_kv_heads = get_config_attr(target_config, "num_key_value_heads")
                d_mlp = get_config_attr(target_config, "intermediate_size")
                vocab_size = get_config_attr(target_config, "vocab_size")
                
                # Intelligent head_dim calculation
                if hasattr(target_config, "head_dim"):
                    head_dim = target_config.head_dim
                else:
                    head_dim = d_model // n_heads
                    print(f"Calculated head_dim: {head_dim} (d_model {d_model} // n_heads {n_heads})")
                
                # --- Dynamic Configuration Logic ---
                # 1. Calculate Layer Index (Relative vs Absolute)
                if "layer_relative" in config:
                    layer_index = int(n_layers * config["layer_relative"])
                    print(f"Calculated relative layer index: {layer_index} (Height: {config['layer_relative']})")
                else:
                    layer_index = config.get("layer_index", 20)
                    print(f"Using static layer index: {layer_index}")
                    
                # 2. Calculate SAE Dictionary Size (Expansion vs Absolute)
                if "expansion_factor" in sae_params:
                    total_dict_size = d_model * sae_params["expansion_factor"]
                    print(f"Calculated dynamic dict size: {total_dict_size} (Expansion: {sae_params['expansion_factor']}x)")
                else:
                    total_dict_size = sae_params["total_dict_size"]
                    print(f"Using static dict size: {total_dict_size}")
                    
                # 3. Recalculate Matryoshka Widths based on actual dict size
                matryoshka_widths = calculate_matryoshka_widths(total_dict_size, group_fractions)
                print(f"Recalculated Matryoshka Widths: {matryoshka_widths}")
                # -----------------------------------
                
                # Check for Gated MLP
                has_gate = False
                if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                     mlp = hf_model.model.layers[0].mlp
                     if hasattr(mlp, "gate_proj"):
                         has_gate = True
                
                print("Loading Tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # 3. Create HookedTransformerConfig
                tl_config = HookedTransformerConfig(
                    model_name=model_type, # Use detected type
                    n_layers=n_layers,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_head=head_dim,
                    d_mlp=d_mlp,
                    d_vocab=vocab_size,
                    n_ctx=8192, 
                    n_key_value_heads=n_kv_heads,
                    act_fn="gelu", # This is often overridden by weight conversion
                    gated_mlp=has_gate,
                    normalization_type="RMS",
                    positional_embedding_type="rotary",
                    rotary_dim=head_dim, 
                    device="cuda", # Match HF model device
                    dtype=torch.bfloat16
                )
                
                print("Creating HookedTransformer and converting weights...")
                
                # --- PROACTIVE DEQUANTIZATION ---
                # For any method other than standard floats, we force dequantization
                # This prevents "silent failures" where raw quantized weights are copied (e.g. FP8, Int4)
                if method not in ["bfloat16", "float16", "float32"]:
                    print(f"Method '{method}' detected. Forcing Universal Dequantization...")
                    try:
                        universal_dequantize_model(hf_model)
                        print("Proactive Dequantization complete.")
                    except Exception as e:
                        print(f"Proactive Dequantization failed: {e}")
                        # We continue and hope conversion works or fails loudly later
                
                model = HookedTransformer(tl_config, tokenizer=tokenizer)
                
                # Dynamic Conversion
                convert_fn = get_conversion_function(model_type)
                
                try:
                    state_dict = convert_fn(hf_model, tl_config)
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"Conversion failed: {e}")
                    # Universal Dequantization via Identity Matrix (Fallback)
                    try:
                        from torch.nn import Linear
                        print("Attempting Universal Dequantization (Identity Method) [Fallback]...")
                        
                        universal_dequantize_model(hf_model)
                        print("Dequantization complete. Retrying conversion...")
                        state_dict = convert_fn(hf_model, tl_config)
                        model.load_state_dict(state_dict, strict=False)
                        print("Conversion successful after dequantization!")
                        
                    except Exception as dequant_e:
                        print(f"Dequantization/Retry failed: {dequant_e}")
                        # print("Debug: Model Layer 0 Structure:")
                        # print(hf_model.model.layers[0])
                        raise e # Raise original error if fallback fails
                
                # Cleanup HF model to save RAM
                del hf_model
                import gc
                gc.collect()
                
                # 4. Configure SAELens Runner
                # layer_index is already calculated dynamically above
                act_site = config.get("activation_site", "resid_post")
                hook_name = f"blocks.{layer_index}.hook_{act_site}"
                
                # Create the specific SAE config first
                sae_config = MatryoshkaBatchTopKTrainingSAEConfig(
                    d_in=d_model,
                    d_sae=total_dict_size, # Use dynamic dict size
                    # Removed apply_b_dec_to_latents
                    
                    # Optimizer params 
                    k=sae_params.get("sparsity_k", 32),
                    matryoshka_widths=matryoshka_widths,
                    
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    dtype="float32" # Force SAE to be Float32 for stability
                )
                
                # Define Run Name uniquely for checkpoints and wandb
                run_name = f"{wandb_params.get('run_label', 'run')}-{method}"

                # Logging Config
                logging_config = LoggingConfig(
                    log_to_wandb=True,
                    wandb_project=wandb_params["project"],
                    wandb_entity=wandb_params.get("entity"),
                    run_name=run_name,
                    eval_every_n_wandb_logs=1_000_000, # Effectively disable online eval
                )

                runner_config = LanguageModelSAERunnerConfig(
                    sae=sae_config,
                    logger=logging_config,
                    
                    # Data
                    model_name="gemma-3-1b-custom",
                    hook_name=hook_name,
                    
                    dataset_path=dataset_params["path"],
                    is_dataset_tokenized=False,
                    streaming=True,
                    
                    # Batching
                    train_batch_size_tokens=dataset_params["batch_size"],
                    context_size=1024,
                    
                    # Training
                    training_tokens=dataset_params["num_tokens"],
                    
                    # Checkpointing
                    n_checkpoints=10,
                    checkpoint_path=f"checkpoints/{run_name}",
                    save_final_checkpoint=True,
                    
                    # Optimizer (Global settings for the runner)
                    lr=sae_params.get("learning_rate", 3e-4),
                    lr_warm_up_steps=sae_params.get("warmup_steps", 100),
                    
                    # Device
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    dtype="float32", # Force Runner to be Float32
                    
                    # Mixed Precision
                    autocast=False, # Disable autocast to avoid BF16 scaling issues
                )

                print("Starting Training Runner...")
                runner = LanguageModelSAETrainingRunner(runner_config, override_model=model)
                runner.run()
                
                print(f"--- Completed {method} ---")
                
            except Exception as e:
                print(f"Failed processing {method}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
