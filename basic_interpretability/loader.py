import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.pretrained.weight_conversions as wc
import transformers.activations
import gc

# --- Monkey Patch for AutoAWQ / Transformers Compatibility ---
# AutoAWQ imports PytorchGELUTanh from transformers.activations,
# which was renamed to GELUTanh in newer transformers versions.
if not hasattr(transformers.activations, "PytorchGELUTanh") and hasattr(transformers.activations, "GELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

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
    
    # Verify we found something with embed_tokens
    if not hasattr(gemma, "embed_tokens"):
        raise AttributeError(f"Could not find 'embed_tokens' in extracted model of type {type(gemma)}. attributes: {list(gemma.__dict__.keys())[:20]}")

    # Embeddings: Gemma 3 scales embeddings by sqrt(hidden_size)
    # In HookedTransformer, we can either scale the weights or use a multiplier.
    # We'll scale the weights in the state_dict for simplicity.
    scale_factor = (cfg.d_model**0.5)
    state_dict["embed.W_E"] = gemma.embed_tokens.weight * scale_factor
    
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
        # Gemma 3 has 4 norms. We map the pre-layer ones to HookedTransformer's ln1/ln2.
        # Note: post_attention_layernorm and post_feedforward_layernorm are currently 
        # not supported by the standard HookedTransformer block and are skipped.
        state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight
        state_dict[f"blocks.{l}.ln2.w"] = layer.pre_feedforward_layernorm.weight
        
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
    mapping = {
        "gemma": "convert_gemma_weights",
        "gemma2": "convert_gemma_weights",
        "gemma3": convert_gemma3_weights, 
        "gemma3_text": convert_gemma3_weights,
        "llama": "convert_llama_weights",
        "mistral": "convert_mistral_weights",
        "qwen2": "convert_qwen2_weights",
        "qwen": "convert_qwen2_weights",
        "qwen3": "convert_qwen3_weights",
        "qwen3_moe": "convert_qwen3_weights", # Treat MoE as standard dense for now if TL supports it, or it might need specific handling 
    }
    
    val = mapping.get(model_type)
    if val:
        if callable(val):
            return val
        elif hasattr(wc, val):
            return getattr(wc, val)
    
    # Fallback: Try to guess function name "convert_{type}_weights"
    guessed_name = f"convert_{model_type}_weights"
    if hasattr(wc, guessed_name):
        return getattr(wc, guessed_name)
        
    raise ValueError(f"No conversion function found for model type '{model_type}'. Available: {dir(wc)}")

def universal_dequantize_model(module):
    """
    Recursively replaces quantized layers with standard nn.Linear layers 
    by projecting the identity matrix through them (functional dequantization).
    """
    from torch.nn import Linear
    for name, child in module.named_children():
        # Recursive call first
        universal_dequantize_model(child)
        
        # --- Method A: Manual Extraction for FP8 (HuggingFace/Accelerate style) ---
        if hasattr(child, "weight") and hasattr(child, "weight_scale_inv"):
             # print(f"Manual Dequantization (FP8/ScaleInv) for {name}...")
             try:
                 device = child.weight.device
                 target_dtype = torch.bfloat16
                 
                 w_float = child.weight.float()
                 s_float = child.weight_scale_inv.float()
                 
                 if w_float.shape != s_float.shape:
                     if w_float.dim() == s_float.dim():
                         for dim in range(w_float.dim()):
                             w_dim = w_float.shape[dim]
                             s_dim = s_float.shape[dim]
                             
                             if w_dim != s_dim:
                                 if w_dim % s_dim == 0:
                                     factor = w_dim // s_dim
                                     s_float = s_float.repeat_interleave(factor, dim=dim)
                 
                 w_extracted = w_float * s_float
                 
                 if w_extracted.shape == (child.out_features, child.in_features):
                     weight_data = w_extracted.to(target_dtype)
                 elif w_extracted.shape == (child.in_features, child.out_features):
                     weight_data = w_extracted.T.to(target_dtype)
                 else:
                     raise ValueError("Shape mismatch in manual extraction")
                 
                 new_layer = Linear(child.in_features, child.out_features, 
                                  bias=child.bias is not None, 
                                  device=device, dtype=target_dtype)
                 new_layer.weight.data = weight_data
                 
                 if child.bias is not None:
                     new_layer.bias.data = child.bias.data.to(target_dtype)
                 
                 setattr(module, name, new_layer)
                 continue 
                 
             except Exception as e:
                 print(f"  Manual extraction failed: {e}. Falling back to Identity method...")

        # --- Method B: BitsAndBytes Specific Dequantization ---
        if "Linear4bit" in type(child).__name__ or "Linear8bit" in type(child).__name__:
            try:
                import bitsandbytes as bnb
                target_dtype = torch.bfloat16
                device = child.weight.device
                
                if hasattr(child.weight, "quant_state"):
                    q_weight = child.weight
                    q_state = q_weight.quant_state
                    
                    if "4bit" in type(child).__name__:
                        dequantized = bnb.functional.dequantize_4bit(q_weight.data, q_state)
                    else:
                        dequantized = bnb.functional.dequantize_blockwise(q_weight.data, q_state)
                    
                    weight_data = dequantized.to(target_dtype)
                    
                    new_layer = Linear(child.in_features, child.out_features, 
                                     bias=child.bias is not None, 
                                     device=device, dtype=target_dtype)
                    new_layer.weight.data = weight_data
                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data.to(target_dtype)
                        
                    setattr(module, name, new_layer)
                    continue
            except Exception as e:
                print(f"  BitsAndBytes dequantization failed for {name}: {e}. Falling back...")

        # --- Method C: Generic Identity Matrix Method ---
        n_in = getattr(child, "in_features", getattr(child, "infeatures", None))
        n_out = getattr(child, "out_features", getattr(child, "outfeatures", None))
        
        if n_in is not None and n_out is not None and \
           type(child) is not Linear and \
           not isinstance(child, (nn.Sequential, nn.ModuleList)):
            
            # print(f"Dequantizing {name} ({type(child).__name__})...") 
            
            success = False
            errors = []
            
            # Smart Dtype Detection for Kernel
            # GPTQ often demands FP16. AWQ demands FP16.
            # We try to infer from the layer's qweight or weight
            kernel_dtype = torch.float16 # Default safe bet for quantized kernels
            
            # Helper to detect if it's likely a quantized linear layer needing half precision
            is_quant_layer = False
            layer_type_name = type(child).__name__.lower()
            if "quant" in layer_type_name or \
               "awq" in layer_type_name or \
               "gptq" in layer_type_name or \
               "4bit" in layer_type_name or \
               "8bit" in layer_type_name or \
               hasattr(child, "qweight"):
                is_quant_layer = True
            
            # If it's a known quantized layer, prioritize Float16
            if is_quant_layer:
                dtypes_to_try = [torch.float16, torch.bfloat16, torch.float32]
            else:
                # Otherwise try native first
                dtypes_to_try = [torch.float16, torch.bfloat16, torch.float32]
                if hasattr(child, "weight"):
                    dtypes_to_try.insert(0, child.weight.dtype)
            
            # Deduplicate while preserving order
            dtypes_to_try = list(dict.fromkeys(dtypes_to_try))

            for extract_dtype in dtypes_to_try:
                try:
                    # Determine device robustly
                    device = getattr(child, "device", None)
                    if device is None:
                        if hasattr(child, "weight") and hasattr(child.weight, "device"):
                            device = child.weight.device
                        elif hasattr(child, "qweight") and hasattr(child.qweight, "device"):
                            device = child.qweight.device
                        elif hasattr(child, "scales") and hasattr(child.scales, "device"):
                            device = child.scales.device
                        else:
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    target_dtype = torch.bfloat16
                    
                    # Create input on the correct device and dtype
                    eye = torch.eye(n_in, device=device, dtype=extract_dtype)
                    
                    with torch.no_grad():
                        # Some kernels crash if input is not contiguous
                        eye = eye.contiguous()
                        output = child(eye)
                    
                    bias_data = None
                    if hasattr(child, "bias") and child.bias is not None:
                        bias = child.bias.to(output.dtype) # Match output
                        weight_T = output - bias
                        bias_data = child.bias.data.to(target_dtype)
                    else:
                        weight_T = output
                    
                    weight = weight_T.T.to(target_dtype)
                    
                    new_layer = Linear(n_in, n_out, 
                                     bias=bias_data is not None, 
                                     device=device, dtype=target_dtype)
                    new_layer.weight.data = weight
                    if bias_data is not None:
                        new_layer.bias.data = bias_data
                        
                    setattr(module, name, new_layer)
                    success = True
                    break 
                    
                except Exception as e:
                    errors.append(f"{extract_dtype}: {str(e)}")
            
            if not success:
                print(f"Failed to dequantize {name} with all dtypes. Errors: {errors}")
            else:
                # print(f"Successfully dequantized {name}")
                pass

def load_model_for_lens(model_path: str, device: str = "cuda") -> HookedTransformer:
    """
    Loads a model (quantized or not), functionally dequantizes it if necessary,
    and returns a HookedTransformer ready for analysis.
    
    Optimized for memory: Loads HF to GPU (distributed), converts to CPU HookedTransformer,
    clears HF, then moves HookedTransformer to GPU.
    """
    print(f"--- Loading Model: {model_path} ---")
    
    # 1. Determine Method / Loading Args
    # STRICTLY force the specified device. No "auto", no CPU offload.
    print(f"Forcing model to device: {device}")
    
    load_kwargs = {
        "device_map": {"": device}, # Force everything to the target device
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16, # Default target
    }
    
    # Check for formats requiring Float16 (GPTQ, AWQ often crash with BF16)
    if "gptq" in model_path.lower() or "awq" in model_path.lower():
        print("Detected GPTQ/AWQ: Switching load dtype to float16 for kernel compatibility.")
        load_kwargs["torch_dtype"] = torch.float16
    
    # Config Loading
    has_quant_config = False
    try:
        auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        has_quant_config = hasattr(auto_config, "quantization_config") and auto_config.quantization_config is not None
    except Exception as e:
        print(f"Warning: Could not load AutoConfig: {e}")
        has_quant_config = False

    # GPTQ specific check
    if "gptq" in model_path.lower() and not has_quant_config:
        try:
            from transformers import GPTQConfig
            load_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)
            print("Injected GPTQConfig.")
        except ImportError:
            pass
            
    # HQQ Specific Check
    if "hqq" in model_path.lower():
        try:
            import hqq
            print("HQQ library imported.")
        except ImportError:
            print("Warning: HQQ model detected but 'hqq' library not installed.")

    print(f"Loading HF Model with kwargs: {list(load_kwargs.keys())}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    # 2. Extract Config Details
    hf_config = hf_model.config
    model_type = getattr(hf_config, "model_type", "unknown")
    
    # Handle Multimodal Configs (Gemma 3)
    target_config = hf_config
    if hasattr(hf_config, "text_config"):
        target_config = hf_config.text_config
    
    def get_config_attr(cfg, attr, default=None):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
        if default is not None: return default
        raise ValueError(f"Missing required config attribute '{attr}' in {cfg}")

    n_layers = get_config_attr(target_config, "num_hidden_layers")
    d_model = get_config_attr(target_config, "hidden_size")
    n_heads = get_config_attr(target_config, "num_attention_heads")
    n_kv_heads = get_config_attr(target_config, "num_key_value_heads")
    d_mlp = get_config_attr(target_config, "intermediate_size")
    vocab_size = get_config_attr(target_config, "vocab_size")
    
    if hasattr(target_config, "head_dim"):
        head_dim = target_config.head_dim
    else:
        head_dim = d_model // n_heads

    # Check for Gated MLP
    has_gate = False
    try:
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                mlp = hf_model.model.layers[0].mlp
                if hasattr(mlp, "gate_proj"):
                    has_gate = True
    except:
        pass

    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer from {model_path}: {e}")
        raise e
    
    # 3. Create HookedTransformerConfig
    # Initialize on CPU first to save VRAM
    tl_config = HookedTransformerConfig(
        model_name=model_type,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_head=head_dim,
        d_mlp=d_mlp,
        d_vocab=vocab_size,
        n_ctx=8192, 
        n_key_value_heads=n_kv_heads,
        act_fn="gelu",
        gated_mlp=has_gate,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=head_dim, 
        device="cpu", # Loading target: CPU
        dtype=torch.bfloat16
    )
    
    # 4. Proactive Dequantization
    print("Running Universal Dequantization...")
    universal_dequantize_model(hf_model)
    
    # 5. Convert to HookedTransformer (CPU)
    print("Converting to HookedTransformer (on CPU)...")
    model = HookedTransformer(tl_config, tokenizer=tokenizer)
    
    convert_fn = get_conversion_function(model_type)
    
    try:
        state_dict = convert_fn(hf_model, tl_config)
        # Ensure state_dict is on CPU
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        model.load_state_dict(cpu_state_dict, strict=False)
        del state_dict
        del cpu_state_dict
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise e
    
    # 6. Cleanup HF Model
    print("Cleaning up HF Model...")
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 7. Move to GPU
    print(f"Moving HookedTransformer to target device: {device}...")
    model.to(device)
    
    print(f"--- Loaded {model_type} into HookedTransformer ---")
    return model

def load_hf_dequantized(model_path: str, device: str = "cuda"):
    """
    Loads a model and dequantizes it, returning the HF model directly.
    Unified backend for all architectures (Gemma 3, Qwen, etc).
    """
    print(f"--- Loading HF Model: {model_path} ---")

    # 1. Determine Method / Loading Args
    load_kwargs = {
        "device_map": {"": device},
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    # Check for Gemma 3 to force eager attention
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if "gemma3" in getattr(config, "model_type", "").lower():
            load_kwargs["attn_implementation"] = "eager"
    except:
        pass

    if "gptq" in model_path.lower() or "awq" in model_path.lower():
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # 2. Dequantize
    universal_dequantize_model(model)

    # 3. Cast entire model to bfloat16 for consistency after dequantization
    # This ensures AWQ/GPTQ models (loaded as float16) match baseline dtype
    model = model.to(torch.bfloat16)

    # 4. Add helper attributes for unified analysis
    model.name_or_path = model_path

    return model
