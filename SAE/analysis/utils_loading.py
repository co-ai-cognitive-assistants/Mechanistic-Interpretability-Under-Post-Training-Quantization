import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.pretrained.weight_conversions as wc

# --- Monkey Patch for AutoAWQ / Transformers Compatibility ---
import transformers.activations
if not hasattr(transformers.activations, "PytorchGELUTanh") and hasattr(transformers.activations, "GELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

def convert_gemma3_weights(model, cfg: HookedTransformerConfig):
    """
    Converts Gemma 3 weights to HookedTransformer format.
    """
    state_dict = {}
    
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        gemma = model.model.language_model
    elif hasattr(model, "language_model"):
        gemma = model.language_model
    elif hasattr(model, "model"):
        gemma = model.model
    elif hasattr(model, "transformer"):
        gemma = model.transformer
    else:
        gemma = model
    
    if not hasattr(gemma, "embed_tokens"):
        # Attempt to find it recursively if needed, or just fail
        raise AttributeError(f"Could not find 'embed_tokens' in extracted model of type {type(gemma)}")

    state_dict["embed.W_E"] = gemma.embed_tokens.weight
    
    for l in range(cfg.n_layers):
        layer = gemma.layers[l]
        
        # Attention Q, K, V
        w_q = layer.self_attn.q_proj.weight
        state_dict[f"blocks.{l}.attn.W_Q"] = w_q.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        w_k = layer.self_attn.k_proj.weight
        state_dict[f"blocks.{l}.attn.W_K"] = w_k.reshape(cfg.n_key_value_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        w_v = layer.self_attn.v_proj.weight
        state_dict[f"blocks.{l}.attn.W_V"] = w_v.reshape(cfg.n_key_value_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
        # Attention O
        w_o = layer.self_attn.o_proj.weight
        state_dict[f"blocks.{l}.attn.W_O"] = w_o.reshape(cfg.d_model, cfg.n_heads, cfg.d_head).permute(1, 2, 0)
        
        # MLPs
        state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        
        # Norms
        state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight
        state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight
        
    state_dict["ln_final.w"] = gemma.norm.weight
    
    if hasattr(model, "lm_head"):
        state_dict["unembed.W_U"] = model.lm_head.weight.T
    else:
        state_dict["unembed.W_U"] = gemma.embed_tokens.weight.T
        
    return state_dict

def get_conversion_function(model_type):
    model_type = model_type.lower()
    mapping = {
        "gemma": "convert_gemma_weights",
        "gemma2": "convert_gemma_weights",
        "gemma3": convert_gemma3_weights,
        "gemma3_text": convert_gemma3_weights,
        "llama": "convert_llama_weights",
        "mistral": "convert_mistral_weights",
        "qwen2": "convert_qwen2_weights",
        "qwen": "convert_qwen2_weights",
    }
    
    val = mapping.get(model_type)
    if val:
        if callable(val):
            return val
        elif hasattr(wc, val):
            return getattr(wc, val)
    
    guessed_name = f"convert_{model_type}_weights"
    if hasattr(wc, guessed_name):
        return getattr(wc, guessed_name)
        
    raise ValueError(f"No conversion function found for model type '{model_type}'.")

def universal_dequantize_model(module):
    from torch.nn import Linear
    for name, child in module.named_children():
        universal_dequantize_model(child)
        
        # FP8 / ScaleInv Logic
        if hasattr(child, "weight") and hasattr(child, "weight_scale_inv"):
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
                             if w_dim != s_dim and w_dim % s_dim == 0:
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
                 print(f"  Manual extraction failed: {e}. Falling back...")

        # Identity Matrix Method
        n_in = getattr(child, "in_features", getattr(child, "infeatures", None))
        n_out = getattr(child, "out_features", getattr(child, "outfeatures", None))
        
        if n_in is not None and n_out is not None and \
           type(child) is not Linear and \
           not isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList)):
            
            print(f"Dequantizing {name} ({type(child).__name__})...") 
            success = False
            for extract_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                try:
                    device = getattr(child, "device", "cuda")
                    if hasattr(child, "weight") and hasattr(child.weight, "device"):
                        device = child.weight.device
                    elif hasattr(child, "qweight") and hasattr(child.qweight, "device"):
                        device = child.qweight.device
                        
                    target_dtype = torch.bfloat16
                    eye = torch.eye(n_in, device=device, dtype=extract_dtype)
                    
                    with torch.no_grad():
                        output = child(eye)
                    
                    bias_data = None
                    if hasattr(child, "bias") and child.bias is not None:
                        bias = child.bias.to(extract_dtype)
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
                except Exception:
                    continue
            
            if not success:
                print(f"Failed to dequantize {name}.")

def load_hooked_model(model_path, device="cuda", dtype=None):
    """
    Robustly loads a model, dequantizes it if necessary, and converts it to HookedTransformer.

    The dtype is automatically inferred from the model path to match train_saelens.py:
    - bfloat16 paths → torch.bfloat16
    - Other paths → torch.float16
    """
    print(f"Loading HF Model from {model_path}...")

    # Infer dtype from path to match train_saelens.py behavior
    if dtype is None:
        if "bfloat16" in model_path.lower():
            dtype = torch.bfloat16
            print(f"  Inferred dtype: bfloat16 (from path)")
        else:
            dtype = torch.float16
            print(f"  Inferred dtype: float16 (default)")

    # Heuristic for dequantization trigger
    is_quantized = any(x in model_path.lower() for x in ["int4", "int8", "awq", "gptq", "bnb", "fp8"])

    if not is_quantized:
        try:
            print("  Attempting native HookedTransformer load...")
            model = HookedTransformer.from_pretrained(
                model_path,
                device=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            return model
        except Exception as e:
            print(f"  Native load failed: {e}. Falling back to manual conversion...")

    load_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
        "torch_dtype": dtype
    }
    
    try:
        auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        has_quant_config = hasattr(auto_config, "quantization_config") and auto_config.quantization_config is not None
    except Exception as e:
        print(f"Warning: Could not load AutoConfig: {e}")
        has_quant_config = False

    if "gptq" in model_path.lower() and not has_quant_config:
         print("Injecting GPTQConfig...")
         from transformers import GPTQConfig
         load_kwargs["quantization_config"] = GPTQConfig(bits=4, disable_exllama=True)
    
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    # Config Extraction
    hf_config = hf_model.config
    model_type = getattr(hf_config, "model_type", "unknown")
    target_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    
    def get_attr(cfg, attr, default=None):
        return getattr(cfg, attr, default)

    n_layers = get_attr(target_config, "num_hidden_layers")
    d_model = get_attr(target_config, "hidden_size")
    n_heads = get_attr(target_config, "num_attention_heads")
    n_kv_heads = get_attr(target_config, "num_key_value_heads")
    d_mlp = get_attr(target_config, "intermediate_size")
    vocab_size = get_attr(target_config, "vocab_size")
    head_dim = getattr(target_config, "head_dim", d_model // n_heads)
    act_fn = get_attr(target_config, "hidden_act", "gelu")
    eps = get_attr(target_config, "rms_norm_eps", 1e-5)

    # Check Gated MLP
    has_gate = False
    try:
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
             mlp = hf_model.model.layers[0].mlp
             if hasattr(mlp, "gate_proj"):
                 has_gate = True
    except:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
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
        act_fn=act_fn,
        gated_mlp=has_gate,
        normalization_type="RMS" if "rms" in str(get_attr(target_config, "architectures", [])).lower() or "qwen" in model_type or "llama" in model_type else "LN",
        eps=eps,
        positional_embedding_type="rotary",
        rotary_dim=head_dim, 
        device=device, 
        dtype=torch.bfloat16
    )
    
    # Dequantize if needed
    if is_quantized:
        print("Forcing Universal Dequantization...")
        try:
            universal_dequantize_model(hf_model)
        except Exception as e:
            print(f"Dequantization error: {e}")

    model = HookedTransformer(tl_config, tokenizer=tokenizer)
    convert_fn = get_conversion_function(model_type)
    
    try:
        state_dict = convert_fn(hf_model, tl_config)
        info = model.load_state_dict(state_dict, strict=False)
        if len(info.missing_keys) > 0:
            print(f"  Missing keys: {info.missing_keys[:5]}... (Total {len(info.missing_keys)})")
        if len(info.unexpected_keys) > 0:
            print(f"  Unexpected keys: {info.unexpected_keys[:5]}... (Total {len(info.unexpected_keys)})")
    except Exception as e:
        print(f"Conversion failed ({e}). Retrying with fallback dequantization...")
        universal_dequantize_model(hf_model)
        state_dict = convert_fn(hf_model, tl_config)
        info = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys: {len(info.missing_keys)}, Unexpected: {len(info.unexpected_keys)}")
    
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return model
