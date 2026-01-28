import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from utils_loading import load_hooked_model

def check_activations():
    model_path = "/experiment/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/bfloat16"
    device = "cuda"
    
    print(f"Loading Native HF Model from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading HookedTransformer (Manual Conversion)...")
    tl_model = load_hooked_model(model_path, device=device)
    
    input_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # 1. Check Output Logits
    print("\nChecking Logits...")
    with torch.no_grad():
        hf_out = hf_model(tokens).logits
        tl_out = tl_model(tokens)
    
    diff = (hf_out - tl_out).abs().mean().item()
    print(f"Logit Mean Diff: {diff:.6f}")
    if diff > 0.1:
        print("CRITICAL: Logits mismatch!")
    
    # 2. Check Layer Activations (Layer 21)
    layer_idx = 21
    print(f"\nChecking Layer {layer_idx} Activations...")
    
    # HF Hook
    hf_acts = {}
    def get_hf_hook(name):
        def hook(module, input, output):
            hf_acts[name] = output[0] if isinstance(output, tuple) else output
        return hook
    
    # Qwen2 structure: model.layers[i]
    hf_layer = hf_model.model.layers[layer_idx]
    handle = hf_layer.register_forward_hook(get_hf_hook("layer_21"))
    
    with torch.no_grad():
        hf_model(tokens)
    handle.remove()
    
    # TL Hook
    _, cache = tl_model.run_with_cache(tokens, names_filter=[f"blocks.{layer_idx}.hook_resid_post"])
    
    if f"blocks.{layer_idx}.hook_resid_post" in cache:
        tl_act = cache[f"blocks.{layer_idx}.hook_resid_post"]
    else:
        print(f"Key not found! Available keys: {list(cache.keys())[:5]}...")
        return

    hf_act = hf_acts["layer_21"]
    
    print(f"HF Act Shape: {hf_act.shape}")
    print(f"TL Act Shape: {tl_act.shape}")
    
    act_diff = (hf_act - tl_act).abs().mean().item()
    print(f"Layer {layer_idx} Act Mean Diff: {act_diff:.6f}")
    
    if act_diff > 0.1:
        print("CRITICAL: Internal Activations mismatch!")

if __name__ == "__main__":
    check_activations()
