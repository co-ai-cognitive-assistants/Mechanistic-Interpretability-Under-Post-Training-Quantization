
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

model_path = "models/google_gemma-3-1b-it/bfloat16"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True)

print(f"Model type: {type(hf_model)}")
print("\n--- Model Attributes ---")
for attr in dir(hf_model):
    if not attr.startswith("_"):
        val = getattr(hf_model, attr)
        if isinstance(val, nn.Module):
             print(f"Module: {attr} ({type(val)})")

print("\n--- Inner Model (hf_model.model) if exists ---")
if hasattr(hf_model, "model"):
    inner = hf_model.model
    print(f"Inner type: {type(inner)}")
    for attr in dir(inner):
        if not attr.startswith("_"):
            val = getattr(inner, attr)
            if isinstance(val, nn.Module):
                 print(f"Module: {attr} ({type(val)})")
    
    if hasattr(inner, "layers"):
        print(f"\nNumber of layers: {len(inner.layers)}")
        layer0 = inner.layers[0]
        print(f"Layer 0 type: {type(layer0)}")
        for attr in dir(layer0):
             if not attr.startswith("_"):
                val = getattr(layer0, attr)
                if isinstance(val, nn.Module):
                     print(f"  Module: {attr} ({type(val)})")

if hasattr(hf_model, "lm_head"):
    print(f"\nlm_head: {hf_model.lm_head}")
