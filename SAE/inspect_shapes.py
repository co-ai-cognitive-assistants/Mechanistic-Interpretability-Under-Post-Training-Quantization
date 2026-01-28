import torch
import os

file_path = "/experiment/SAE/checkpoints/google_gemma-3-1b-it_bfloat16_sae_gemma3-1b/checkpoint_244000.pt"

print(f"--- Inspecting {file_path} ---")
try:
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        sd = data["model_state_dict"]
        for k, v in sd.items():
            if hasattr(v, "shape"):
                print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Error: {e}")
