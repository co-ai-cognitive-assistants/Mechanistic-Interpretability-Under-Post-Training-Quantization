import torch
import os
import json

file_path = "/experiment/SAE/checkpoints/google_gemma-3-1b-it_bfloat16_sae_gemma3-1b/checkpoint_244000.pt"
dir_path = "/experiment/SAE/checkpoints/DeepSeek-R1-1.5b-bfloat16/qg701fza/final_1000001536"

print(f"--- Inspecting {file_path} ---")
try:
    # Use weights_only=False as it seems to be a custom object
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:10])
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"{k}: {v.shape}")
            elif isinstance(v, (dict, list, int, float, str)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {type(v)}")
    elif hasattr(data, "__dict__"):
         print("Object attributes:", list(data.__dict__.keys()))
         # If it has a state_dict or similar
         if hasattr(data, "state_dict"):
             sd = data.state_dict()
             print("State dict keys:", list(sd.keys())[:10])
except Exception as e:
    print(f"Error loading .pt: {e}")

print(f"\n--- Inspecting Directory {dir_path} ---")
for f in os.listdir(dir_path):
    print(f"Found file: {f}")

cfg_path = os.path.join(dir_path, "cfg.json")
if os.path.exists(cfg_path):
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        print("Config keys:", list(cfg.keys()))
    except Exception as e:
        print(f"Error reading config: {e}")