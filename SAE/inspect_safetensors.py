from safetensors.torch import load_file
import os

dir_path = "/experiment/SAE/checkpoints/DeepSeek-R1-1.5b-bfloat16/qg701fza/final_1000001536"
weights_path = os.path.join(dir_path, "sae_weights.safetensors")

if os.path.exists(weights_path):
    print(f"--- Inspecting {weights_path} ---")
    try:
        sd = load_file(weights_path)
        for k, v in sd.items():
            print(f"{k}: {v.shape}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {weights_path}")
