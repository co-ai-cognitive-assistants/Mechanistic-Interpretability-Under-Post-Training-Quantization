from safetensors.torch import load_file
import sys

def inspect():
    path = "/experiment/SAE/checkpoints/google_gemma-3-1b-it-awq_sae_gemma3-1b/edw6gsei/final_1000001536/sae_weights.safetensors"
    sd = load_file(path)
    print(f"Keys in {path}:")
    for k, v in sd.items():
        print(f"  {k}: {v.shape}")

if __name__ == "__main__":
    inspect()
