import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_interpretability.loader import load_model_for_lens
import torch

def test_load():
    # Path to a model - adjust based on what's available
    # Trying int4 first as it likely triggers the dequantization logic
    model_path = "models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/int4"
    
    if not os.path.exists(model_path):
        print(f"Path {model_path} does not exist. checking bfloat16...")
        model_path = "models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/bfloat16"
        
    if not os.path.exists(model_path):
        print("No suitable model found to test. Exiting.")
        return

    print(f"Testing loader with {model_path}...")
    try:
        model = load_model_for_lens(model_path)
        print("Success! Model loaded.")
        print(f"Model type: {type(model)}")
        print(f"Layer 0 W_O shape: {model.blocks[0].attn.W_O.shape}")
        
        # Simple inference check
        output = model.generate("Hello, world!", max_new_tokens=5)
        print(f"Generation: {output}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
