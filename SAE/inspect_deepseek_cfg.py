import json
import os

dir_path = "/experiment/SAE/checkpoints/DeepSeek-R1-1.5b-bfloat16/qg701fza/final_1000001536"
cfg_path = os.path.join(dir_path, "cfg.json")

if os.path.exists(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    print("Architecture:", cfg.get("architecture"))
    print("k:", cfg.get("k"))
    print("d_in:", cfg.get("d_in"))
    print("d_sae:", cfg.get("d_sae"))
