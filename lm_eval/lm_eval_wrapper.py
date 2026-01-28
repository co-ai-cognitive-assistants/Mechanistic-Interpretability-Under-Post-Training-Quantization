import sys
import os
import copy
import torch
import json
import transformers.models.auto.configuration_auto

# Try to import llmcompressor to register compressed-tensors config
try:
    import llmcompressor
except ImportError:
    pass

import transformers.configuration_utils

def sanitize_dict(d):
    """Recursively convert torch.dtype to string."""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    elif isinstance(d, torch.dtype):
        return str(d).split(".")[1]
    return d

# Monkeypatch AutoConfig.from_pretrained to fix quantization_config if it's None
original_from_pretrained = transformers.models.auto.configuration_auto.AutoConfig.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    result = original_from_pretrained(*args, **kwargs)
    if isinstance(result, tuple):
        config = result[0]
    else:
        config = result
    
    # Fix quantization_config if it got lost (became None) during loading
    if hasattr(config, "quantization_config") and config.quantization_config is None:
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None)
        if not pretrained_model_name_or_path and args:
            pretrained_model_name_or_path = args[0]
        
        if pretrained_model_name_or_path and isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        raw_config = json.load(f)
                        if "quantization_config" in raw_config:
                            config.quantization_config = raw_config["quantization_config"]
                            print(f"Wrapper: Manually restored quantization_config from {config_file}", file=sys.stderr)
                except Exception as e:
                    print(f"Wrapper: Failed to restore quantization_config: {e}", file=sys.stderr)

    return result

transformers.models.auto.configuration_auto.AutoConfig.from_pretrained = patched_from_pretrained

# Monkeypatch PreTrainedModel.from_pretrained to ignore fix_mistral_regex if passed to model
import transformers.modeling_utils
original_model_from_pretrained = transformers.modeling_utils.PreTrainedModel.from_pretrained

@classmethod
def patched_model_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    # Remove fix_mistral_regex if present, as models generally don't accept it in __init__
    if "fix_mistral_regex" in kwargs:
        # We assume tokenizer loading (handled separately) will still use it if passed to it
        kwargs.pop("fix_mistral_regex")
    return original_model_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

transformers.modeling_utils.PreTrainedModel.from_pretrained = patched_model_from_pretrained

# Monkeypatch PretrainedConfig.to_dict and to_diff_dict to fix AttributeError: 'NoneType' object has no attribute 'to_dict'
original_to_dict = transformers.configuration_utils.PretrainedConfig.to_dict
original_to_diff_dict = transformers.configuration_utils.PretrainedConfig.to_diff_dict

def safe_to_dict(self):
    try:
        return original_to_dict(self)
    except AttributeError as e:
        if "object has no attribute 'to_dict'" in str(e):
             output = copy.deepcopy(self.__dict__)
             if hasattr(self, "return_dict"):
                 output.pop("return_dict", None)
             
             # Handle quantization_config
             if hasattr(self, "quantization_config"):
                 if self.quantization_config is None:
                     output["quantization_config"] = None
                 elif isinstance(self.quantization_config, dict):
                     output["quantization_config"] = self.quantization_config
                 else:
                     try:
                         output["quantization_config"] = self.quantization_config.to_dict()
                     except:
                         output["quantization_config"] = self.quantization_config
             
             return sanitize_dict(output)
        raise e

def safe_to_diff_dict(self):
    try:
        return original_to_diff_dict(self)
    except AttributeError as e:
        if "object has no attribute 'to_dict'" in str(e):
             return safe_to_dict(self)
        raise e

transformers.configuration_utils.PretrainedConfig.to_dict = safe_to_dict
transformers.configuration_utils.PretrainedConfig.to_diff_dict = safe_to_diff_dict

# Monkeypatch to ensure .items() exists on Config objects (fixes Gemma 3 AWQ/GPTQ)
def add_items_method(cls):
    if not hasattr(cls, 'items'):
        def items(self):
            return self.to_dict().items()
        cls.items = items

# Apply to base PretrainedConfig and specific Gemma3TextConfig if it exists
add_items_method(transformers.configuration_utils.PretrainedConfig)
try:
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    add_items_method(Gemma3TextConfig)
except ImportError:
    pass

# Import the main entry point of lm_eval
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.exit(cli_evaluate())