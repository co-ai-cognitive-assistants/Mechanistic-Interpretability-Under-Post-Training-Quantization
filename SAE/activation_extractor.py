import torch
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from config import Config
from data_loader import DataLoader
from typing import Iterator

class ActivationExtractor:
    def __init__(self, model_base_path: str, method: str, config: Config):
        self.model_path = os.path.join(model_base_path, method)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_idx = config.layer_index
        self.site = config.activation_site
        
        print(f"Loading model from {self.model_path}...")
        
        # Load Tokenizer
        # Try loading tokenizer from model path, else base path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        except:
            print(f"Tokenizer not found in {self.model_path}, trying base path {model_base_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        # Handle specific loading requirements based on method/config if needed
        # Most quantized models (AWQ, GPTQ) saved with save_pretrained can be loaded with AutoModel
        
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if method in ["bfloat16", "bf16"]:
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif method in ["float16", "fp16"]:
            load_kwargs["torch_dtype"] = torch.float16
        
        # For AWQ/GPTQ, AutoModel usually detects config. 
        # But sometimes we need to be explicit if config is missing? 
        # Assuming run_quantization.py saved them correctly.
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model.eval()
        
        self.activations_buffer = []
        self._register_hook()

    def _register_hook(self):
        # Identify layer module
        # Common architectures: model.layers, model.model.layers, model.transformer.h
        
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        else:
            # Fallback search
            for name, module in self.model.named_modules():
                if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
                    layers = module
                    break
                    
        if layers is None:
            raise ValueError("Could not locate transformer layers in model.")
            
        target_layer = layers[self.layer_idx]
        
        # Define hook function
        def hook_fn(module, inputs, outputs):
            # outputs is usually (hidden_states, past_key_values, ...)
            # We want hidden_states
            if isinstance(outputs, tuple):
                act = outputs[0]
            else:
                act = outputs
                
            # Act shape: [batch, seq_len, hidden_dim]
            # We move to CPU to save GPU memory if buffer is large, 
            # or keep on GPU if SAE is trained on same GPU.
            # Let's keep on same device as model output (likely GPU) to avoid transfer overhead,
            # but detach to free graph.
            self.activations_buffer.append(act.detach())

        # Register based on site
        if self.site == "resid_post":
            self.hook_handle = target_layer.register_forward_hook(hook_fn)
        elif self.site == "mlp_out":
            # Try to find MLP
            if hasattr(target_layer, "mlp"):
                target_layer.mlp.register_forward_hook(hook_fn)
            elif hasattr(target_layer, "feed_forward"):
                target_layer.feed_forward.register_forward_hook(hook_fn)
            else:
                raise ValueError(f"Could not find MLP module in layer {self.layer_idx}")
        elif self.site == "attn_out":
            if hasattr(target_layer, "self_attn"):
                target_layer.self_attn.register_forward_hook(hook_fn)
            elif hasattr(target_layer, "attention"):
                target_layer.attention.register_forward_hook(hook_fn)
            else:
                 raise ValueError(f"Could not find Attention module in layer {self.layer_idx}")
        else:
            raise ValueError(f"Unknown activation site: {self.site}")

    def get_activations(self, data_loader: DataLoader) -> Iterator[torch.Tensor]:
        for batch_tokens in data_loader:
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                self.model(batch_tokens)
                
            # Hook has populated self.activations_buffer
            # It might contain one or more batches if gradient accumulation was used (not here)
            # Usually just one.
            
            for acts in self.activations_buffer:
                # acts: [B, L, D]
                # Flatten to [B*L, D]
                # Filter out pad tokens? 
                # We can't easily filter pad tokens unless we passed attention_mask to hook.
                # But SAE usually trains on all tokens or valid ones.
                # Ideally we mask out padding.
                # For simplicity, we ignore padding masking for now as data_loader packs/truncates mostly.
                
                flat_acts = acts.reshape(-1, acts.shape[-1])
                yield flat_acts
                
            self.activations_buffer = []

    def close(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
        # Free memory
        del self.model
        torch.cuda.empty_cache()
