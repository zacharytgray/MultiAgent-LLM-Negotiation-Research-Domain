import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Optional, Dict, Union, Any
import math
import os
import json

class RhoContext:
    """
    Context object to hold the current rho (scalar control) state for the model.
    This avoids global variables or fragile closures.
    """
    def __init__(self):
        self.current_rho: Optional[torch.Tensor] = None

class RhoEncoderFourier(nn.Module):
    """
    Encodes scalar rho using Fourier features + optional raw value.
    
    Output dimension = (1 if include_raw else 0) + 2 * num_frequencies
    """
    def __init__(self, num_frequencies: int = 8, include_raw: bool = True, scale: float = 1.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_raw = include_raw
        self.scale = scale
        self.out_dim = (1 if include_raw else 0) + 2 * num_frequencies
        
        # Precompute frequencies 2^k * pi * scale
        # We store them as buffer so they move with device
        frequencies = [math.pi * scale * (2**k) for k in range(num_frequencies)]
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        rho: [B, 1] float32
        returns: [B, out_dim]
        """
        # Ensure rho is physically on the same device as frequencies
        if rho.device != self.frequencies.device:
            rho = rho.to(self.frequencies.device)

        features = []
        if self.include_raw:
            features.append(rho)
        
        for freq in self.frequencies:
            # rho * freq broadcast -> [B, 1]
            arg = rho * freq
            features.append(torch.sin(arg))
            features.append(torch.cos(arg))
            
        return torch.cat(features, dim=-1)

class RhoHyperNet(nn.Module):
    """
    Hypernetwork: Embeds rho (optionally) -> MLP -> Gating vector g.
    """
    def __init__(
        self, 
        rank: int, 
        input_dim: int = 1, # Changed from implicit 1 to explicit
        hidden_dim: int = 64, 
        gate_fn: str = 'sigmoid',
        gate_clamp_min: Optional[float] = None,
        gate_clamp_max: Optional[float] = None
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.gate_fn = gate_fn.lower()
        self.gate_clamp_min = gate_clamp_min
        self.gate_clamp_max = gate_clamp_max
        
        # MLP Layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank)
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [Batch_Size, input_dim]
        Returns:
            g: Gating values [Batch_Size, rank]
        """
        # Ensure x is on the same device/dtype as the network parameters
        # Usually float32 is preferred for the hypernet MLP
        target_dtype = self.net[0].weight.dtype
        target_device = self.net[0].weight.device
        
        x_in = x.to(dtype=target_dtype, device=target_device)
        
        logits = self.net(x_in)
        
        # Apply Gating Function
        if self.gate_fn == 'sigmoid':
            out = torch.sigmoid(logits)
        elif self.gate_fn == 'tanh':
            # Tanh is (-1, 1). If we want (0, 1), we rescale: (tanh + 1) / 2
            # Or assume the user wants signed gates (rare for diagonal scaling but possible)
            # Defaulting to standard tanh (-1, 1)
            out = torch.tanh(logits)
        elif self.gate_fn == 'tanh01':
            out = (torch.tanh(logits) + 1.0) / 2.0
        elif self.gate_fn == 'softplus':
            out = F.softplus(logits)
        elif self.gate_fn == 'identity':
            out = logits
        else:
            raise ValueError(f"Unknown gate_fn: {self.gate_fn}")
            
        # Optional Clamping
        if (self.gate_clamp_min is not None) or (self.gate_clamp_max is not None):
            out = torch.clamp(out, min=self.gate_clamp_min, max=self.gate_clamp_max)
            
        return out

class HyperLoRALinear(nn.Module):
    """
    Wraps a base nn.Linear layer with HyperLoRA adaptation.
    W = W_base + (alpha/rank) * ( B @ diag(g(rho)) @ A )
    
    Refactored to usage RhoContext and support cleaner Injection.
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int, 
        alpha: float, 
        rho_context: RhoContext,
        rho_encoder: Optional[RhoEncoderFourier], 
        hyper_net: RhoHyperNet,
        dropout_p: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer 
        # Note: We do NOT register base_layer as a submodule to avoid saving its weights 
        # in the adapter state dict, but we do need it accessible. 
        # However, nn.Module logic will register it if we assign it to self.
        # To avoid saving base weights, we can keep it, but careful with state_dicts.
        # For simplicity in 'remove_hyperlora', we keep it.
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.rho_context = rho_context
        self.rho_encoder = rho_encoder
        self.hyper_net = hyper_net
        
        # Freeze base layer (redundant if done globally, but safe)
        self.base_layer.requires_grad_(False)
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA Matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def compute_gate(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Computes the gate vector g for a given rho, without modifying state.
        Useful for smoothness regularization.
        """
        # 1. Encode
        if self.rho_encoder is not None:
             # Ensure encoder is on correct device
            if self.rho_encoder.frequencies.device != rho.device:
                self.rho_encoder.to(rho.device)
            emb = self.rho_encoder(rho)
        else:
            emb = rho
            
        # 2. HyperNet
        # Ensure hypernet matches needed device
        target_device = self.hyper_net.net[0].weight.device
        if emb.device != target_device:
            emb = emb.to(target_device)
            
        g = self.hyper_net(emb)
        return g

    def forward(self, x: torch.Tensor, rho_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Base Output
        # Base layer handles its own dtype.
        y_base = self.base_layer(x)
        
        # 2. Resolve Rho
        rho = rho_override if rho_override is not None else self.rho_context.current_rho
        
        if rho is None:
            raise RuntimeError("HyperLoRALinear: rho is None. Set model.rho_context.current_rho or pass rho_override.")
            
        # 3. Compute Gate
        # Ensure rho is on x device if possible, but hypernet might be elsewhere.
        # Let compute_gate handle device logic for hypernet.
        if rho.device != x.device:
            rho = rho.to(x.device)
            
        g = self.compute_gate(rho) # [Batch, rank]
        
        # 4. LoRA Path
        # x: [Batch, Seq, In]
        x_lora = x.to(self.lora_A.dtype)
        
        # Move params if needed (robustness)
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)

        z = F.linear(x_lora, self.lora_A) # [Batch, Seq, Rank]
        
        # Gate Broadcasting
        g_casted = g.to(dtype=z.dtype, device=z.device)
        
        if z.dim() == 3:
            z_gated = z * g_casted.unsqueeze(1) # [B, 1, R]
        else:
            z_gated = z * g_casted
            
        delta = F.linear(z_gated, self.lora_B)
        
        output = y_base + (self.scaling * self.dropout(delta)).to(y_base.dtype)
        return output
        
    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

# --- Utilities ---

def model_compute_all_gates(model: nn.Module, rho: torch.Tensor) -> List[torch.Tensor]:
    """
    Computes gates for all HyperLoRA layers in the model for a given rho.
    Returns list of [Batch, Rank] tensors.
    """
    gates = []
    for module in model.modules():
        if isinstance(module, HyperLoRALinear):
            gates.append(module.compute_gate(rho))
    return gates

def inject_hyperlora(
    model: nn.Module, 
    target_module_names: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank: int = 16, 
    alpha: float = 32.0, 
    dropout: float = 0.05, 
    hyper_hidden: int = 64,
    # New Configs
    use_fourier: bool = True,
    fourier_freqs: int = 8,
    include_raw: bool = True,
    fourier_scale: float = 1.0,
    gate_fn: str = "sigmoid",
    gate_clamp_min: float = None,
    gate_clamp_max: float = None
) -> nn.Module:
    
    # 1. Setup Rho Context
    if not hasattr(model, "rho_context"):
        model.rho_context = RhoContext()
    
    # 2. Setup Encoder (Shared or per module? Shared is better for params but needs to be accessible)
    # We will create one encoder instance. If we want it trainable, it should be registered.
    # We'll attach it to the model for saving convenience.
    
    if use_fourier:
        rho_encoder = RhoEncoderFourier(num_frequencies=fourier_freqs, include_raw=include_raw, scale=fourier_scale)
        input_dim = rho_encoder.out_dim
    else:
        rho_encoder = None
        input_dim = 1
        
    # We attach these to model to persist them easily in standard save/load if not careful,
    # but strictly we should manage them in the save/load function.
    model.rho_encoder_module = rho_encoder 
    
    # Store config
    model._hyperlora_config = {
        "adapter_version": "v2",
        "target_module_names": target_module_names,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "hyper_hidden": hyper_hidden,
        "use_fourier": use_fourier,
        "fourier_freqs": fourier_freqs,
        "include_raw": include_raw,
        "fourier_scale": fourier_scale,
        "gate_fn": gate_fn,
        "gate_clamp_min": gate_clamp_min,
        "gate_clamp_max": gate_clamp_max
    }
    model._hyperlora_injected = True
    
    # 3. Injection Loop
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(target) for target in target_module_names):
                modules_to_replace.append(name)
                
    print(f"Injecting HyperLoRA into {len(modules_to_replace)} modules...")
    
    for name in modules_to_replace:
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            child_name = name
            parent = model
            
        target_linear = getattr(parent, child_name)
        
        # Create unique HyperNet per layer (since they learn layer-specific policy mappings)
        hyper_net = RhoHyperNet(
            rank=rank, 
            input_dim=input_dim, 
            hidden_dim=hyper_hidden, 
            gate_fn=gate_fn,
            gate_clamp_min=gate_clamp_min,
            gate_clamp_max=gate_clamp_max
        )
        
        new_module = HyperLoRALinear(
            base_layer=target_linear,
            rank=rank,
            alpha=alpha,
            rho_context=model.rho_context,
            rho_encoder=model.rho_encoder_module, # Shared encoder
            hyper_net=hyper_net,
            dropout_p=dropout
        )
        
        setattr(parent, child_name, new_module)
        
    # 4. Freeze Base, Unfreeze Adapter
    for n, p in model.named_parameters():
        if "lora_" in n or "hyper_net" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    return model

def save_hyperlora_adapter(model: nn.Module, output_dir: str):
    """
    Saves adapter state dict and config.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect state dict
    state_dict = {}
    for n, p in model.named_parameters():
        if "lora_" in n or "hyper_net" in n:
            state_dict[n] = p.cpu()
            
    # Save weights
    torch.save(state_dict, os.path.join(output_dir, "adapter_state.pt"))
    
    # Save config
    if hasattr(model, "_hyperlora_config"):
        config = model._hyperlora_config
    else:
        config = {"version": "unknown"}
        
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Saved HyperLoRA adapter to {output_dir}")

def load_hyperlora_adapter(model: nn.Module, adapter_path: str):
    """
    Loads adapter Config and Weights. Checks version.
    Expects model to NOT be injected yet, or cleanly re-injects.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    weights_path = os.path.join(adapter_path, "adapter_state.pt")
    
    if not os.path.exists(config_path):
        raise ValueError(f"No config found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    version = config.get("adapter_version", "v1")
    
    # Re-inject based on loaded config
    if version == "v2":
        inject_hyperlora(
            model,
            target_module_names=config.get("target_module_names", []),
            rank=config.get("rank", 16),
            alpha=config.get("alpha", 32.0),
            dropout=config.get("dropout", 0.05),
            hyper_hidden=config.get("hyper_hidden", 64),
            use_fourier=config.get("use_fourier", True),
            fourier_freqs=config.get("fourier_freqs", 8),
            include_raw=config.get("include_raw", True),
            fourier_scale=config.get("fourier_scale", 1.0),
            gate_fn=config.get("gate_fn", "sigmoid"),
            gate_clamp_min=config.get("gate_clamp_min", None),
            gate_clamp_max=config.get("gate_clamp_max", None)
        )
    else:
        # Compatibility Mode
        print("Detected v1 or unknown adapter. Attempting compatibility injection...")
        # V1 logic usually had 1-dim input.
        inject_hyperlora(
            model,
            rank=config.get("rank", 16),
            alpha=config.get("alpha", 32.0),
            use_fourier=False, # Force raw
            include_raw=True
        )
        
    # Load Weights
    state_dict = torch.load(weights_path, map_location="cpu")
    # Handling key mismatches?
    # v2 code keys are likely compatible if module names didn't change.
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded adapter weights. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")

def remove_hyperlora(model: nn.Module):
    """
    Restores original Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, HyperLoRALinear):
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                child_name = name
                parent = model
                
            # Restore base
            # Note: base_layer is froze, we should unfreeze?
            # User might want to continue training base.
            module.base_layer.requires_grad_(True) 
            setattr(parent, child_name, module.base_layer)
            
    if hasattr(model, "rho_context"):
        del model.rho_context
    if hasattr(model, "_hyperlora_injected"):
        del model._hyperlora_injected
    print("HyperLoRA modules removed.")

def list_injected_modules(model: nn.Module) -> List[Dict]:
    info = []
    for name, module in model.named_modules():
        if isinstance(module, HyperLoRALinear):
            info.append({
                "name": name,
                "in_features": module.base_layer.in_features,
                "out_features": module.base_layer.out_features,
                "rank": module.rank
            })
    return info

