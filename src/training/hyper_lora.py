
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Optional, Dict, Union

class RhoHyperNet(nn.Module):
    """
    Hypernetwork that maps a scalar rho to a gating vector g of size [rank].
    
    Structure:
      rho [B, 1] -> Linear(1, hidden) -> SiLU -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, rank) -> Sigmoid
    """
    def __init__(self, rank: int, hidden_dim: int = 64, activation: str = 'sigmoid'):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim
        
        # MLP Layers
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank)
        )
        
        if activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif activation == 'tanh':
            self.final_act = nn.Tanh()
        elif activation == 'softplus':
            self.final_act = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho: Tensor of shape [Batch_Size, 1] (float32)
        
        Returns:
            g: Tensor of shape [Batch_Size, rank] (same dtype as rho usually, or cast later)
        """
        # Ensure input is float32 for stability in the hypernet, usually
        x = rho.to(self.net[0].weight.dtype) 
        output = self.net(x)
        return self.final_act(output)

class HyperLoRALinear(nn.Module):
    """
    Wraps a base nn.Linear layer with HyperLoRA adaptation.
    
    W = W_base + (alpha/rank) * ( B @ diag(g(rho)) @ A )
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int, 
        alpha: float, 
        rho_getter: Callable[[], torch.Tensor],
        hyper_hidden: int = 64,
        dropout_p: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.rho_getter = rho_getter
        
        # Freeze base layer
        self.base_layer.requires_grad_(False)
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA Matrices
        # A: [rank, in_dim]
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # B: [out_dim, rank]
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        # HyperNetwork (Gating)
        self.hyper_net = RhoHyperNet(rank, hidden_dim=hyper_hidden)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Initialize
        self.reset_parameters()
        
    def reset_parameters(self):
        # A: Kaiming Uniform
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # B: Zeros
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with injected HyperLoRA logic.
        """
        # 1. Base Output
        # Base layer handles its own dtype. 
        y_base = self.base_layer(x)
        
        # 2. Get Rho (Batch Context)
        # Expected shape [Batch, 1]
        rho = self.rho_getter()
        
        # If rho is None or not set, fallback to base only (or raise error)
        if rho is None:
            return y_base
            
        # Ensure rho is physically on the same device
        if rho.device != x.device:
            rho = rho.to(x.device)
            
        # 3. Compute Gating Vector g(rho)
        # HyperNet typically runs in float32 or same as parameters.
        # Force rho to match hypernet params dtype
        g = self.hyper_net(rho) # [Batch, rank]
        
        # 4. LoRA Path
        # x: [Batch, Seq_Len, In] or [Batch, In]
        input_dtype = x.dtype
        
        # Cast input to LoRA weights dtype if needed
        x_lora = x.to(self.lora_A.dtype)
        
        # z = x @ A.T -> [Batch, ..., rank]
        z = F.linear(x_lora, self.lora_A) 
        
        # Apply Gating
        # z is [Batch, Seq, Rank] or [Batch, Rank]
        # g is [Batch, Rank]
        # We need to broadcast g. If z is 3D, unsqueeze g at dim 1.
        
        # Cast g to match z (bf16 likely)
        g_casted = g.to(z.dtype)
        
        if z.dim() == 3:
            # [Batch, Seq, Rank] * [Batch, 1, Rank]
            z_gated = z * g_casted.unsqueeze(1)
        else:
            # [Batch, Rank] * [Batch, Rank]
            z_gated = z * g_casted
            
        # delta = z_gated @ B.T -> [Batch, ..., Out]
        delta = F.linear(z_gated, self.lora_B)
        
        # 5. Combine
        # cast delta back to output dtype of base (y0)
        output = y_base + (self.scaling * self.dropout(delta)).to(y_base.dtype)
        
        return output

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"


def inject_hyperlora(
    model: nn.Module, 
    target_module_names: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rank: int = 16, 
    alpha: float = 32.0, 
    dropout: float = 0.05, 
    hyper_hidden: int = 64
):
    """
    Injects HyperLoRALinear layers into the model in-place.
    
    Args:
        model: Example: Qwen2ForCausalLM
        target_module_names: Suffixes of modules to replace (e.g. "q_proj")
        rank, alpha, dropout: LoRA params
    """
    
    # 1. Setup Global Rho Context on Model
    # We attach an attribute to the model to store the current rho batch.
    # Initialized to None.
    if not hasattr(model, "current_rho"):
        model.current_rho = None
        
    # Create the getter closure. 
    # Note: We bind 'model' from this scope.
    rho_getter = lambda: model.current_rho
    
    # 2. Traverse and Replace
    # We collect targets first to avoid modifying OrderedDict while iterating
    modules_to_replace = []
    
    for name, module in model.named_modules():
        # check if it matches target suffixes and is a Linear layer
        if isinstance(module, nn.Linear):
            # Check suffix
            # e.g. "model.layers.0.self_attn.q_proj" ends with "q_proj"
            if any(name.endswith(target) for target in target_module_names):
                modules_to_replace.append(name)
                
    print(f"Found {len(modules_to_replace)} modules to replace with HyperLoRA.")
    
    for name in modules_to_replace:
        # Get the module and its parent
        parent_name, child_name = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        target_linear = getattr(parent, child_name)
        
        # Create Wrapper
        new_module = HyperLoRALinear(
            base_layer=target_linear,
            rank=rank,
            alpha=alpha,
            rho_getter=rho_getter,
            hyper_hidden=hyper_hidden,
            dropout_p=dropout
        )
        
        # Replace
        setattr(parent, child_name, new_module)
        
    # 3. Freeze & Verify
    # Freeze all parameters that are NOT LoRA or HyperNet
    for n, p in model.named_parameters():
        if "lora_" in n or "hyper_net" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    return model

def count_trainable_params(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return trainable_params

# --- Minimal Test Script ---

if __name__ == "__main__":
    import sys
    
    print("--- Running HyperLoRA Sanity Check ---")
    
    # Try loading a small model or creating a dummy config
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Create a dummy Qwen2 config to avoid needing actual weights for this test
        # Qwen2-7B config approx:
        config = AutoConfig.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        # Use tiny config for speed
        config.num_hidden_layers = 2
        config.hidden_size = 64
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        
        print("Creating dummy Qwen2 model...")
        model = AutoModelForCausalLM.from_config(config)
        
    except Exception as e:
        print(f"Could not load Transformers/QwenConfig: {e}")
        # Construct a simple barebones nn.Module to simulate structure
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.v_proj = nn.Linear(32, 32)
                self.fc = nn.Linear(32, 10) # Should NOT be replaced
        model = DummyModel()
        print("Created simple dummy model.")
        
    # Inject
    print("Injecting HyperLoRA...")
    inject_hyperlora(
        model, 
        target_module_names=["q_proj", "output_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], 
        rank=8, 
        alpha=16
    )
    
    count_trainable_params(model)
    
    # Test Forward with Rho
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    model.to(device)
    
    batch_size = 2
    seq_len = 5
    hidden_dim = getattr(model.config, "hidden_size", 32)
    
    # Dummy Inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Set Rho
    # Shape [B, 1]
    rho_val = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float32)
    model.current_rho = rho_val
    
    # Forward 1
    print("Running Forward Pass 1 (rho=[0, 1])...")
    with torch.no_grad():
        res1 = model(input_ids)
        out1 = res1.logits if hasattr(res1, 'logits') else res1
        
    # Perturb LoRA A/B manually to ensure they do something (since B init is zero)
    # We want to trace if rho changes result.
    # Currently B=0, so LoRA contribution is 0. Changing rho won't change output yet.
    # Let's set some random weights in LoRA B
    print("Perturbing LoRA B weights to verify gating...")
    for n, m in model.named_modules():
        if isinstance(m, HyperLoRALinear):
            nn.init.normal_(m.lora_B, std=0.1)
            
    # Forward 2 (Same rho)
    print("Running Forward Pass 2 (Post-Perturb, rho=[0, 1])...")
    with torch.no_grad():
        res2 = model(input_ids)
        out2 = res2.logits if hasattr(res2, 'logits') else res2
        
    # Forward 3 (Different rho)
    model.current_rho = torch.tensor([[-1.0], [0.5]], device=device, dtype=torch.float32)
    print("Running Forward Pass 3 (rho=[-1, 0.5])...")
    with torch.no_grad():
        res3 = model(input_ids)
        out3 = res3.logits if hasattr(res3, 'logits') else res3
        
    # Checks
    print("\n--- Results ---")
    
    # 1. Did B=0 vs B!=0 change anything? (Yes it should)
    diff_init = (out2 - out1).abs().mean().item()
    print(f"Difference after initializing B (should be > 0): {diff_init:.6f}")
    
    # 2. Does changing Rho change output?
    diff_rho = (out3 - out2).abs().mean().item()
    print(f"Difference after changing Rho (should be > 0): {diff_rho:.6f}")
    
    print("HyperLoRA Module Test Complete.")

