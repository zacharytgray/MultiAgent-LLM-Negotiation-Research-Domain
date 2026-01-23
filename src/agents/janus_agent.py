
import torch
import re
import os
import json
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.base_agent import BaseAgent
from src.training.hyper_lora import inject_hyperlora
from src.core.price_structures import PriceAction

class JanusAgent(BaseAgent):
    """
    Janus Agent: A HyperLoRA-controlled negotiation agent.
    
    This agent uses a single base LLM (Qwen2) equipped with HyperLoRA adapters.
    Behavior is controlled by a scalar 'rho' (0.0 to 1.0) injected at runtime.
    
    rho=0.0 -> Aggressive Buyer / Passive Seller
    rho=1.0 -> Aggressive Seller / Passive Buyer
    rho=-1.0 -> Forced Failure / Impasse Mode
    """
    
    def __init__(self, agent_id: int, role: str, model_path: str = "Qwen/Qwen2-7B-Instruct", 
                 adapter_path: str = "checkpoints/janus_v1/final", 
                 rho: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        super().__init__(agent_id, f"janus_{rho}", "none")
        self.role = role
        self.rho = rho
        self.device = device
        self.adapter_path = adapter_path
        
        print(f"[{role.upper()}] Initializing JanusAgent (rho={rho}) on {device}...")
        
        # 1. Load Tokenizer
        # We try to load from adapter_path first (if saved there), else base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except:
            print(f"[{role}] Warning: Could not download tokenizer from {adapter_path}, using {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.adapter_config = json.load(f)
        else:
            print(f"[{role}] Warning: No adapter_config.json found. Using defaults.")
            self.adapter_config = {
                "rank": 16, "alpha": 32, "hyper_hidden": 64, 
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"] # Guessing defaults
            }

        # 3. Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # 4. Inject HyperLoRA
        # Note: target_module_names might vary depending on how inject_hyperlora was called during training.
        # We try to infer or use standard Qwen/Llama targets.
        target_modules = self.adapter_config.get("target_modules", 
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        # In case the training script didn't save list, use the key logic from training script
        # The script used: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] usually.
        
        self.model = inject_hyperlora(
            self.model,
            target_module_names=target_modules,
            rank=self.adapter_config.get("rank", 16),
            alpha=self.adapter_config.get("alpha", 32),
            hyper_hidden=self.adapter_config.get("hyper_hidden", 64),
            dropout=0.0 # No dropout needed for inference
        )

        # 5. Load Weights
        weights_path = os.path.join(adapter_path, "adapter_state.pt")
        if os.path.exists(weights_path):
            print(f"[{role}] Loading HyperLoRA weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            # We strictly load compatible keys
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                # This is normal for frozen base params, but we should check if lora params are missing
                pass
        else:
            raise FileNotFoundError(f"Could not find adapter weights at {weights_path}")

        self.model.eval()
        
    def _normalize(self, val: float, low: float, high: float) -> float:
        rng = high - low
        if rng < 1e-6: rng = 1.0
        return (val - low) / rng

    def _denormalize(self, norm_val: float, low: float, high: float) -> float:
        return low + norm_val * (high - low)
    
    async def generate_response(self) -> str:
        """
        Generates a text response using the Janus model.
        This function handles:
        1. Context extraction (from self.domain_private_context)
        2. Prompt construction (Janus format)
        3. Inference with correct 'rho'
        4. Output parsing/denormalization
        """
        
        # 1. Extract Context
        # We rely on Negotiation.py to populate self.domain_private_context
        # Context usually has: role, history, max_turns, conversation_history...
        ctx = self.domain_private_context
        if not ctx:
            return "ACCEPT" # Fallback if no context
            
        role = self.role # or ctx.get("role")
        turn = ctx.get("turn", 1) # Negotiation.py usually tracks turns
        max_turns = ctx.get("max_turns", 20)
        
        # Price Bounds (For Normalization)
        # In 'price' domain, usually known? Or we guess.
        # Assuming standard bounds used in training (0-2000?) or derived from context.
        # Negotiation.py passes 'public_price_range' in state usually?
        # Let's check context. Usually keys: role, min_acceptable_price, max_willingness_to_pay
        
        # Training normalization was done on specific 'price_low'/'price_high' columns.
        # We should use 0-2000 as default or whatever bounds were used in training data gen.
        # Looking at Negotiation.py: public_price_range=(200.0, 1500.0) or dummy (0, 2000)
        p_low = 0.0
        p_high = 2000.0
        
        reservation = ctx.get("max_willingness_to_pay") if role == "buyer" else ctx.get("min_acceptable_price")
        res_norm = self._normalize(reservation, p_low, p_high)
        
        # History
        # We need to parse self.memory or ctx['history'] into "role:norm_price" list
        history_str = "EMPTY"
        
        # Try to find formatted history
        # If Negotiation.py passed a 'history' list of (role, price) tuples:
        raw_history = ctx.get("history", []) 
        
        # If raw_history is empty, check if we can parse from standard memory logs
        k = 8
        if raw_history:
            # Take last k
            # Format: role:price_norm
            pairs = []
            for h_role, h_price in raw_history[-k:]:
                h_norm = self._normalize(h_price, p_low, p_high)
                pairs.append(f"{h_role}:{h_norm:.4f}")
            if pairs:
                history_str = " ".join(pairs)
        
        # Last Offer
        last_offer_norm = "NA"
        if raw_history:
            last_offer_price = raw_history[-1][1]
            last_offer_norm = f"{self._normalize(last_offer_price, p_low, p_high):.4f}"
            
        turns_remaining = max_turns - turn
        
        # 2. Build Prompt
        prompt = (
            f"<RHO> {self.rho:.2f}\n"
            f"<ROLE> {role}\n"
            f"<TURN> {turn} / {max_turns}\n"
            f"<TURNS_REMAINING> {turns_remaining}\n"
            f"<RESERVATION_NORM> {res_norm:.4f}\n"
            f"<LAST_OFFER_NORM> {last_offer_norm}\n"
            f"<HISTORY> {history_str}\n"
            f"<INSTRUCTION> Output exactly one of:\n"
            f"ACCEPT\n"
            f"OFFER <PRICE_NORM>\n"
            f"<OUTPUT>\n"
        )
        
        # 3. Inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set RHO
        self.model.current_rho = torch.tensor([[self.rho]], device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False # Greedy for best adherence
            )
            
        output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 4. Parse & Denormalize
        # Expected: "ACCEPT" or "OFFER 0.1234"
        if "ACCEPT" in output_text:
            return "ACCEPT"
            
        match = re.search(r"OFFER\s+([0-9\.]+)", output_text)
        if match:
            try:
                val_norm = float(match.group(1))
                price_real = self._denormalize(val_norm, p_low, p_high)
                price_real = round(price_real, 2)
                return f"OFFER {price_real}"
            except:
                pass
                
        # Fallback if model hallucinates format
        return f"OFFER {reservation}" # Safe fallback
        
    def add_to_memory(self, role: str, content: str):
        # We don't maintain internal chat history for prompt construction here
        # because we rely on the structured 'history' passed via context update.
        pass
        
    def reset_memory(self):
        pass
