import os
import sys
import argparse
import random
import json
import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, 
    BitsAndBytesConfig
)
from tqdm import tqdm

# Ensure we can import from the sibling module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.training.hyper_lora import (
        inject_hyperlora, 
        HyperLoRALinear, 
        save_hyperlora_adapter, 
        model_compute_all_gates
    )
except ImportError:
    print("Could not import hyper_lora directly. Trying local import...")
    import hyper_lora
    from hyper_lora import (
        inject_hyperlora, 
        HyperLoRALinear, 
        save_hyperlora_adapter,
        model_compute_all_gates
    )

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
SPECIAL_TOKENS = [
    "<RHO>", "<ROLE>", "<TURN>", "<TURNS_REMAINING>", 
    "<RESERVATION_NORM>", "<LAST_OFFER_NORM>", "<HISTORY>", 
    "<INSTRUCTION>", "<OUTPUT>", "<RHO_FAIL>"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperLoRA Janus Negotiation Agent (v2)")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B", help="HF Model ID")
    parser.add_argument("--decision_steps_path", type=str, required=True, help="Path to decision_steps.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--k_history", type=int, default=8, help="Number of history items to show")
    parser.add_argument("--include_failures", type=str, default="true", help="Include rho=-1.0 trajectories? (true/false)")
    
    # Training Hyperparams
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit loading (QLoRA)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    
    # HyperLoRA Params
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hyper_hidden", type=int, default=64)
    
    # v2 Upgrades: Fourier Rho Encoder
    parser.add_argument("--rho_encoder", type=str, default="fourier", choices=["fourier", "raw"], help="Type of rho encoding")
    parser.add_argument("--rho_num_frequencies", type=int, default=8, help="Number of Fourier frequencies")
    parser.add_argument("--rho_include_raw", type=str, default="true", help="Include raw scalar in embedding?")
    parser.add_argument("--rho_scale", type=float, default=1.0, help="Scaling factor for Fourier frequencies")
    
    # v2 Upgrades: Gating
    parser.add_argument("--gate_fn", type=str, default="sigmoid", choices=["sigmoid", "tanh", "tanh01", "softplus", "identity"])
    parser.add_argument("--gate_clamp_min", type=float, default=None)
    parser.add_argument("--gate_clamp_max", type=float, default=None)
    
    # v2 Upgrades: Smoothness Regularization
    parser.add_argument("--lambda_smooth", type=float, default=0.0, help="Coefficient for smoothness regularization")
    parser.add_argument("--smooth_eps", type=float, default=0.02, help="Epsilon for numeric derivative approximation")
    parser.add_argument("--smooth_clamp_min", type=float, default=-1.0, help="Min bound for rho perturbation")
    parser.add_argument("--smooth_clamp_max", type=float, default=1.0, help="Max bound for rho perturbation")

    # Eval
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_num_samples", type=int, default=128, help="Number of samples to eval generation on")
    parser.add_argument("--eval_generate", action="store_true", help="Enable generation-based eval (slow)")
    
    args = parser.parse_args()
    
    # Boolean conversions
    args.include_failures = (args.include_failures.lower() == "true")
    args.rho_include_raw = (args.rho_include_raw.lower() == "true")
    
    return args

def normalize_price(val: float, low: float, high: float) -> float:
    """Normalize price to [0, 1] given range. Robust to zero range."""
    rng = high - low
    if rng < 1e-6:
        # If range is zero, return 0. (Or 0.5? 0 implies low, which is also high)
        return 0.0
    return (val - low) / rng

class NegotiationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int, k_history: int):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_history = k_history
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Values extraction
        rho = float(row['rho_train'])
        role = str(row['role'])
        turn = int(row['turn'])
        max_turns = int(row['max_turns'])
        turns_rem = int(row['turns_remaining'])
        
        p_low = float(row['price_low']) if not pd.isna(row['price_low']) else 0.0
        p_high = float(row['price_high']) if not pd.isna(row['price_high']) else 2000.0
        
        # Use simple validation for p_low/p_high
        if p_high < p_low: p_high = p_low + 1.0 # fix inversion
        
        # Norms
        res_price = float(row['reservation_price']) if not pd.isna(row['reservation_price']) else p_low
        res_norm = normalize_price(res_price, p_low, p_high)
        
        last_offer_norm = "NA"
        if not pd.isna(row['last_offer_price']):
            lo_val = float(row['last_offer_price'])
            last_offer_norm = f"{normalize_price(lo_val, p_low, p_high):.4f}"
            
        # History
        hist_str = "EMPTY"
        h_roles = row['history_roles']
        h_prices = row['history_prices']
        
        # Ensure they are lists
        if hasattr(h_roles, 'tolist'): h_roles = h_roles.tolist()
        if hasattr(h_prices, 'tolist'): h_prices = h_prices.tolist()
        
        if h_roles and len(h_roles) > 0:
            # Take last K
            start_k = max(0, len(h_roles) - self.k_history)
            pairs = []
            for r, p in zip(h_roles[start_k:], h_prices[start_k:]):
                p_norm = normalize_price(p, p_low, p_high)
                pairs.append(f"{r}:{p_norm:.4f}")
            hist_str = " ".join(pairs)
            
        # Target
        t_action = row['target_action']
        target_str = ""
        if t_action == "ACCEPT":
            target_str = "ACCEPT"
        else:
            t_price = float(row['target_price']) if not pd.isna(row['target_price']) else 0.0
            t_p_norm = normalize_price(t_price, p_low, p_high)
            target_str = f"OFFER {t_p_norm:.4f}"
            
        # 2. PROMPT Construction
        # Rho Token: If failure (-1.0), use <RHO_FAIL>. Else value.
        if abs(rho - (-1.0)) < 1e-6:
            rho_token_str = "<RHO_FAIL>"
        else:
            rho_token_str = f"{rho:.2f}"
            
        prompt = (
            f"<RHO> {rho_token_str}\n"
            f"<ROLE> {role}\n"
            f"<TURN> {turn} / {max_turns}\n"
            f"<TURNS_REMAINING> {turns_rem}\n"
            f"<RESERVATION_NORM> {res_norm:.4f}\n"
            f"<LAST_OFFER_NORM> {last_offer_norm}\n"
            f"<HISTORY> {hist_str}\n"
            f"<INSTRUCTION> Output exactly one of:\n"
            f"ACCEPT\n"
            f"OFFER <PRICE_NORM>\n"
            f"<OUTPUT> " # Include trailing space?
        )
        
        # Add a trailing newline to separate strictly
        prompt = prompt.strip() + "\n"
        
        full_text = prompt + target_str + self.tokenizer.eos_token
        
        # 3. Tokenize
        input_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = input_enc.input_ids[0]
        attention_mask = input_enc.attention_mask[0]
        
        labels = input_ids.clone()
        
        # Determine prompt boundary to mask labels
        # Tokenize prompt alone. Add BOS if model uses it by default.
        prompt_enc = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]
        
        # If prompt_enc has one less token (maybe EOS?), just be careful. 
        # Usually prompt_len covers the prompt.
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            # If prompt truncates, the whole sample is prompt. Mask all.
            labels[:] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rho": torch.tensor([rho], dtype=torch.float32),
            "prompt_text": prompt, # For eval generation
            "target_text": target_str
        }

class HyperLoRACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [x['attention_mask'] for x in batch]
        labels = [x['labels'] for x in batch]
        rhos = [x['rho'] for x in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        rhos_tensor = torch.stack(rhos) 
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "rho": rhos_tensor,
            "raw_batch": batch # pass through for eval text access
        }

def run_restricted_eval(model, tokenizer, loader, num_samples=32, device="cuda"):
    """
    Runs generation on a subset of data and computes strict accuracy/MSE.
    """
    model.eval()
    samples_count = 0
    correct_accept = 0
    offer_mse_sum = 0.0
    offer_count = 0
    total_correct_action = 0
    
    print(f"Running Eval Generation on {num_samples} samples...")
    
    with torch.no_grad():
        for batch in loader:
            if samples_count >= num_samples: break
            
            # Use raw prompts
            raw_items = batch["raw_batch"]
            curr_bs = len(raw_items)
            
            # Prepare inputs for generation
            prompts = [item["prompt_text"] for item in raw_items]
            targets = [item["target_text"] for item in raw_items]
            rhos = batch["rho"].to(device)
            
            # Tokenize prompts only
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Set context
            model.rho_context.current_rho = rhos
            
            # Generate
            gen_out = model.generate(
                **inputs, 
                max_new_tokens=16, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode
            # Slice off input
            input_len = inputs.input_ids.shape[1]
            generated_tokens = gen_out[:, input_len:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for pred_str, true_str in zip(decoded, targets):
                samples_count += 1
                pred_clean = pred_str.strip()
                true_clean = true_str.strip()
                
                # Check Action
                pred_action = "ACCEPT" if "ACCEPT" in pred_clean else "OFFER"
                true_action = "ACCEPT" if "ACCEPT" in true_clean else "OFFER"
                
                if pred_action == true_action:
                    total_correct_action += 1
                    
                # Check Price Logic
                if true_action == "OFFER" and pred_action == "OFFER":
                    # Extract floats
                    try:
                        # Regex for float
                        p_vals = re.findall(r"[-+]?\d*\.\d+|\d+", pred_clean)
                        t_vals = re.findall(r"[-+]?\d*\.\d+|\d+", true_clean)
                        if p_vals and t_vals:
                            p_val = float(p_vals[0])
                            t_val = float(t_vals[0])
                            offer_mse_sum += (p_val - t_val) ** 2
                            offer_count += 1
                    except:
                        pass
                
            if samples_count >= num_samples: break
            
    acc = total_correct_action / max(1, samples_count)
    mse = offer_mse_sum / max(1, offer_count)
    
    print(f"Eval Result: Action Acc={acc:.4f} | Offer MSE={mse:.6f} | Samples={samples_count}")
    return {"acc": acc, "mse": mse}

def train():
    args = parse_args()
    
    # Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load Data
    logger.info(f"Loading data from {args.decision_steps_path}...")
    df = pd.read_parquet(args.decision_steps_path)
    logger.info(f"Loaded {len(df)} rows.")
    
    if not args.include_failures:
        logger.info("Excluding failures (rho == -1.0)...")
        df = df[df['rho_train'] != -1.0]
    
    # Simple Split
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_len = int(len(shuffled) * 0.02)
    eval_len = max(eval_len, min(len(shuffled), 32)) # Ensure at least some eval if possible
    
    train_df = shuffled.iloc[eval_len:]
    eval_df = shuffled.iloc[:eval_len]
    
    logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Dataset
    train_dataset = NegotiationDataset(train_df, tokenizer, args.max_length, args.k_history)
    eval_dataset = NegotiationDataset(eval_df, tokenizer, args.max_length, args.k_history)
    
    collator = HyperLoRACollator(tokenizer.pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader_gen = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    # Model
    logger.info("Loading Base Model...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using dtype: {torch_dtype}")

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype, trust_remote_code=True)
        
    model.resize_token_embeddings(len(tokenizer))
    
    # Inject
    logger.info("Injecting HyperLoRA V2...")
    model = inject_hyperlora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        hyper_hidden=args.hyper_hidden,
        use_fourier=(args.rho_encoder == "fourier"),
        fourier_freqs=args.rho_num_frequencies,
        include_raw=args.rho_include_raw,
        fourier_scale=args.rho_scale,
        gate_fn=args.gate_fn,
        gate_clamp_min=args.gate_clamp_min,
        gate_clamp_max=args.gate_clamp_max
    )
    
    if not args.use_qlora:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    
    global_step = 0
    model.train()
    
    # Training Loop
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    max_steps_reached = False
    while not max_steps_reached:
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            rhos = batch['rho'].to(model.device)
            
            # 1. Update Rho Context
            model.rho_context.current_rho = rhos
            
            # 2. Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs.loss
            
            # 3. Smoothness Regularization
            loss = lm_loss
            if args.lambda_smooth > 0:
                # Mask out negative rhos (failures) from smoothness penalty?
                # Usually rho=-1.0 is failure.
                # Valid mask: rho >= -0.1 (approx) or rho != -1.0
                valid_mask = (rhos.squeeze() > -0.5) 
                
                if valid_mask.any():
                    # Create Perturbed Rho
                    # Clamp to ensure we stay in [-1, 1] or [0, 1] depending on domain
                    eps = args.smooth_eps
                    rhos_p = rhos.clone()
                    rhos_p[valid_mask] += eps
                    rhos_p = torch.clamp(rhos_p, args.smooth_clamp_min, args.smooth_clamp_max)
                    
                    # Compute Gates
                    # We need to catch these so we don't build huge graphs if not needed
                    # but we do need grads for HyperNet.
                    
                    # Gates 1: Current Rho
                    gates_1 = model_compute_all_gates(model, rhos)
                    # Gates 2: Perturbed Rho
                    gates_2 = model_compute_all_gates(model, rhos_p)
                    
                    # Flatten and concat -> [B, Total_Rank]
                    g1_flat = torch.cat([g.reshape(g.shape[0], -1) for g in gates_1], dim=1)
                    g2_flat = torch.cat([g.reshape(g.shape[0], -1) for g in gates_2], dim=1)
                    
                    # Filter by valid mask
                    # MSE
                    diff = (g1_flat[valid_mask] - g2_flat[valid_mask])
                    smooth_loss = (diff ** 2).mean()
                    
                    loss = loss + args.lambda_smooth * smooth_loss
            
            loss = loss / args.grad_accum
            loss.backward()
            
            if (global_step + 1) % args.grad_accum == 0:
                # Clip?
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=loss.item() * args.grad_accum)
                
                # Checkpointing
                if global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_hyperlora_adapter(model, save_path)
                    tokenizer.save_pretrained(save_path)
                    
                # Eval
                if global_step % args.eval_every == 0 and args.eval_generate:
                    run_restricted_eval(model, tokenizer, eval_loader_gen, num_samples=args.eval_num_samples, device=model.device)
                    model.train() 

                if global_step >= args.max_steps:
                    max_steps_reached = True
                    break
        
        if max_steps_reached: break
        
    logger.info("Training Complete.")
    final_output = os.path.join(args.output_dir, "final")
    save_hyperlora_adapter(model, final_output)
    tokenizer.save_pretrained(final_output)

if __name__ == "__main__":
    train()
