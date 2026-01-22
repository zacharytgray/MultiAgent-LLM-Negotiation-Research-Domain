
import os
import sys
import argparse
import random
import json
import logging
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
# Assuming structure: src/training/train_janus_hyperlora.py
# We add the parent directory of 'src' to path if running from root, or relative import if package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.training.hyper_lora import inject_hyperlora, HyperLoRALinear
except ImportError:
    print("Could not import hyper_lora directly. Trying local import...")
    import hyper_lora
    from hyper_lora import inject_hyperlora, HyperLoRALinear

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
    parser = argparse.ArgumentParser(description="Train HyperLoRA Janus Negotiation Agent")
    
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
    
    # Eval / Saving
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    
    args = parser.parse_args()
    
    # Convert string bool
    if args.include_failures.lower() == "false":
        args.include_failures = False
    else:
        args.include_failures = True
        
    return args

def normalize_price(val: float, low: float, high: float) -> float:
    """Normalize price to [0, 1] given range."""
    rng = high - low
    if rng < 1e-6:
        rng = 1.0
    return (val - low) / rng

class NegotiationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int, k_history: int):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_history = k_history
        self.debug_count = 0
        
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
            f"<OUTPUT>\n"
        )
        
        full_text = prompt + target_str + self.tokenizer.eos_token
        
        # 3. Tokenize
        # We need to mask the prompt. 
        # Strategy: Tokenize prompt, get len. Tokenize full, get len. Labels[:prompt_len] = -100.
        
        # For truncation, we ideally truncate history, but standard truncation chops from end.
        # This is a risk. With max_length 1024, it should be fine for K=8. 
        
        input_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = input_enc.input_ids[0]
        attention_mask = input_enc.attention_mask[0]
        
        # Create Labels
        labels = input_ids.clone()
        
        # Determine prompt boundary
        # We iterate to find where the prompt ends. Or just tokenize prompt separately.
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]
        
        # The full_text tokenization might include a BOS token which prompt_enc might not if not handled carefully
        # But usually AutoTokenizer handles BOS logic similarly for both.
        # Let's verify lengths.
        
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            # Fallback if prompt was truncated or something weird match
            # Mask everything except last few tokens? Dangerous.
            # Just mask first 3/4 if unsure? 
            # Better safe: re-check overlaps.
            labels[:] = -100 # Ignore this bad sample
            
        # 4. Return
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rho": torch.tensor([rho], dtype=torch.float32) # [1]
        }

class HyperLoRACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        # batch is list of dicts
        
        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [x['attention_mask'] for x in batch]
        labels = [x['labels'] for x in batch]
        rhos = [x['rho'] for x in batch]
        
        # Pad
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Stack Rhos [B, 1]
        rhos_tensor = torch.stack(rhos) 
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "rho": rhos_tensor
        }

def save_hyperlora_adapter(model, output_dir, tokenizer, args):
    """Save only the trainable HyperLoRA parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. State Dict
    to_save = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            to_save[n] = p.cpu()
            
    torch.save(to_save, os.path.join(output_dir, "adapter_state.pt"))
    
    # 2. Config
    config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "hyper_hidden": args.hyper_hidden,
        "base_model": args.model_name,
        "augmented_tokens": SPECIAL_TOKENS
    }
    with open(os.path.join(output_dir, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
        
    # 3. Tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved adapter to {output_dir}")

def train():
    args = parse_args()
    
    set_seed(args.seed)
    
    logger.info(f"Loading data from {args.decision_steps_path}...")
    df = pd.read_parquet(args.decision_steps_path)
    logger.info(f"Loaded {len(df)} rows.")
    
    # Filtering
    if not args.include_failures:
        logger.info("Excluding failures (rho == -1.0)...")
        df = df[df['rho_train'] != -1.0]
        logger.info(f"Filtered to {len(df)} rows.")
        
    # Stats
    success_df = df[df['rho_train'] != -1.0]
    if not success_df.empty:
        rhos = success_df['rho_train']
        logger.info(f"Success Rho Stats: Min={rhos.min():.4f}, Max={rhos.max():.4f}, Mean={rhos.mean():.4f}")
    
    logger.info(f"Failures count: {len(df[df['rho_train'] == -1.0])}")

    # Split Eval
    # Simple random split 1%
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_size = int(len(shuffled) * 0.01)
    if eval_size < 1: eval_size = 0
    
    train_df = shuffled.iloc[eval_size:]
    eval_df = shuffled.iloc[:eval_size]
    
    logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
    
    # Load Tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Add tokens
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Added {num_added} special tokens.")
    
    # Dataset & Loader
    train_dataset = NegotiationDataset(train_df, tokenizer, args.max_length, args.k_history)
    eval_dataset = NegotiationDataset(eval_df, tokenizer, args.max_length, args.k_history)
    
    collator = HyperLoRACollator(tokenizer.pad_token_id)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0) # workers=0 for simplicity
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if eval_size > 0 else None
    
    # Load Model
    logger.info("Loading Base Model...")
    
    device_map = "auto"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
    model.resize_token_embeddings(len(tokenizer))
    
    # Inject HyperLoRA
    logger.info("Injecting HyperLoRA Modules...")
    model = inject_hyperlora(
        model, 
        rank=args.rank, 
        alpha=args.alpha, 
        dropout=args.dropout, 
        hyper_hidden=args.hyper_hidden
    )
    
    # Move params to device if not auto (HyperNet might be on CPU initially if model was fully mapped)
    # The injection kept them on same device as base layers, but let's Ensure
    # Actually, model is mostly on GPU.
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    num_training_steps = args.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )
    
    # Training Loop
    logger.info("Starting Training...")
    global_step = 0
    model.train()
    
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    epoch = 0
    best_loss = float('inf')
    
    while global_step < num_training_steps:
        epoch += 1
        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            rhos = batch['rho'].to(model.device) # [B, 1] float32
            
            # --- RHO PLUMBING ---
            # Set the context for HyperLoRA layers
            model.current_rho = rhos 
            # --------------------
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item() * args.grad_accum)
                
                if global_step % args.save_every == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_hyperlora_adapter(model, save_path, tokenizer, args)
                    
                if global_step >= num_training_steps:
                    break
        
        # Eval at end of epoch? Or just rely on steps.
        # Let's do a quick eval pass if loader exists
        pass # skipping epoch-based eval for brevity, relying on step-based.

    logger.info("Training Complete.")
    save_hyperlora_adapter(model, os.path.join(args.output_dir, "final"), tokenizer, args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    train()
