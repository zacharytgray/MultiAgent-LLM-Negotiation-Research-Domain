
import torch
import os
import sys
import numpy as np

def analyze_checkpoint(checkpoint_path):
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    try:
        # Load map_location='cpu' to avoid cuda errors if just analyzing
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"Loaded {len(state_dict)} keys.")
    
    lora_a_stats = []
    lora_b_stats = []
    hyper_net_stats = []
    
    print("\n--- detailed inspection ---")
    
    # Check a few specific keys to see structure
    keys = list(state_dict.keys())
    print("Sample keys:", keys[:5])
    
    for key, tensor in state_dict.items():
        tensor = tensor.float() # Ensure float for stats
        mean = tensor.mean().item()
        std = tensor.std().item()
        abs_max = tensor.abs().max().item()
        norm = tensor.norm().item()
        
        stat = {
            "key": key,
            "mean": mean,
            "std": std,
            "abs_max": abs_max,
            "norm": norm,
            "shape": tuple(tensor.shape)
        }
        
        if "lora_A" in key:
            lora_a_stats.append(stat)
        elif "lora_B" in key:
            lora_b_stats.append(stat)
        elif "hyper_net" in key:
            hyper_net_stats.append(stat)
            
    # Report LoRA A
    print("\n--- LoRA A (Expect Kaiming Init / Non-Zero) ---")
    if lora_a_stats:
        # Just show aggregate or first few
        means = [s['mean'] for s in lora_a_stats]
        stds = [s['std'] for s in lora_a_stats]
        print(f"Count: {len(lora_a_stats)}")
        print(f"Mean of Means: {np.mean(means):.6f}")
        print(f"Mean of Stds: {np.mean(stds):.6f}")
        print(f"First A stats: {lora_a_stats[0]}")
    else:
        print("No LoRA A keys found!")

    # Report LoRA B
    print("\n--- LoRA B (Expect learned weights. Init is usually 0. If still 0, training failed) ---")
    if lora_b_stats:
        means = [s['mean'] for s in lora_b_stats]
        stds = [s['std'] for s in lora_b_stats]
        abs_maxes = [s['abs_max'] for s in lora_b_stats]
        
        print(f"Count: {len(lora_b_stats)}")
        print(f"Mean of Means: {np.mean(means):.10f}")
        print(f"Mean of Stds: {np.mean(stds):.10f}")
        print(f"Max absolute value across all B: {np.max(abs_maxes):.10f}")
        
        zero_count = sum(1 for s in lora_b_stats if s['abs_max'] == 0.0)
        print(f"Number of perfectly zero matrices: {zero_count}/{len(lora_b_stats)}")
        
        print(f"First B stats: {lora_b_stats[0]}")
    else:
        print("No LoRA B keys found!")

    # Report HyperNet
    print("\n--- HyperNet (Expect learned weights) ---")
    if hyper_net_stats:
        means = [s['mean'] for s in hyper_net_stats]
        print(f"Count: {len(hyper_net_stats)}")
        print(f"Mean of Means: {np.mean(means):.6f}")
        print(f"First HyperNet stats: {hyper_net_stats[0]}")

if __name__ == "__main__":
    analyze_checkpoint("checkpoints/janus_v1/final/adapter_state.pt")
