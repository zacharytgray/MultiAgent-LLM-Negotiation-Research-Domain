
import asyncio
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from Negotiation import NegotiationSession

async def run_test():
    print("--- Running Janus (Buyer) vs Basic (Seller) Test ---")
    
    # Configure Janus as Agent 1 (Buyer) with rho=0.0 (Aggressive)
    # Explicitly set HF model path so it doesn't try to load the Ollama tag from HF
    agent1_config = {
        "rho": 0.0, # 0.0 is buyer, 1.0 is seller
        "adapter_path": "checkpoints/final",
        "model_path": "Qwen/Qwen2-7B-Instruct" 
    }
    
    # Configure Basic as Agent 2 (Seller)
    agent2_config = {}
    
    session = NegotiationSession(
        num_rounds=1,
        agent1_type="janus",
        agent2_type="basic_price",
        agent1_config=agent1_config,
        agent2_config=agent2_config,
        domain_type="price",
        model_name="qwen2:7b" # Use Ollama tag for the session default
    )
    
    await session.run_round(1)

if __name__ == "__main__":
    asyncio.run(run_test())
