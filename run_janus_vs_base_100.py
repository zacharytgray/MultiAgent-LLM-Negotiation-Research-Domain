
import asyncio
import sys
import os
import random
from src.utils.CSVLogger import CSVLogger
from colorama import Fore

# Ensure we can import from src
sys.path.append(os.getcwd())

from Negotiation import NegotiationSession

async def run_loop():
    print("--- Running 100 Rounds: Base LLM vs Janus (Alternating) ---")
    
    # Janus Config
    janus_adapter = "checkpoints/final"
    janus_model = "Qwen/Qwen2-7B-Instruct"
    
    # Base LLM Config (Ollama)
    base_model_ollama = "qwen2:7b"
    
    # Initialize Logger ONCE
    # Filename: qwen2_7b_janus_alternating_100_runs_<num_items>_<timestamp>.csv
    # We pass 0 for num_items as it is a price domain
    logger = CSVLogger(f"{base_model_ollama.replace(':','_')}_janus_alternating_100_runs", 0)
    print(f"{Fore.GREEN}[CSV] Logging all rounds to: {logger.get_filename()}{Fore.RESET}")
    
    for i in range(1, 101):
        
        is_janus_buyer = (i % 2 == 1) # Odd rounds: Janus is Buyer (1)
        
        # Concise status update
        role_str = "Buyer (rho=0.0)" if is_janus_buyer else "Seller (rho=1.0)"
        print(f"\n{Fore.MAGENTA}>>> Starting Round {i}/100 | Janus Role: {role_str}{Fore.RESET}")
        
        if is_janus_buyer:
            agent1_type = "janus"
            agent1_config = {
                "rho": 0.0,
                "adapter_path": janus_adapter,
                "model_path": janus_model
            }
            
            agent2_type = "basic_price"
            agent2_config = {}
            
        else:
            agent1_type = "basic_price"
            agent1_config = {}
            
            agent2_type = "janus"
            agent2_config = {
                "rho": 1.0, # Aggressive Seller
                "adapter_path": janus_adapter,
                "model_path": janus_model
            }
            
        # Create Session for this round
        # Pass the SHARED LOGGER so we append to the same file
        session = NegotiationSession(
            num_rounds=1, 
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            agent1_config=agent1_config,
            agent2_config=agent2_config,
            domain_type="price",
            model_name=base_model_ollama,
            csv_logger=logger # Shared Logger
        )
        
        # Monkey-patch domain reset
        original_reset = session.domain.reset
        def fixed_reset(round_id, **kwargs):
            session.domain.round_id = round_id
            session.domain.buyer_max = round(random.gauss(900, 50), 2)
            session.domain.seller_min = round(session.domain.buyer_max - 500.0, 2)
            session.domain.offer_history = []
            session.domain.current_offer = None
            return {"items": []}
        
        session.domain.reset = fixed_reset
        
        # Run Round 'i' (Log as Round i, not Round 1)
        await session.run_round(i)

if __name__ == "__main__":
    asyncio.run(run_loop())
