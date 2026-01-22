import matplotlib.pyplot as plt
import os
import random
# Ensure we use the src module from current directory
import sys
sys.path.append(os.getcwd())

from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.core.price_structures import PriceState

# Configuration
OUTPUT_DIR = "concession_plots"
MAX_TURNS = 20
BUYER_MAX = 900.0
SELLER_MIN = 400.0
# ZOPA is [400, 900]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

strategies = list(STRATEGY_REGISTRY.keys())
print(f"Found {len(strategies)} strategies: {strategies}")

for strategy_name in strategies:
    print(f"Simulating {strategy_name}...")
    
    # --- Simulate Buyer ---
    # Agent is Buyer (Max 900). Opponent acts as Stubborn Seller at 1500 (Way outside ZOPA)
    # This forces Buyer to reveal their full willingness to concede up to their Reservation (900).
    buyer_agent = DeterministicPriceAgent(1, strategy_name)
    
    # Inject Oracle params if needed (for random_zopa)
    if strategy_name == "random_zopa":
        buyer_agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})

    buyer_offers = []
    
    # History for state tracking
    offer_history_buyer_view = []
    last_offer_price = 1500.0 # High start from seller
    
    for t in range(1, MAX_TURNS + 1):
        # State construction
        state = PriceState(
            timestep=t,
            max_turns=MAX_TURNS,
            role="buyer",
            last_offer_price=None if t==1 else 1500.0,
            offer_history=list(offer_history_buyer_view), # Pass copy
            effective_reservation_price=BUYER_MAX,
            true_reservation_price=BUYER_MAX,
            public_price_range=(200.0, 2000.0)
        )
        
        try:
            action = buyer_agent.propose_action(state)
            if action.type == "OFFER":
                price = action.price
                buyer_offers.append(price)
                offer_history_buyer_view.append(("buyer", price))
                # Add dummy opponent response to history
                offer_history_buyer_view.append(("seller", 1500.0)) 
            else:
                # Agent Accepted (Unlikely given stubborn opponent, but possible)
                buyer_offers.append(last_offer_price) 
        except Exception as e:
            print(f"Error in {strategy_name} (Buyer): {e}")
            buyer_offers.append(None)

    # --- Simulate Seller ---
    # Agent is Seller (Min 400). Opponent acts as Stubborn Buyer at 0 (Way outside ZOPA)
    seller_agent = DeterministicPriceAgent(2, strategy_name)
    if strategy_name == "random_zopa":
        seller_agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})

    seller_offers = []
    offer_history_seller_view = []
    
    for t in range(1, MAX_TURNS + 1):
        state = PriceState(
            timestep=t,
            max_turns=MAX_TURNS,
            role="seller",
            last_offer_price=None if t==1 else 0.0,
            offer_history=list(offer_history_seller_view),
            effective_reservation_price=SELLER_MIN,
            true_reservation_price=SELLER_MIN,
            public_price_range=(0.0, 2000.0)
        )
        
        try:
            action = seller_agent.propose_action(state)
            if action.type == "OFFER":
                price = action.price
                seller_offers.append(price)
                offer_history_seller_view.append(("seller", price))
                offer_history_seller_view.append(("buyer", 0.0))
            else:
                seller_offers.append(0.0)
        except Exception as e:
            print(f"Error in {strategy_name} (Seller): {e}")
            seller_offers.append(None)


    # --- Plot ---
    plt.figure(figsize=(10, 6))
    turns = range(1, len(buyer_offers) + 1)
    
    # Filter Nones
    safe_buyer_offers = [o if o is not None else 0 for o in buyer_offers]
    safe_seller_offers = [o if o is not None else 0 for o in seller_offers]
    
    plt.plot(turns, safe_buyer_offers, label=f"Buyer (Res={BUYER_MAX})", marker='o', color='blue')
    plt.plot(turns, safe_seller_offers, label=f"Seller (Res={SELLER_MIN})", marker='x', color='red')
    
    plt.axhline(y=BUYER_MAX, color='blue', linestyle='--', alpha=0.3, label="Buyer Max")
    plt.axhline(y=SELLER_MIN, color='red', linestyle='--', alpha=0.3, label="Seller Min")
    
    # Shade ZOPA
    plt.fill_between(turns, SELLER_MIN, BUYER_MAX, color='green', alpha=0.1, label="ZOPA")
    
    plt.title(f"Concession Curve: {strategy_name}")
    plt.xlabel("Turn")
    plt.ylabel("Price Offer")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(OUTPUT_DIR, f"concession_{strategy_name}.png")
    plt.savefig(filename)
    plt.close()

print(f"Plots saved to {os.path.abspath(OUTPUT_DIR)}")
