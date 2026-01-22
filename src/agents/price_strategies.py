"""
Strategy registry and implementations for deterministic price negotiation agents.
"""
import random
import math
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, Tuple, Literal
from src.core.price_structures import PriceAction, PriceState
from src.agents.base_agent import BaseAgent

# --- Strategy Logic Implementations ---

def strategy_boulware(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Boulware strategy: time-dependent concession curve.
    Target = Start + (Reservation - Start) * (t/T)^beta
    """
    beta = params.get("beta", 1.0) # Concession rate. <1: conceding, >1: hardliner
    
    # Identify bounds and start
    # For a buyer: Start low (lower bound or lower) and move up to reservation
    # For a seller: Start high and move down to reservation
    
    # We need a starting offer. If not provided in params, we need a default.
    # Params should probably define 'initial_margin' or similar.
    # Assuming params: {'beta': float, 'initial_offer_margin': float}
    # where initial_offer_margin is absolute amount away from reservation? 
    # Or maybe relative to the "visible range"?
    
    # Simplified approach:
    # Buyer starts at seller_min (if known) or some low value? 
    # But seller_min is PRIVATE to seller. Buyer doesn't know it.
    # The requirement says "A spectrum of 5 Boulware agents with different parameters".
    
    # Let's assume the agent has an "initial_price" concept or "start_fraction".
    # Since we need to be purely deterministic based on STATE and PARAMS.
    # State has effective_reservation_price.
    
    reservation = state.effective_reservation_price

    # Cap time_frac to prevent full capitulation to reservation price at the last turn.
    # This ensures a small margin remains, allowing for impasses if the opponent doesn't bridge the gap.
    concession_cap = params.get("concession_cap", 0.95)
    time_frac = min(concession_cap, state.timestep / state.max_turns)
    
    # We need a "start_price".
    # Ideally, a buyer starts at 0 or substantial margin.
    # A seller starts at 2*reservation or substantial margin.
    
    # Let's define start_price = reservation * start_factor (e.g. 0.5 for buyer, 1.5 for seller)
    # OR start_price = reservation +/- margin
    
    margin = params.get("static_margin", 50.0) # Absolute scalar if not specified
    
    if state.role == "buyer":
        start_price = max(0.0, reservation - margin)
        # Target goes from start_price to reservation
        # P(t) = Start + (Reservation - Start) * (t/T)^beta
        target_price = start_price + (reservation - start_price) * (time_frac ** beta)
        
        # Cap target at reservation
        target_price = min(target_price, reservation)
        
    else: # seller
        start_price = reservation + margin
        # Target goes from start_price to reservation (downwards)
        target_price = start_price + (reservation - start_price) * (time_frac ** beta)
        
        # Floor target at reservation
        target_price = max(target_price, reservation)

    # Check for acceptance
    if state.last_offer_price is not None:
        if state.role == "buyer":
            if state.last_offer_price <= target_price:
                return PriceAction(type="ACCEPT", price=None)
            # Or if last offer is better than our reservation, and we are near end?
            # Standard Boulware just proposes target.
            # But we should accept if offer is better than our target.
            if state.last_offer_price <= reservation and state.last_offer_price <= target_price:
                 return PriceAction(type="ACCEPT", price=None)
        else: # seller
            if state.last_offer_price >= target_price:
                return PriceAction(type="ACCEPT", price=None)
            if state.last_offer_price >= reservation and state.last_offer_price >= target_price:
                 return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(target_price, 2))

def strategy_price_fixed(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Fixed Price strategy: Always offers reservation +/- margin.
    """
    margin = params.get("margin", 0.0)
    reservation = state.effective_reservation_price
    
    if state.role == "buyer":
        target_price = reservation - margin
        # Accept if offer <= target
        if state.last_offer_price is not None and state.last_offer_price <= target_price:
            return PriceAction(type="ACCEPT", price=None)
    else:
        target_price = reservation + margin
        # Accept if offer >= target
        if state.last_offer_price is not None and state.last_offer_price >= target_price:
            return PriceAction(type="ACCEPT", price=None)
            
    return PriceAction(type="OFFER", price=round(target_price, 2))

def strategy_tit_for_tat(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Tit-for-Tat: Mirror opponent's concession.
    Start with an extreme offer.
    """
    role = state.role
    reservation = state.effective_reservation_price
    initial_margin = params.get("initial_margin", 50.0)
    
    # Initial offer or if no history
    if not state.offer_history:
        if role == "buyer":
            return PriceAction(type="OFFER", price=round(reservation - initial_margin, 2))
        else:
            return PriceAction(type="OFFER", price=round(reservation + initial_margin, 2))
            
    # Need at least one previous offer from opponent to gauge concession, 
    # and one previous offer from US to base our next move on.
    
    # Filter history
    my_offers = [p for r, p in state.offer_history if r == role]
    opp_offers = [p for r, p in state.offer_history if r != role]
    
    if not my_offers:
        # We haven't made an offer yet (e.g. we are second mover), start high/low
         if role == "buyer":
            target = reservation - initial_margin
         else:
            target = reservation + initial_margin
         
         # Check acceptance first
         if state.last_offer_price is not None:
             if role == "buyer" and state.last_offer_price <= target:
                 return PriceAction(type="ACCEPT")
             if role == "seller" and state.last_offer_price >= target:
                 return PriceAction(type="ACCEPT")
         return PriceAction(type="OFFER", price=round(target, 2))
    
    if len(opp_offers) < 2:
        # Opponent hasn't moved twice, so we can't see a concession.
        # Just repeat last offer or concede slightly?
        # TFT usually starts with cooperation or defection. 
        # Here we just hold or small concession? Let's hold.
        target = my_offers[-1]
    else:
        # Calculate opponent concession
        # opp_new = opp_offers[-1], opp_old = opp_offers[-2]
        concession = abs(opp_offers[-1] - opp_offers[-2])
        
        # We concede same amount
        if role == "buyer":
            target = my_offers[-1] + concession
            # Cap at reservation
            target = min(target, reservation)
        else:
            target = my_offers[-1] - concession
            # Floor at reservation
            target = max(target, reservation)
            
    # Check acceptance
    if state.last_offer_price is not None:
        # Standard acceptance logic against reservation
        is_acceptable = (state.last_offer_price <= reservation if role == "buyer" else state.last_offer_price >= reservation)
        # And better than our planned target?
        is_better_than_target = (state.last_offer_price <= target if role == "buyer" else state.last_offer_price >= target)
        
        if is_acceptable and is_better_than_target:
             return PriceAction(type="ACCEPT")
             
    return PriceAction(type="OFFER", price=round(target, 2))

def strategy_linear(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Linear concessions from Start to Reservation over MaxRounds.
    """
    # Effectively Boulware with beta=1.0
    return strategy_boulware(state, {**params, "beta": 1.0})

def strategy_split_difference(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Split difference between last offer and own previous offer (or reservation).
    """
    role = state.role
    reservation = state.effective_reservation_price
    
    if state.last_offer_price is None:
        # Start aggressively
        margin = params.get("initial_margin", 50.0)
        target = reservation - margin if role == "buyer" else reservation + margin
        return PriceAction(type="OFFER", price=round(target, 2))
        
    opp_price = state.last_offer_price
    
    # Get my last offer
    my_offers = [p for r, p in state.offer_history if r == role]
    if not my_offers:
        # Use reservation + margin as anchor if no history? 
        # Or split difference between Reservation and Offer?
        # "Split the difference" usually implies meeting in the middle.
        my_anchor = reservation
    else:
        my_anchor = my_offers[-1]
        
    # Calculate midpoint
    midpoint = (my_anchor + opp_price) / 2.0
    
    # Check bounds
    if role == "buyer":
        # Don't exceed reservation
        target = min(midpoint, reservation)
        if opp_price <= target: # They offered lower than our midpoint
             return PriceAction(type="ACCEPT")
    else:
        target = max(midpoint, reservation)
        if opp_price >= target:
             return PriceAction(type="ACCEPT")
             
    return PriceAction(type="OFFER", price=round(target, 2))

def strategy_time_dependent_threshold(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Accepts only if offer improves. Threshold relaxes as deadline approaches.
    Does NOT calculate a target offer curve explicitly but sets an acceptance threshold.
    If must offer, offers threshold.
    """
    reservation = state.effective_reservation_price
    role = state.role
    margin = params.get("margin", 20.0)
    
    # Acceptance threshold relaxes linearly from (Res +/- Margin) to Res
    t = state.timestep
    T = state.max_turns
    
    # Cap frac to retain some margin at the deadline
    concession_cap = params.get("concession_cap", 0.95)
    frac = min(concession_cap, t / T)
    
    if role == "buyer":
        # Start threshold: Res - Margin. End threshold: Res.
        current_threshold = (reservation - margin) + (margin * frac)
        target = min(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price <= target:
            return PriceAction(type="ACCEPT")
    else:
        current_threshold = (reservation + margin) - (margin * frac)
        target = max(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price >= target:
            return PriceAction(type="ACCEPT")
            
    return PriceAction(type="OFFER", price=round(target, 2))

def strategy_hardliner(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Maintains a tough stance until the very last round, then concedes to reservation.
    """
    reservation = state.effective_reservation_price
    role = state.role
    margin = params.get("margin", 30.0)
    
    if state.timestep >= state.max_turns - 1:
        # Cave in, but preserve a small margin to avoid automatic acceptance of worst-case deals.
        cave_in_margin = params.get("cave_in_margin", 5.0) 
        if role == "buyer":
            target = reservation - cave_in_margin
        else:
            target = reservation + cave_in_margin
    else:
        # Hold line
        if role == "buyer":
            target = reservation - margin
        else:
            target = reservation + margin
            
    if state.last_offer_price is not None:
        if role == "buyer" and state.last_offer_price <= target:
                return PriceAction(type="ACCEPT")
        if role == "seller" and state.last_offer_price >= target:
                return PriceAction(type="ACCEPT")

    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_random_in_zopa(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Randomly offers within ZOPA. Oracle strategy (training only).
    Requires ZOPA bounds passed in params (cheating).
    """
    zopa_min = params.get("zopa_min")
    zopa_max = params.get("zopa_max")
    
    if zopa_min is None or zopa_max is None:
        # Fallback if no oracle info
        return strategy_linear(state, {})
        
    # Generate random offer in ZOPA
    offer = random.uniform(zopa_min, zopa_max)
    
    # If we can accept current offer (it's in ZOPA and better than our random pick?)
    # Random agent behavior varies. Let's say it randomly accepts if offer is in ZOPA.
    if state.last_offer_price is not None:
        if zopa_min <= state.last_offer_price <= zopa_max:
             if random.random() < 0.3: # 30% chance to accept a valid ZOPA offer
                 return PriceAction(type="ACCEPT")
                 
    return PriceAction(type="OFFER", price=round(offer, 2))

def strategy_micro(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    MiCRO (Minimal Concession Strategy).
    Offers from a pre-sorted grid (fine/coarse) based on 'step_size' (delta).
    Concedes (offers next best price) only if we haven't made more unique offers than opponent.
    Otherwise repeats history.
    """
    step_size = params.get("step_size", 10.0)
    role = state.role
    my_res = state.true_reservation_price
    
    # 1. Grid Construction
    # Use public range to define the negotiation field
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Create Grid
    grid = []
    curr = math.floor(pub_min)
    end = math.ceil(pub_max)
    while curr <= end:
        grid.append(float(curr))
        curr += step_size
        
    # Sort Grid by Preference order
    if role == "buyer":
        # Buyer: Low to High
        sorted_prices = sorted(grid)
    else:
        # Seller: High to Low
        sorted_prices = sorted(grid, reverse=True)
            
    # 2. History Analysis
    own_offers = [] # List for valid resampling
    own_offers_set = set() # Set for counting unique
    opp_offers_set = set()
    last_opp_offer = None
    
    for r, p in state.offer_history:
        if r == role:
            own_offers.append(p)
            own_offers_set.add(p)
        else:
            opp_offers_set.add(p)
            last_opp_offer = p
            
    m = len(own_offers_set)
    n = len(opp_offers_set)
    
    # 3. Identify Next Potential Concession (P_new)
    # The first price in our sorted preference that we haven't offered yet
    p_new = None
    for p in sorted_prices:
        if p not in own_offers_set:
            p_new = p
            break
            
    # Helpers
    def is_acceptable(price):
        if price is None: return False
        if role == "buyer":
            return price <= my_res
        else:
            return price >= my_res

    def is_better_or_equal(p1, p2):
        if role == "buyer":
            return p1 <= p2
        else:
            return p1 >= p2

    # 4. Determine Threshold (P_thresh) for Acceptance/Concession limit
    # "What I am willing to offer next/now"
    p_thresh = None
    
    # Rule: If m <= n, we are "behind" or "even" in unique offers, so we are willing to concede to p_new.
    # If m > n, we are "ahead", so we hold firm at our previous worst offer.
    
    can_concede = (m <= n)
    
    if can_concede and p_new is not None and is_acceptable(p_new):
        p_thresh = p_new
    else:
        # Threshold is worst proposed so far
        if own_offers:
             if role == "buyer":
                 p_thresh = max(own_offers_set)
             else:
                 p_thresh = min(own_offers_set)
        else:
            # Nothing proposed yet (Start of game), threshold is best possible
            p_thresh = sorted_prices[0] if sorted_prices else my_res

    # 5. Acceptance Check
    if last_opp_offer is not None:
        # We accept if opponent offer is better than our threshold AND acceptable wrt reservation
        # Note: MiCRO accepts if u(p_opp) >= max(u(p_thresh), u(r))
        # Since p_thresh is already checked against r (mostly), this combines them.
        if is_better_or_equal(last_opp_offer, p_thresh) and is_acceptable(last_opp_offer):
            return PriceAction(type="ACCEPT")
            
    # 6. Proposal Generation
    proposal_price = None
    
    if can_concede and p_new is not None and is_acceptable(p_new):
        # We can concede, and p_new is valid
        proposal_price = p_new
    else:
        # Repetition Strategy (MiCRO standard: repeat random)
        if own_offers:
            proposal_price = random.choice(own_offers)
        else:
            # Fallback start
             proposal_price = sorted_prices[0] if sorted_prices else my_res

    return PriceAction(type="OFFER", price=round(proposal_price, 2))

# --- Registry and Metadata ---

@dataclass
class StrategySpec:
    name: str
    description: str
    func: Callable
    default_params: Dict[str, Any]

STRATEGY_REGISTRY = {
    # Boulware Spectrum
    "boulware_very_conceding": StrategySpec(
        "boulware_very_conceding", 
        "Rapidly concedes early (beta=0.2)", 
        strategy_boulware, 
        {"beta": 0.2, "static_margin": 400.0}
    ),
    "boulware_conceding": StrategySpec(
        "boulware_conceding", 
        "Concedes moderately early (beta=0.5)", 
        strategy_boulware, 
        {"beta": 0.5, "static_margin": 400.0}
    ),
    "boulware_linear": StrategySpec(
        "boulware_linear", 
        "Linear concession (beta=1.0)", 
        strategy_boulware, 
        {"beta": 1.0, "static_margin": 400.0}
    ),
     "boulware_firm": StrategySpec(
        "boulware_firm", 
        "Concedes slowly (beta=2.0)", 
        strategy_boulware, 
        {"beta": 2.0, "static_margin": 400.0}
    ),
    "boulware_hard": StrategySpec(
        "boulware_hard", 
        "Concedes very slowly (beta=4.0)", 
        strategy_boulware, 
        {"beta": 4.0, "static_margin": 400.0}
    ),
    
    # Price Fixed
    "price_fixed_strict": StrategySpec(
        "price_fixed_strict",
        "Offers exactly reservation +/- small margin",
        strategy_price_fixed,
        {"margin": 20.0} # Narrow margin
    ),
    "price_fixed_loose": StrategySpec(
        "price_fixed_loose",
        "Offers reservation +/- large margin",
        strategy_price_fixed,
        {"margin": 100.0}
    ),

    # Tit for Tat
    "tit_for_tat": StrategySpec(
        "tit_for_tat",
        "Mirrors opponent concessions",
        strategy_tit_for_tat,
        {"initial_margin": 100.0}
    ),
    
    # Linear / Steady (Generic)
    "linear_standard": StrategySpec(
        "linear_standard",
        "Standard linear concession",
        strategy_linear,
        {"static_margin": 400.0}
    ),

    # Split Difference
    "split_difference": StrategySpec(
        "split_difference",
        "Splits difference between last offer and own history",
        strategy_split_difference,
        {"initial_margin": 400.0}
    ),
    
    # Time Dependent
    "time_dependent": StrategySpec(
        "time_dependent",
        "Acceptance threshold relaxes over time",
        strategy_time_dependent_threshold,
        {"margin": 200.0}
    ),
    
    # Hardliner
    "hardliner": StrategySpec(
        "hardliner",
        "Hold firm until final round",
        strategy_hardliner,
        {"margin": 400.0}
    ),
    
    # Random Oracle
    "random_zopa": StrategySpec(
        "random_zopa",
        "Random offers within ZOPA (Oracle)",
        strategy_random_in_zopa,
        {} # Expects zopa_min/max injected at runtime
    ),
    
    # MiCRO Strategies
    "micro_fine": StrategySpec(
        "micro_fine",
        "MiCRO agent with fine grid (step=5.0)",
        strategy_micro,
        {"step_size": 5.0} 
    ),
    "micro_moderate": StrategySpec(
        "micro_moderate",
        "MiCRO agent with moderate grid (step=25.0)",
        strategy_micro,
        {"step_size": 25.0} 
    ),
    "micro_coarse": StrategySpec(
        "micro_coarse",
        "MiCRO agent with coarse grid (step=100.0)",
        strategy_micro,
        {"step_size": 100.0} 
    )
}

# --- Deterministic Agent Wrapper ---

class DeterministicPriceAgent(BaseAgent):
    """
    A concrete agent class that uses a strategy function for 'propose_action'.
    """
    def __init__(self, agent_id: int, strategy_name: str, strategy_params: Optional[Dict] = None):
        # We don't need model_name or system_instructions for this mode, pass dummy
        super().__init__(agent_id, "deterministic", "none")
        
        self.strategy_name = strategy_name
        
        spec = STRATEGY_REGISTRY.get(strategy_name)
        if not spec:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        self.strategy_func = spec.func
        # Merge default params with overrides
        self.params = spec.default_params.copy()
        if strategy_params:
            self.params.update(strategy_params)
            
        self.description = spec.description
        
    async def generate_response(self) -> str:
        raise NotImplementedError("DeterministicPriceAgent is for dataset_mode only (no LLM text).")
        
    def add_to_memory(self, role: str, content: str):
        pass # No memory needed for pure state functions
        
    def reset_memory(self):
        pass
        
    def propose_action(self, state: PriceState) -> PriceAction:
        """
        Delegates to the strategy function.
        """
        return self.strategy_func(state, self.params)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_type": "deterministic_price_agent",
            "strategy": self.strategy_name,
            "strategy_params": self.params,
            "strategy_description": self.description
        }
