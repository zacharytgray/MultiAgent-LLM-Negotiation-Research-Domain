import re
import random
from typing import Dict, Any, Optional
from .base_domain import BaseDomain, ParsedAction

class SingleIssuePriceDomain(BaseDomain):
    """
    Domain for single-issue price negotiation.
    Seller wants higher price, Buyer wants lower price.
    """
    
    def __init__(self):
        super().__init__("single_issue_price")
        self.buyer_id = 1 # Convention: Agent 1 is Buyer
        self.seller_id = 2 # Convention: Agent 2 is Seller
        
        # State
        self.current_offer: Optional[float] = None
        self.buyer_max: float = 0.0
        self.seller_min: float = 0.0
        self.agreement_price: Optional[float] = None
        self.last_offerer_id: Optional[int] = None
        
        # History
        self.offer_history = []
        
    def reset(self, round_id: int, **kwargs) -> Dict[str, Any]:
        self.round_id = round_id
        self.is_terminal_state = False
        self.current_offer = None
        self.agreement_price = None
        self.last_offerer_id = None
        self.offer_history = []
        
        # Standard Training Distribution (Paper Replication)
        # Buyer Max ~ N(900, 50)
        # Seller Min = Buyer Max - 500 (Fixed ZOPA width of 500)
        # This ensures agents trained on dataset generation see similar distributions in live testing.
        
        if "buyer_max_range" in kwargs:
             # Manual override path (legacy or specific tests)
             buyer_max_range = kwargs.get("buyer_max_range", (100, 200))
             self.buyer_max = round(random.uniform(*buyer_max_range), 2)
             # If seller range provided, use it, else generic
             seller_min_range = kwargs.get("seller_min_range", (50, 150))
             self.seller_min = round(random.uniform(*seller_min_range), 2)
        else:
             # Default to matching the training dataset distribution
             self.buyer_max = round(random.gauss(900, 50), 2)
             self.seller_min = round(self.buyer_max - 500.0, 2)
        
        return {
            "buyer_max": self.buyer_max,
            "seller_min": self.seller_min
        }

    def get_private_context(self, agent_id: int) -> Dict[str, Any]:
        context = {
            "history": self.offer_history,
            "last_offer": self.current_offer
        }
        if agent_id == self.buyer_id:
            context.update({
                "role": "buyer",
                "max_willingness_to_pay": self.buyer_max,
                "profit_formula": "Utility = Max_Willingness - Price"
            })
            return context
        elif agent_id == self.seller_id:
            context.update({
                "role": "seller",
                "min_acceptable_price": self.seller_min,
                "profit_formula": "Utility = Price - Min_Acceptable"
            })
            return context
        return {}

    def get_public_context(self) -> Dict[str, Any]:
        return {
            "item_name": "Widget",
            "currency": "USD"
        }

    def parse_agent_action(self, agent_id: int, text: str) -> ParsedAction:
        # Regex for OFFER <number> or just $<number>
        # We prefer explicit OFFER keyword, but fallback to finding a price if it's the only one.
        
        # 0. Check for explicit ACCEPT/AGREE *first* to prioritize agreement over implicit prices
        # However, we must ensure it's not "I cannot ACCEPT..."
        # Heuristic: If they say ACCEPT and OFFER, it's likely a counter-offer.
        # If they say ACCEPT and don't say OFFER, it's an agreement.
        
        text_upper = text.upper()
        has_offer_keyword = "OFFER" in text_upper
        has_accept_keyword = ("ACCEPT" in text_upper or "AGREE" in text_upper)
        
        # 1. Try explicit OFFER pattern
        offer_pattern = r"OFFER\s+\$?(\d+(?:\.\d+)?)"
        match = re.search(offer_pattern, text, re.IGNORECASE)
        
        if match:
            try:
                price = float(match.group(1))
                return ParsedAction("OFFER", price, text)
            except ValueError:
                pass

        # 2. Check for ACCEPT
        # If it has ACCEPT but no explicit OFFER, treat as ACCEPT.
        # Even if they mention a price implicitly "ACCEPT $100", that usually means "I accept the deal at $100".
        # We assume the domain validator will ensure it matches the current offer if needed, 
        # or we just consider it a generic accept.
        if has_accept_keyword and not match:
             # Even if there are implicit prices (like "I accept $100"), we treat it as ACCEPT.
             return ParsedAction("ACCEPT", None, text)

        # 3. Try implicit price pattern ($X)
        price_pattern = r"\$\s?(\d+(?:\.\d+)?)"
        matches = re.findall(price_pattern, text)
        if matches:
            unique_prices = list(set([float(m) for m in matches]))
            if len(unique_prices) == 1:
                 return ParsedAction("OFFER", unique_prices[0], text)
            elif len(unique_prices) > 0:
                 return ParsedAction("OFFER", float(matches[-1]), text)
            
        return ParsedAction("INVALID", None, text)

    def is_valid_action(self, action: ParsedAction) -> bool:
        if action.action_type == "OFFER":
            # Price must be non-negative
            return isinstance(action.offer_content, (int, float)) and action.offer_content >= 0
        
        if action.action_type == "ACCEPT":
            # Can only accept if there is an offer on the table
            return self.current_offer is not None
            
        return False

    def apply_action(self, action: ParsedAction, agent_id: int) -> bool:
        if not self.is_valid_action(action):
            return False
            
        if action.action_type == "OFFER":
            self.current_offer = action.offer_content
            self.last_offerer_id = agent_id
            self.offer_history.append((agent_id, self.current_offer))
            
        elif action.action_type == "ACCEPT":
            if self.last_offerer_id is not None and self.last_offerer_id != agent_id:
                self.agreement_price = self.current_offer
                self.is_terminal_state = True
                
        return True

    def is_agreement(self) -> bool:
        return self.agreement_price is not None

    def get_outcome(self) -> Dict[str, Any]:
        if not self.agreement_price:
            return {
                "agreement": False,
                "agent1_utility": 0.0,
                "agent2_utility": 0.0,
                "price": None,
                "within_zopa": False
            }
            
        # ZOPA Check: Seller Min <= Price <= Buyer Max
        within_zopa = (self.seller_min <= self.agreement_price <= self.buyer_max)
            
        return {
            "agreement": True,
            "agent1_utility": self.buyer_max - self.agreement_price, # Buyer
            "agent2_utility": self.agreement_price - self.seller_min, # Seller
            "price": self.agreement_price,
            "within_zopa": within_zopa
        }

    def format_agent_prompt_context(self, agent_id: int) -> str:
        ctx = self.get_private_context(agent_id)
        role = ctx.get("role", "negotiator")
        
        msg = f"\nYou are the {role}.\n"
        if role == "buyer":
            msg += f"You want to buy at the lowest possible price.\n"
            msg += f"Your maximum willingness to pay is ${self.buyer_max}.\n"
            msg += f"Any price above ${self.buyer_max} results in negative utility.\n"
        else:
            msg += f"You want to sell at the highest possible price.\n"
            msg += f"Your minimum acceptable price is ${self.seller_min}.\n"
            msg += f"Any price below ${self.seller_min} results in negative utility.\n"
            
        msg += "\nTo make an offer, say: OFFER <amount>\n"
        msg += "To accept the current standing offer, say: ACCEPT\n"
        
        if self.current_offer is not None:
            msg += f"The current standing offer is ${self.current_offer}.\n"
        
        return msg
