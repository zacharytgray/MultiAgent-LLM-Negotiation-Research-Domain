from typing import Optional, Dict, Any
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent
from config.settings import MAX_TURNS_PER_ROUND

class PriceBoulwareAgent(BaseAgent):
    """
    Boulware Agent for Single Issue Price Domain.
    Concedes over time based on exponential curve.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str,
                 beta: float = 3.0): # High beta = tough negotiator (concedes late)
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        self.beta = beta
        
    async def generate_response(self) -> str:
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.addToMemory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()
        
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        return True
    
    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        if not self.domain_private_context:
            return None

        role = self.domain_private_context.get("role")
        
        # Determine strict bounds
        reservation_price = 0.0
        start_price = 0.0
        
        if role == "buyer":
            reservation_price = self.domain_private_context.get("max_willingness_to_pay", 100)
            # Start low. say 50% of reservation? 
            start_price = reservation_price * 0.5 
        else:
            reservation_price = self.domain_private_context.get("min_acceptable_price", 100)
            # Start high. say 150% of reservation?
            start_price = reservation_price * 1.5

        # Boulware time fraction
        # t/T
        time_fraction = min(turn_number / MAX_TURNS_PER_ROUND, 1.0)
        
        # Curve: f(t) = (t/T)^beta
        # Buyer: Offer = Start + (Reservation - Start) * f(t)  (Going UP)
        # Seller: Offer = Start - (Start - Reservation) * f(t) (Going DOWN)
        
        curve_val = time_fraction ** self.beta
        
        target_price = 0.0
        if role == "buyer":
            target_price = start_price + (reservation_price - start_price) * curve_val
        else:
            target_price = start_price - (start_price - reservation_price) * curve_val
            
        target_price = round(max(0.0, target_price), 2)
        
        return {"action": "OFFER", "price": target_price}

    def should_accept_proposal(self, proposal: Any, turn_number: int = 1) -> bool:
        if not hasattr(proposal, "offer_content") or not isinstance(proposal.offer_content, float):
             return False

        offered_price = proposal.offer_content
        role = self.domain_private_context.get("role")
        
        # Calculate my current internal target price (what I would offer)
        # If the offer is better than what I am willing to offer now, Accept?
        # Or if it is better than my reservation?
        # Typically, standard strategy: Accept if offer is better than my value in next round?
        # Or simple: Accept if offer meets my reservation price? 
        # Requirement: "Accept if current offer meets bound" (from fixed price description, but applies generally)
        
        if role == "buyer":
            # Buyer wants Low. Accept if Price <= Reservation
            reservation = self.domain_private_context.get("max_willingness_to_pay", 0)
            return offered_price <= reservation
        else:
            # Seller wants High. Accept if Price >= Reservation
            reservation = self.domain_private_context.get("min_acceptable_price", 0)
            return offered_price >= reservation

    def validate_output_matches_intent(self, response: str, intended_proposal: Optional[Dict]) -> bool:
         if not intended_proposal: return True
         # Rough check
         return "OFFER" in response.upper() or "ACCEPT" in response.upper()
