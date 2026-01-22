from typing import Optional, Dict, Any
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent

class PriceFixedAgent(BaseAgent):
    """
    Fixed Price agent for Single Issue Price Domain.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str, 
                 fixed_margin: float = 10.0):
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        self.fixed_margin = fixed_margin
        
    async def generate_response(self) -> str:
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.addToMemory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()

    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        return True
    
    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        """
        Returns a dict describing the intended action.
        For price domain: {"action": "OFFER", "price": float} or {"action": "ACCEPT"}
        """
        if not self.domain_private_context:
            return None
            
        role = self.domain_private_context.get("role")
        
        target_price = 0.0
        if role == "buyer":
            buyer_max = self.domain_private_context.get("max_willingness_to_pay", 0)
            target_price = buyer_max - self.fixed_margin
        else:
            seller_min = self.domain_private_context.get("min_acceptable_price", 0)
            target_price = seller_min + self.fixed_margin
            
        # Ensure non-negative
        target_price = max(0.0, target_price)
        
        return {"action": "OFFER", "price": target_price}

    def should_accept_proposal(self, proposal: Any, turn_number: int = 1) -> bool:
        """
        proposal: ParsedAction (or similar)
        """
        # Note: Negotiation.py currently passes ParsedProposal (multi-item).
        # We need to support ParsedAction for price domain.
        
        # If proposal is not a price offer, can't accept
        if not hasattr(proposal, "offer_content") or not isinstance(proposal.offer_content, float):
             return False

        offered_price = proposal.offer_content
        role = self.domain_private_context.get("role")
        
        if role == "buyer":
            max_pay = self.domain_private_context.get("max_willingness_to_pay", 0)
            # Accept if price <= max willing + maybe some margin? Or strict?
            # Requirement: "Accept if current offer meets bound"
            return offered_price <= max_pay
        else:
            min_acc = self.domain_private_context.get("min_acceptable_price", 0)
            return offered_price >= min_acc

    def validate_output_matches_intent(self, response: str, intended_proposal: Optional[Dict]) -> bool:
        # Simple validation: check if number is in response
        if not intended_proposal:
            return True
            
        if intended_proposal.get("action") == "OFFER":
            price_str = str(intended_proposal["price"])
            # Allow some float formatting flexibility? 
            # Or just check if "OFFER" is present.
            return "OFFER" in response.upper() and str(int(intended_proposal["price"])) in response # Rough check
            
        if intended_proposal.get("action") == "ACCEPT":
             return "ACCEPT" in response.upper()
             
        return True
