from typing import Optional, Dict, Any, List, Tuple
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent
from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.core.price_structures import PriceState, PriceAction

class PriceStrategyWrapperAgent(BaseAgent):
    """
    Wrapper Agent for Single Issue Price Domain.
    Uses a deterministic strategy (from price_strategies) to calculate the next move,
    then uses an LLM to wrap that move in natural language.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str,
                 strategy_name: str = "boulware_linear", strategy_params: Optional[Dict] = None):
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        
        # We use the DeterministicPriceAgent logic internally
        # We create a dummy instance just to leverage its propose_action method if needed,
        # or we just call the strategy function directly.
        # But DeterministicPriceAgent wraps the registry text.
        self.det_agent = DeterministicPriceAgent(agent_id, strategy_name, strategy_params)
        
    async def generate_response(self) -> str:
        # This is called by Negotiation.py when it wants the LLM text.
        # The prompt with the mandated offer should have already been added to memory
        # by Negotiation.py's `_process_deterministic_agent_turn`.
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.addToMemory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()
        
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        # We WANT the system to drive us via _process_deterministic_agent_turn
        # which calculates the move, prompts us, validates output.
        return True
    
    def validate_output_matches_intent(self, response: str, intended_proposal: Dict) -> bool:
        """
        Check if the LLM's response actually contains the intended price/action.
        Negotiation.py calls this.
        """
        action = intended_proposal.get("action")
        
        if action == "ACCEPT":
            # Check for accept/agree keywords
            # Simple check
            return "accept" in response.lower() or "agree" in response.lower()
            
        elif action == "OFFER":
            target_price = intended_proposal.get("price")
            # We need to extract the price from the text and see if it matches target_price
            # Extract numbers
            import re
            # Regex for money like $100 or 100.00
            matches = re.findall(r'\$?\s?(\d+(?:\.\d+)?)', response)
            
            if not matches:
                return False
                
            # Check if any match equals the target price (with tolerance)
            for m in matches:
                try:
                    val = float(m)
                    if abs(val - target_price) < 1.0: # $1 tolerance
                        return True
                except:
                    continue
            
            return False
            
        return False

    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        """
        Calculate the move using the underlying strategy.
        Requires domain_private_context to be up to date (specifically history).
        """
        if not self.domain_private_context:
            return None

        # 1. Reconstruct PriceState from context
        # domain_private_context should contain: role, history, max_turns, reservation...
        # We rely on Negotiation.py to have updated this via set_domain_context.
        
        role = self.domain_private_context.get("role")
        history = self.domain_private_context.get("history", []) # List of (role, price)
        max_turns = self.domain_private_context.get("max_turns", 20)
        
        # Reservation
        # Private context in SingleIssuePriceDomain: {"role":..., "min_acceptable_price": ..., "max_willingness_to_pay":...}
        if role == "buyer":
            true_res = self.domain_private_context.get("max_willingness_to_pay")
        else:
            true_res = self.domain_private_context.get("min_acceptable_price")
            
        last_offer = None
        if history:
            last_offer = history[-1][1]
            
        # Reformat history for State (list of tuples) - assuming it matches
        # state.offer_history expects List[Tuple[str, float]]
        
        state = PriceState(
            timestep=turn_number,
            max_turns=max_turns,
            role=role,
            last_offer_price=last_offer,
            offer_history=history,
            effective_reservation_price=true_res,
            true_reservation_price=true_res,
            public_price_range=(0, 2000) # Dummy/Approx
        )
        
        # 2. Get Action
        action = self.det_agent.propose_action(state)
        
        # 3. Convert to Dict for Negotiation.py
        result = {
            "action": action.type, # "OFFER" or "ACCEPT"
        }
        if action.type == "OFFER":
            result["price"] = action.price
            
        return result
