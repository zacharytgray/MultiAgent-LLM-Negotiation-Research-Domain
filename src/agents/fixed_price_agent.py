"""
Fixed Price agent implementation with constant threshold strategy.
Uses a fixed threshold that never changes throughout the negotiation.
"""

import itertools
from typing import List, Optional, Dict, Tuple
from src.core.Item import Item
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent
from src.utils.MessageParser import ParsedProposal
from config.settings import BOULWARE_INITIAL_THRESHOLD


class FixedPriceAgent(BaseAgent):
    """
    Fixed Price agent that uses a constant threshold strategy.
    The LLM is used as a wrapper to present the deterministic proposals naturally.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str, 
                 fixed_threshold: Optional[float] = None):
        """
        Initialize the Fixed Price agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            fixed_threshold: Fixed threshold percentage (defaults to settings.BOULWARE_INITIAL_THRESHOLD)
        """
        super().__init__(agent_id, model_name, system_instructions_file)
        self.fixed_threshold = fixed_threshold if fixed_threshold is not None else BOULWARE_INITIAL_THRESHOLD
        self.ranked_allocations = []
        self.all_possible_allocations = []
        self.intended_proposal = None
        
        # Create the underlying ollama agent with special system instructions
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
    
    def _calculate_agent_welfare(self, allocation: Dict[str, List[str]]) -> float:
        """
        Calculate this agent's welfare for a given allocation.
        
        Args:
            allocation: Dictionary with agent1 and agent2 item lists
            
        Returns:
            float: Total welfare for this agent
        """
        if not self.current_items:
            return 0.0
            
        agent_key = f"agent{self.agent_id}"
        agent_items = allocation.get(agent_key, [])
        
        total_welfare = 0.0
        for item in self.current_items:
            if item.name in agent_items:
                agent_value_key = f"agent{self.agent_id}Value"
                total_welfare += getattr(item, agent_value_key)
        
        return total_welfare
    
    def _generate_all_possible_allocations(self) -> List[Dict[str, List[str]]]:
        """
        Generate all possible allocations of items between the two agents.
        
        Returns:
            List of allocation dictionaries
        """
        if not self.current_items:
            return []
        
        item_names = [item.name for item in self.current_items]
        all_allocations = []
        
        # Generate all possible subsets for agent1
        for r in range(len(item_names) + 1):
            for agent1_items in itertools.combinations(item_names, r):
                agent1_list = list(agent1_items)
                agent2_list = [item for item in item_names if item not in agent1_list]
                
                allocation = {
                    "agent1": agent1_list,
                    "agent2": agent2_list
                }
                all_allocations.append(allocation)
        
        return all_allocations
    
    def _rank_allocations_by_welfare(self) -> List[Dict[str, List[str]]]:
        """
        Rank all possible allocations by this agent's welfare.
        
        Returns:
            List of allocations sorted by welfare (lowest to highest)
        """
        all_allocations = self._generate_all_possible_allocations()
        
        # Calculate welfare for each allocation and sort
        allocation_welfare_pairs = []
        for allocation in all_allocations:
            welfare = self._calculate_agent_welfare(allocation)
            allocation_welfare_pairs.append((allocation, welfare))
        
        # Sort by welfare (lowest to highest)
        allocation_welfare_pairs.sort(key=lambda x: x[1])
        
        # Return just the allocations in sorted order
        return [allocation for allocation, welfare in allocation_welfare_pairs]
    
    def _get_fixed_price_proposal_index(self) -> int:
        """
        Get the index in ranked_allocations based on fixed threshold.
        Unlike Boulware, this threshold never changes.
        
        Returns:
            int: Index of the proposal to make
        """
        if not self.ranked_allocations:
            return 0
        
        max_index = len(self.ranked_allocations) - 1
        
        # Calculate index based on fixed threshold
        # threshold of 1.0 = best allocation (last index)
        # threshold of 0.0 = worst allocation (first index)
        index = int(self.fixed_threshold * max_index)
        
        # Ensure index is within bounds
        return min(max(index, 0), max_index)
    
    async def generate_response(self) -> str:
        """
        Generate a response using the underlying ollama agent with deterministic guidance.
        
        Returns:
            str: The agent's response message
        """
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str, tool_call_id: Optional[str] = None):
        """
        Add a message to the agent's memory via the ollama agent.
        
        Args:
            role: Role of the message (system, user, assistant, tool)
            content: Content of the message
            tool_call_id: Optional tool call ID for tool messages
        """
        self.ollama_agent.addToMemory(role, content, tool_call_id)
    
    def reset_memory(self):
        """
        Reset the agent's memory to initial state and reset strategy.
        """
        self.ollama_agent.reset_memory()
        self.ranked_allocations = []
        self.all_possible_allocations = []
        self.intended_proposal = None
    
    def set_items(self, items: List[Item]):
        """
        Set the items for this negotiation round and calculate strategy.
        
        Args:
            items: List of items with values for both agents
        """
        self.current_items = items
        # Recalculate allocations when items change
        self.ranked_allocations = self._rank_allocations_by_welfare()
        self.all_possible_allocations = self._generate_all_possible_allocations()
    
    def get_agent_items_context(self) -> str:
        """
        Get the context string showing items and values for this agent.
        
        Returns:
            str: Formatted string of items with this agent's values
        """
        if not self.current_items:
            return ""
        
        agent_value_key = f"agent{self.agent_id}Value"
        items_str = []
        
        for item in self.current_items:
            value = getattr(item, agent_value_key)
            items_str.append(f"{item.name}={value}")
        
        return ", ".join(items_str)
    
    def should_make_deterministic_proposal(self, turn_number: int) -> bool:
        """
        Fixed Price agents always make deterministic proposals.
        
        Args:
            turn_number: Current turn number
            
        Returns:
            bool: Always True for Fixed Price agents
        """
        return True
    
    def get_deterministic_proposal(self, turn_number: int) -> Optional[Dict[str, List[str]]]:
        """
        Get the deterministic proposal for the Fixed Price agent.
        Always uses the same fixed threshold-based allocation.
        
        Args:
            turn_number: Current turn number (ignored for Fixed Price)
            
        Returns:
            Dict: The proposed allocation
        """
        if not self.ranked_allocations:
            return None
        
        # Get proposal index based on fixed threshold
        proposal_index = self._get_fixed_price_proposal_index()
        
        # Store intended proposal for validation
        self.intended_proposal = self.ranked_allocations[proposal_index].copy()
        
        return self.intended_proposal
    
    def should_accept_proposal(self, proposal: ParsedProposal, turn_number: int) -> bool:
        """
        Determine if the Fixed Price agent should accept a proposal.
        Accepts if the proposal gives welfare at or above the fixed threshold.
        
        Args:
            proposal: The proposal to evaluate
            turn_number: Current turn number (ignored for Fixed Price)
            
        Returns:
            bool: True if proposal should be accepted
        """
        if not proposal or not proposal.is_valid:
            return False
        
        # Convert proposal to allocation format
        allocation = {
            "agent1": proposal.agent1_items,
            "agent2": proposal.agent2_items
        }
        
        # Calculate welfare for this proposal
        proposal_welfare = self._calculate_agent_welfare(allocation)
        
        # Get the welfare at our fixed threshold level
        if not self.ranked_allocations:
            return False
        
        threshold_index = self._get_fixed_price_proposal_index()
        threshold_allocation = self.ranked_allocations[threshold_index]
        threshold_welfare = self._calculate_agent_welfare(threshold_allocation)
        
        # Accept if proposal welfare is at or above our fixed threshold welfare
        return proposal_welfare >= threshold_welfare
    
    def validate_output_matches_intent(self, llm_output: str, intended_proposal: Dict[str, List[str]]) -> bool:
        """
        Validate that the LLM output matches the intended deterministic proposal.
        
        Args:
            llm_output: The LLM's actual output
            intended_proposal: The proposal we intended to make
            
        Returns:
            bool: True if the output matches the intent
        """
        from src.utils.MessageParser import MessageParser
        
        parser = MessageParser()
        parsed = parser.extract_proposal(llm_output, [item.name for item in self.current_items])
        
        if not parsed or not parsed.is_valid:
            return False
        
        # Check if parsed proposal matches intended proposal
        def normalize_list(items):
            return sorted(items) if items else []
        
        return (normalize_list(parsed.agent1_items) == normalize_list(intended_proposal.get("agent1", [])) and
                normalize_list(parsed.agent2_items) == normalize_list(intended_proposal.get("agent2", [])))
    
    def get_strategy_info(self) -> str:
        """
        Get information about the current strategy state.
        
        Returns:
            str: Strategy information for debugging
        """
        if not self.ranked_allocations:
            return "Fixed Price Agent (no allocations calculated yet)"
        
        proposal_index = self._get_fixed_price_proposal_index()
        proposal = self.ranked_allocations[proposal_index]
        proposal_welfare = self._calculate_agent_welfare(proposal)
        
        return f"Fixed Price Agent (threshold: {self.fixed_threshold:.2f}, target welfare: {proposal_welfare:.2f})"