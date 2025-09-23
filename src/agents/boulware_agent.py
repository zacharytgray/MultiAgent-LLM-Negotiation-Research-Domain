"""
Boulware agent implementation with deterministic strategy.
Uses the Boulware negotiation strategy with decreasing thresholds over time.
"""

import itertools
from typing import List, Optional, Dict, Tuple
from src.core.Item import Item
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent
from src.utils.MessageParser import ParsedProposal


class BoulwareAgent(BaseAgent):
    """
    Boulware agent that uses a deterministic strategy with decreasing thresholds.
    The LLM is used as a wrapper to present the deterministic proposals naturally.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str, 
                 initial_threshold: float = 0.80, use_tools: bool = False):
        """
        Initialize the Boulware agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            initial_threshold: Starting threshold percentage (default 0.80)
            use_tools: Whether to enable tool usage
        """
        super().__init__(agent_id, model_name, system_instructions_file)
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.use_tools = use_tools
        self.ranked_allocations = []
        self.all_possible_allocations = []
        self.intended_proposal = None
        
        # Create the underlying ollama agent with special system instructions
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file, use_tools)
    
    def _calculate_agent_welfare(self, allocation: Dict[str, List[str]]) -> float:
        """
        Calculate this agent's welfare for a given allocation.
        
        Args:
            allocation: Dictionary with agent1 and agent2 item lists
            
        Returns:
            float: Total welfare value for this agent
        """
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
            List[Dict]: All possible allocations
        """
        if not self.current_items:
            return []
        
        item_names = [item.name for item in self.current_items]
        num_items = len(item_names)
        all_allocations = []
        
        # Generate all possible combinations of items for agent1
        # Agent2 gets the remaining items
        for r in range(num_items + 1):
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
        Rank all possible allocations from lowest to highest welfare for this agent.
        
        Returns:
            List[Dict]: Sorted allocations (lowest welfare first, highest welfare last)
        """
        all_allocations = self._generate_all_possible_allocations()
        
        # Calculate welfare for each allocation and sort
        allocation_welfare_pairs = []
        for allocation in all_allocations:
            welfare = self._calculate_agent_welfare(allocation)
            allocation_welfare_pairs.append((allocation, welfare))
        
        # Sort by welfare (lowest first)
        allocation_welfare_pairs.sort(key=lambda x: x[1])
        
        # Return just the allocations
        return [pair[0] for pair in allocation_welfare_pairs]
    
    def _get_boulware_proposal_index(self) -> int:
        """
        Get the index in ranked_allocations based on current threshold.
        
        Returns:
            int: Index of the proposal to make
        """
        if not self.ranked_allocations:
            return 0
        
        # Calculate index based on threshold
        # threshold of 1.0 = best allocation (last index)
        # threshold of 0.0 = worst allocation (first index)
        max_index = len(self.ranked_allocations) - 1
        index = int(max_index * self.current_threshold)
        
        # Ensure index is within bounds
        return min(max(index, 0), max_index)
    
    def _decrease_threshold(self, turn_number: int):
        """
        Decrease threshold according to Boulware schedule.
        
        Args:
            turn_number: Current turn number
        """
        # Simple linear decrease - can be made more sophisticated
        # Decrease by 10% each turn, but don't go below 0.1
        decrease_rate = 0.05
        self.current_threshold = max(0.1, self.initial_threshold - (turn_number * decrease_rate))
    
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
        self.current_threshold = self.initial_threshold
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
    
    def should_make_deterministic_proposal(self) -> bool:
        """
        Boulware agents always make deterministic proposals.
        
        Returns:
            bool: Always True for Boulware agents
        """
        return True
    
    def get_deterministic_proposal(self, current_proposal: Optional[ParsedProposal] = None) -> Optional[Dict]:
        """
        Get the deterministic proposal based on Boulware strategy.
        
        Args:
            current_proposal: Current proposal on the table (if any)
            
        Returns:
            Optional[Dict]: Deterministic allocation dict
        """
        if not self.ranked_allocations:
            return None
        
        # Always use threshold-based proposal (whether initial or counter-proposal)
        proposal_index = self._get_boulware_proposal_index()
        self.intended_proposal = self.ranked_allocations[proposal_index]
        
        return self.intended_proposal
    
    def should_accept_proposal(self, proposal: ParsedProposal) -> bool:
        """
        Check if the Boulware agent should accept the given proposal.
        Accept if the proposal gives us welfare >= current threshold.
        
        Args:
            proposal: The proposal to evaluate
            
        Returns:
            bool: True if agent should accept
        """
        if not proposal or not proposal.allocation:
            return False
        
        proposal_welfare = self._calculate_agent_welfare(proposal.allocation)
        
        # Find the welfare that corresponds to our current threshold
        if not self.ranked_allocations:
            return False
        
        threshold_index = self._get_boulware_proposal_index()
        threshold_welfare = self._calculate_agent_welfare(self.ranked_allocations[threshold_index])
        
        return proposal_welfare >= threshold_welfare
    
    def validate_output_matches_intent(self, response: str, intended_proposal: Optional[Dict]) -> bool:
        """
        Validate that the agent's output contains the intended deterministic proposal.
        
        Args:
            response: The agent's response
            intended_proposal: The intended deterministic proposal
            
        Returns:
            bool: True if output matches intent
        """
        if not intended_proposal:
            return True  # No specific intent to validate
        
        # Use MessageParser to extract proposal from response
        from src.utils.MessageParser import MessageParser
        parser = MessageParser()
        item_names = [item.name for item in self.current_items]
        
        extracted_proposal = parser.extract_proposal(response, item_names)
        
        if not extracted_proposal or not extracted_proposal.allocation:
            return False
        
        # Check if extracted allocation matches intended allocation
        extracted_allocation = extracted_proposal.allocation
        
        # Compare agent allocations
        for agent_key in ["agent1", "agent2"]:
            intended_items = set(intended_proposal.get(agent_key, []))
            extracted_items = set(extracted_allocation.get(agent_key, []))
            
            if intended_items != extracted_items:
                return False
        
        return True
    
    def update_strategy_state(self, turn_number: int):
        """
        Update the Boulware agent's threshold based on turn number.
        
        Args:
            turn_number: Current turn number in the negotiation
        """
        self._decrease_threshold(turn_number)
    
    def get_strategy_info(self) -> Dict:
        """
        Get information about the current strategy state.
        
        Returns:
            Dict: Strategy information for debugging/logging
        """
        return {
            "agent_type": "boulware",
            "current_threshold": self.current_threshold,
            "initial_threshold": self.initial_threshold,
            "num_allocations": len(self.ranked_allocations),
            "intended_proposal": self.intended_proposal
        }