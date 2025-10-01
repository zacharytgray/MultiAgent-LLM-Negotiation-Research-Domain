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
from config.settings import BOULWARE_INITIAL_THRESHOLD, BOULWARE_MIN_THRESHOLD


class BoulwareAgent(BaseAgent):
    """
    Boulware agent that uses a deterministic strategy with decreasing thresholds.
    The LLM is used as a wrapper to present the deterministic proposals naturally.
    Uses a tiered utility-based approach for smoother concession curves.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str, 
                 initial_threshold: Optional[float] = None, 
                 min_threshold: Optional[float] = None):
        """
        Initialize the Boulware agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            initial_threshold: Starting threshold percentage (defaults to settings.BOULWARE_INITIAL_THRESHOLD)
            min_threshold: Minimum threshold value (defaults to settings.BOULWARE_MIN_THRESHOLD)
        """
        super().__init__(agent_id, model_name, system_instructions_file)
        self.initial_threshold = initial_threshold if initial_threshold is not None else BOULWARE_INITIAL_THRESHOLD
        self.min_threshold = min_threshold if min_threshold is not None else BOULWARE_MIN_THRESHOLD
        self.current_threshold = self.initial_threshold
        self.ranked_allocations = []
        self.all_possible_allocations = []
        self.intended_proposal = None
        self.utility_tiers = {}  # Maps normalized utility values to lists of allocations
        self.sorted_utility_values = []  # Sorted list of utility values for binary search
        self.max_welfare = 0.0  # Maximum possible welfare (for normalization)
        
        # Create the underlying ollama agent with special system instructions
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
    
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
        Also builds utility tiers dictionary grouping allocations by normalized utility.
        
        Returns:
            List[Dict]: Sorted allocations (lowest welfare first, highest welfare last)
        """
        all_allocations = self._generate_all_possible_allocations()
        
        # Calculate welfare for each allocation
        allocation_welfare_pairs = []
        for allocation in all_allocations:
            welfare = self._calculate_agent_welfare(allocation)
            allocation_welfare_pairs.append((allocation, welfare))
        
        # Find max welfare for normalization
        self.max_welfare = max([welfare for _, welfare in allocation_welfare_pairs]) if allocation_welfare_pairs else 1.0
        
        # Build utility tiers dictionary grouping allocations by normalized utility
        self.utility_tiers = {}
        for allocation, welfare in allocation_welfare_pairs:
            # Normalize utility to [0.0, 1.0]
            normalized_utility = welfare / self.max_welfare if self.max_welfare > 0 else 0
            # Round to 2 decimal places for tiering
            rounded_utility = round(normalized_utility, 2)
            
            if rounded_utility not in self.utility_tiers:
                self.utility_tiers[rounded_utility] = []
            self.utility_tiers[rounded_utility].append(allocation)
        
        # Create sorted list of utility values for binary search
        self.sorted_utility_values = sorted(self.utility_tiers.keys())
        
        # Sort by welfare (lowest first) and return traditional ranked list for compatibility
        allocation_welfare_pairs.sort(key=lambda x: x[1])
        return [pair[0] for pair in allocation_welfare_pairs]
    
    def _find_nearest_utility_tier(self, target_utility: float) -> float:
        """
        Use binary search to find the nearest utility tier that is greater than or equal to the target.
        Always rounds up to ensure we meet minimum utility requirements.
        
        Args:
            target_utility: The target utility value to search for
            
        Returns:
            float: The nearest utility tier value (rounded up)
        """
        if not self.sorted_utility_values:
            return 0.0
            
        # If target is higher than our max utility, return the max
        if target_utility >= self.sorted_utility_values[-1]:
            return self.sorted_utility_values[-1]
            
        # If target is lower than our min utility, return the min
        if target_utility <= self.sorted_utility_values[0]:
            return self.sorted_utility_values[0]
            
        # Binary search to find insertion point
        left, right = 0, len(self.sorted_utility_values) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_utility_values[mid] < target_utility:
                left = mid + 1
            elif self.sorted_utility_values[mid] > target_utility:
                right = mid - 1
            else:
                return self.sorted_utility_values[mid]  # Exact match
        
        # At this point, left is the insertion point
        # Return the utility tier at or just above the target
        return self.sorted_utility_values[left]
        
    def _get_boulware_proposal_index(self) -> int:
        """
        Get the index in ranked_allocations based on current threshold.
        Kept for backward compatibility but no longer used directly.
        
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
        Decrease threshold according to a true Boulware (nonlinear) schedule.
        Starts off firm, then decreases more rapidly as rounds progress.
        Scales to MAX_TURNS_PER_ROUND from settings.py.
        Args:
            turn_number: Current turn number (0-based)
        """
        from config.settings import MAX_TURNS_PER_ROUND
        # Boulware formula: threshold = min + (initial - min) * (1 - (t/T)^e)
        # where t = current turn, T = max turns, e = boulware exponent (large = more boulware)
        e = 3  # Exponent for Boulware curve; can be made configurable
        T = MAX_TURNS_PER_ROUND
        t = min(turn_number, T)
        initial = self.initial_threshold
        min_thr = self.min_threshold
        # Boulware curve: slow at first, then drops quickly
        boulware_value = min_thr + (initial - min_thr) * (1 - pow(t / T, e))
        self.current_threshold = max(min_thr, min(initial, boulware_value))
    
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
        self.utility_tiers = {}
        self.sorted_utility_values = []
        self.max_welfare = 0.0
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
    
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        """
        Boulware agents always make deterministic proposals.
        
        Args:
            turn_number: Current turn number (not used, but required for interface)
        
        Returns:
            bool: Always True for Boulware agents
        """
        return True
    
    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        """
        Get the deterministic proposal based on Boulware strategy using utility tiers.
        
        Args:
            turn_number: Current turn number in the negotiation
            
        Returns:
            Optional[Dict]: Deterministic allocation dict
        """
        if not self.utility_tiers or not self.sorted_utility_values:
            return None
        
        # Update threshold based on turn number
        self._decrease_threshold(turn_number)
        
        # Convert threshold to target utility
        target_utility = self.current_threshold
        
        # Find the nearest utility tier that meets or exceeds our target
        nearest_tier = self._find_nearest_utility_tier(target_utility)
        
        # Get the first allocation from that tier
        if nearest_tier in self.utility_tiers and self.utility_tiers[nearest_tier]:
            self.intended_proposal = self.utility_tiers[nearest_tier][0]
            return self.intended_proposal
        
        # Fallback to old method if something went wrong
        if self.ranked_allocations:
            proposal_index = self._get_boulware_proposal_index()
            self.intended_proposal = self.ranked_allocations[proposal_index]
            return self.intended_proposal
            
        return None
    
    def should_accept_proposal(self, proposal: ParsedProposal, turn_number: int) -> bool:
        """
        Check if the Boulware agent should accept the given proposal.
        Accept if the proposal gives us normalized utility >= current threshold.
        
        Args:
            proposal: The proposal to evaluate
            turn_number: Current turn number
            
        Returns:
            bool: True if agent should accept
        """
        if not proposal or not proposal.allocation:
            return False
        
        # Update threshold based on turn number
        self._decrease_threshold(turn_number)
        
        # Calculate welfare and normalize
        proposal_welfare = self._calculate_agent_welfare(proposal.allocation)
        proposal_utility = proposal_welfare / self.max_welfare if self.max_welfare > 0 else 0
        
        # Find the utility tier that corresponds to our current threshold
        target_tier = self._find_nearest_utility_tier(self.current_threshold)
        
        # Accept if proposal utility meets or exceeds our current threshold-based tier
        return proposal_utility >= target_tier
    
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
            "min_threshold": self.min_threshold,
            "max_welfare": self.max_welfare,
            "num_allocations": len(self.ranked_allocations),
            "num_utility_tiers": len(self.utility_tiers),
            "utility_range": [min(self.sorted_utility_values), max(self.sorted_utility_values)] if self.sorted_utility_values else [0, 0],
            "intended_proposal": self.intended_proposal
        }