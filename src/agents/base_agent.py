"""
Base agent class that defines the interface for all negotiation agents.
All agent types inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from src.core.Item import Item
from src.utils.MessageParser import ParsedProposal


class BaseAgent(ABC):
    """
    Abstract base class for all negotiation agents.
    Defines the interface that all agents must implement.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.system_instructions_file = system_instructions_file
        self.current_items = []
        self.memory = []
        
    @abstractmethod
    async def generate_response(self) -> str:
        """
        Generate a response during negotiation.
        
        Returns:
            str: The agent's response message
        """
        pass
    
    @abstractmethod
    def add_to_memory(self, role: str, content: str):
        """
        Add a message to the agent's memory.
        
        Args:
            role: Role of the message (system, user, assistant)
            content: Content of the message
        """
        pass
    
    @abstractmethod
    def reset_memory(self):
        """
        Reset the agent's memory to initial state (system instructions only).
        """
        pass
    
    @abstractmethod
    def set_items(self, items: List[Item]):
        """
        Set the items for this negotiation round.
        
        Args:
            items: List of items with values for both agents
        """
        pass
    
    @abstractmethod
    def get_agent_items_context(self) -> str:
        """
        Get the context string showing items and values for this agent.
        
        Returns:
            str: Formatted string of items with this agent's values
        """
        pass
    
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        """
        Check if this agent should make a deterministic proposal.
        Override in deterministic agent types (like Boulware).
        
        Args:
            turn_number: Current turn number
        
        Returns:
            bool: True if agent should make deterministic proposal
        """
        return False
    
    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        """
        Get the deterministic proposal for this agent.
        Override in deterministic agent types.
        
        Args:
            turn_number: Current turn number
            
        Returns:
            Optional[Dict]: Deterministic allocation dict, or None if not applicable
        """
        return None
    
    def should_accept_proposal(self, proposal: ParsedProposal, turn_number: int = 1) -> bool:
        """
        Check if this agent should accept the given proposal.
        Override in deterministic agent types for custom acceptance logic.
        
        Args:
            proposal: The proposal to evaluate
            turn_number: Current turn number
            
        Returns:
            bool: True if agent should accept
        """
        return False
    
    def validate_output_matches_intent(self, response: str, intended_proposal: Optional[Dict]) -> bool:
        """
        Validate that the agent's output matches the intended deterministic proposal.
        Override in deterministic agent types.
        
        Args:
            response: The agent's response
            intended_proposal: The intended deterministic proposal
            
        Returns:
            bool: True if output matches intent
        """
        return True
    
    def update_strategy_state(self, turn_number: int):
        """
        Update the agent's internal strategy state based on turn number.
        Override in agents that have turn-based strategy changes.
        
        Args:
            turn_number: Current turn number in the negotiation
        """
        pass