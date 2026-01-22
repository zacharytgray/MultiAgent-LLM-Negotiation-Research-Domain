from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ParsedAction:
    action_type: str  # "OFFER", "ACCEPT", "INVALID"
    offer_content: Any  # Could be price (float) or items (Dict)
    raw_text: str

class BaseDomain(ABC):
    """
    Abstract base class for negotiation domains.
    Handles domain-specific logic, state, and rules.
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.is_terminal_state = False
    
    @abstractmethod
    def reset(self, round_id: int, **kwargs) -> Dict[str, Any]:
        """
        Reset the domain for a new round.
        Returns the initial domain state.
        """
        pass
    
    @abstractmethod
    def get_private_context(self, agent_id: int) -> Dict[str, Any]:
        """
        Get private information for a specific agent.
        """
        pass
    
    @abstractmethod
    def get_public_context(self) -> Dict[str, Any]:
        """
        Get public information shared by all agents.
        """
        pass
    
    @abstractmethod
    def parse_agent_action(self, agent_id: int, text: str) -> ParsedAction:
        """
        Parse raw text from an agent into a structured action.
        """
        pass
    
    @abstractmethod
    def format_agent_prompt_context(self, agent_id: int) -> str:
        """
        Get a text block to append to the agent's LLM prompt,
        explaining the domain rules and current state.
        """
        pass
    
    @abstractmethod
    def is_valid_action(self, action: ParsedAction) -> bool:
        """
        Check if the parsed action is valid in the current state.
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: ParsedAction, agent_id: int) -> bool:
        """
        Apply the action to the domain state.
        Returns True if the action caused a state change (or was valid),
        False otherwise.
        """
        pass
    
    @abstractmethod
    def is_agreement(self) -> bool:
        """
        Check if an agreement has been reached.
        """
        pass
    
    @abstractmethod
    def get_outcome(self) -> Dict[str, Any]:
        """
        Get the outcome of the negotiation (utilities, agreement details).
        """
        pass
