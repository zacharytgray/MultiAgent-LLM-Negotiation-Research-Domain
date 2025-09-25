"""
Agent factory for creating different types of negotiation agents.
Supports easy addition of new agent types and configuration.
"""

from typing import Dict, Any, Type, Optional
from src.agents.base_agent import BaseAgent
from src.agents.default_agent import DefaultAgent
from src.agents.boulware_agent import BoulwareAgent
from src.agents.fixed_price_agent import FixedPriceAgent
from src.agents.charming_agent import CharmingAgent
from src.agents.rude_agent import RudeAgent
from config.settings import BOULWARE_INITIAL_THRESHOLD, BOULWARE_DECREASE_RATE, BOULWARE_MIN_THRESHOLD


class AgentFactory:
    """
    Factory class for creating different types of negotiation agents.
    """
    
    # Registry of available agent types
    AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
        "default": DefaultAgent,
        "boulware": BoulwareAgent,
        "fixed_price": FixedPriceAgent,
        "charming": CharmingAgent,
        "rude": RudeAgent,
    }
    
    @classmethod
    def create_agent(cls, agent_type: str, agent_id: int, model_name: str, 
                    system_instructions_file: str, **kwargs) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create ("default", "boulware", etc.)
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            **kwargs: Additional parameters specific to the agent type (tool-related parameters ignored)
            
        Returns:
            BaseAgent: Instance of the specified agent type
            
        Raises:
            ValueError: If agent_type is not recognized
        """
        if agent_type not in cls.AGENT_TYPES:
            available_types = ", ".join(cls.AGENT_TYPES.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Available types: {available_types}")
        
        agent_class = cls.AGENT_TYPES[agent_type]
        return agent_class(agent_id, model_name, system_instructions_file, **kwargs)
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]):
        """
        Register a new agent type with the factory.
        
        Args:
            agent_type: Name of the agent type
            agent_class: Class that implements the agent
        """
        cls.AGENT_TYPES[agent_type] = agent_class
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available agent types.
        
        Returns:
            list: List of available agent type names
        """
        return list(cls.AGENT_TYPES.keys())
    
    @classmethod
    def get_agent_class(cls, agent_type: str) -> Type[BaseAgent]:
        """
        Get the agent class for a given type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Type[BaseAgent]: The agent class
            
        Raises:
            ValueError: If agent_type is not recognized
        """
        if agent_type not in cls.AGENT_TYPES:
            available_types = ", ".join(cls.AGENT_TYPES.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Available types: {available_types}")
        
        return cls.AGENT_TYPES[agent_type]


# Configuration helpers for different agent types
class AgentConfig:
    """
    Configuration helper for different agent types.
    """
    
    @staticmethod
    def default_config() -> Dict[str, Any]:
        """
        Get default configuration for default agents.
        
        Returns:
            Dict: Configuration parameters
        """
        return {}
    
    @staticmethod
    def boulware_config(initial_threshold: Optional[float] = None, 
                       decrease_rate: Optional[float] = None,
                       min_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Get configuration for Boulware agents.
        
        Args:
            initial_threshold: Starting threshold percentage (defaults to settings value)
            decrease_rate: Amount to decrease threshold per turn (defaults to settings value)
            min_threshold: Minimum threshold value (defaults to settings value)
            
        Returns:
            Dict: Configuration parameters
        """
        return {
            "initial_threshold": initial_threshold if initial_threshold is not None else BOULWARE_INITIAL_THRESHOLD,
            "decrease_rate": decrease_rate if decrease_rate is not None else BOULWARE_DECREASE_RATE,
            "min_threshold": min_threshold if min_threshold is not None else BOULWARE_MIN_THRESHOLD,
        }
    
    @staticmethod
    def fixed_price_config(fixed_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Get configuration for Fixed Price agents.
        
        Args:
            fixed_threshold: Fixed threshold percentage (defaults to Boulware initial threshold)
            
        Returns:
            Dict: Configuration parameters
        """
        return {
            "fixed_threshold": fixed_threshold if fixed_threshold is not None else BOULWARE_INITIAL_THRESHOLD,
        }
    
    @staticmethod
    def charming_config() -> Dict[str, Any]:
        """
        Get configuration for Charming agents.
        
            
        Returns:
            Dict: Configuration parameters
        """
        return {}
    
    @staticmethod
    def rude_config() -> Dict[str, Any]:
        """
        Get configuration for Rude agents.
        
            
        Returns:
            Dict: Configuration parameters
        """
        return {}
    
    @staticmethod
    def get_config_for_type(agent_type: str, **kwargs) -> Dict[str, Any]:
        """
        Get appropriate configuration for the given agent type.
        
        Args:
            agent_type: Type of agent
            **kwargs: Override parameters
            
        Returns:
            Dict: Configuration parameters
        """
        if agent_type == "default":
            config = AgentConfig.default_config()
        elif agent_type == "boulware":
            config = AgentConfig.boulware_config()
        elif agent_type == "fixed_price":
            config = AgentConfig.fixed_price_config()
        elif agent_type == "charming":
            config = AgentConfig.charming_config()
        elif agent_type == "rude":
            config = AgentConfig.rude_config()
        else:
            config = {}
        
        # Remove tool-related parameters from kwargs
        kwargs.pop("use_tools", None)
        config.update(kwargs)
        return config