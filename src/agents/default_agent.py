"""
Default agent implementation - wraps the existing ollamaAgentModule for backward compatibility.
This is the standard negotiation agent that uses pure LLM responses.
"""

from typing import List, Optional
from src.core.Item import Item
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent


class DefaultAgent(BaseAgent):
    """
    Default agent that uses pure LLM responses without deterministic strategies.
    This wraps the existing ollamaAgentModule for backward compatibility.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str, use_tools: bool = False):
        """
        Initialize the default agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            use_tools: Whether to enable tool usage
        """
        super().__init__(agent_id, model_name, system_instructions_file)
        self.use_tools = use_tools
        
        # Create the underlying ollama agent
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file, use_tools)
    
    async def generate_response(self) -> str:
        """
        Generate a response using the underlying ollama agent.
        
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
        Reset the agent's memory to initial state.
        """
        self.ollama_agent.reset_memory()
    
    def set_items(self, items: List[Item]):
        """
        Set the items for this negotiation round.
        
        Args:
            items: List of items with values for both agents
        """
        self.current_items = items
    
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
    
    def print_memory(self, skip_system_message: bool = False):
        """
        Print the agent's conversation memory.
        
        Args:
            skip_system_message: Whether to skip printing system messages
        """
        self.ollama_agent.printMemory(skip_system_message)