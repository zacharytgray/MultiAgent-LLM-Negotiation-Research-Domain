"""
Charming agent implementation - uses enthusiastic and persuasive communication style.
This agent has the same negotiation goals as the default agent but uses charming, 
sales-oriented language to make proposals sound irresistible.
"""

from typing import List, Optional
from src.core.Item import Item
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent


class CharmingAgent(BaseAgent):
    """
    Charming agent that uses enthusiastic and persuasive communication.
    Same underlying behavior as default agent but with charming personality.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str):
        """
        Initialize the charming agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
        """
        super().__init__(agent_id, model_name, system_instructions_file)
        # Create the underlying ollama agent
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
    
    async def generate_response(self) -> str:
        """
        Generate a response using the underlying ollama agent.
        
        Returns:
            str: The agent's response message
        """
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str):
        """
        Add a message to the agent's memory via the ollama agent.
        
        Args:
            role: Role of the message (system, user, assistant, tool)
            content: Content of the message
        """
        self.ollama_agent.addToMemory(role, content)
    
    def reset_memory(self):
        """
        Reset the agent's memory to initial state.
        """
        self.ollama_agent.reset_memory()