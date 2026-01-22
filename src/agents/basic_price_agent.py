from typing import Optional, Dict
from src.agents.base_agent import BaseAgent
from src.agents.ollamaAgentModule import Agent as OllamaAgent

class BasicPriceAgent(BaseAgent):
    """
    Basic Price Agent.
    No deterministic backend. Pure LLM.
    Relies on system instructions to negotiate.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str):
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        
    async def generate_response(self) -> str:
        return await self.ollama_agent.generateResponse()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.addToMemory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()
        
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        return False
