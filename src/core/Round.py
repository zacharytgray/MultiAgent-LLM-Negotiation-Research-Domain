from typing import List, Dict, Optional, Tuple
from .Item import Item
from ..agents.ollamaAgentModule import Agent

class Round:
    """
    Manages a single round of negotiation between two agents.
    """
    def __init__(self, round_number: int, items: List[Item], agent1: Agent, agent2: Agent, starting_agent: int):
        self.round_number = round_number
        self.items = items
        self.agent1 = agent1
        self.agent2 = agent2
        self.starting_agent = starting_agent  # 1 or 2
        self.conversation_history = []
        self.final_allocation = None
        self.is_complete = False
        
    def get_items_for_agent(self, agent_num: int) -> str:
        """
        Return items visible to the specified agent (only their own values).
        """
        items_str = "["
        for item in self.items:
            if agent_num == 1:
                items_str += f"{item.name}(agent1Value={item.agent1Value}), "
            else:
                items_str += f"{item.name}(agent2Value={item.agent2Value}), "
        items_str = items_str.rstrip(", ") + "]"
        return items_str
    
    def detect_agreement(self, message1: str, message2: str) -> bool:
        """
        Check if both agents have agreed (both messages contain 'AGREE').
        """
        return "AGREE" in message1.upper() and "AGREE" in message2.upper()