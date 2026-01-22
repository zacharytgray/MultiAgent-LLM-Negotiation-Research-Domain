import random
from typing import Dict, Any, List, Optional
from .base_domain import BaseDomain, ParsedAction
from src.core.Item import Item
from src.utils.MessageParser import MessageParser, ParsedProposal
from config.settings import *

class MultiItemAllocationDomain(BaseDomain):
    """
    Domain for multi-item allocation negotiations.
    Uses existing logic from MessageParser and AllocationTracker style logic.
    """
    
    def __init__(self):
        super().__init__("multi_item_allocation")
        self.message_parser = MessageParser()
        self.items: List[Item] = []
        self.current_proposal: Optional[ParsedProposal] = None
        self.agreement_reached = False
        self.item_names = [f"Item_{i}" for i in range(10)] # Default pool
        
        # State tracking
        self.last_proposer_id: Optional[int] = None
        self.consecutive_agreements = 0

    def reset(self, round_id: int, **kwargs) -> Dict[str, Any]:
        """
        Reset for a new round.
        kwargs can contain: items_per_round, item_names
        """
        self.round_id = round_id
        items_per_round = kwargs.get("items_per_round", DEFAULT_ITEMS_PER_ROUND)
        self.item_names = kwargs.get("item_names", ITEM_NAMES)
        
        self.items = self._generate_random_items(items_per_round)
        self.current_proposal = None
        self.agreement_reached = False
        self.consecutive_agreements = 0
        self.last_proposer_id = None
        self.is_terminal_state = False
        
        return {
            "items": self.items
        }

    def _generate_random_items(self, count: int) -> List[Item]:
        items = []
        for i in range(count):
            name = self.item_names[i] if i < len(self.item_names) else f"Item{i+1}"
            a1_val = round(random.uniform(MIN_ITEM_VALUE, MAX_ITEM_VALUE), ITEM_VALUE_PRECISION)
            a2_val = round(random.uniform(MIN_ITEM_VALUE, MAX_ITEM_VALUE), ITEM_VALUE_PRECISION)
            items.append(Item(name, a1_val, a2_val))
        return items

    def get_private_context(self, agent_id: int) -> Dict[str, Any]:
        # Filter items to only show one agent's values
        # Kept compatible with BaseAgent.set_items expectation
        return {"items": self.items}

    def get_public_context(self) -> Dict[str, Any]:
        return {
            "item_names": [i.name for i in self.items],
            "total_items": len(self.items)
        }

    def parse_agent_action(self, agent_id: int, text: str) -> ParsedAction:
        # Check for PROPOSAL first
        available_item_names = [i.name for i in self.items]
        proposal = self.message_parser.extract_proposal(text, available_item_names)
        
        if proposal:
            return ParsedAction(
                action_type="OFFER",
                offer_content=proposal,
                raw_text=text
            )
        
        # Check for AGREEMENT
        if "AGREE" in text.upper():
             return ParsedAction(
                action_type="ACCEPT",
                offer_content=None,
                raw_text=text
            )
            
        return ParsedAction(
            action_type="INVALID",
            offer_content=None,
            raw_text=text
        )

    def is_valid_action(self, action: ParsedAction) -> bool:
        if action.action_type == "OFFER":
             return action.offer_content.is_valid
        if action.action_type == "ACCEPT":
             # Can only accept if there is a valid proposal on the table
             return self.current_proposal is not None and self.current_proposal.is_valid
        return False

    def apply_action(self, action: ParsedAction, agent_id: int) -> bool:
        if not self.is_valid_action(action):
            return False

        if action.action_type == "OFFER":
            self.current_proposal = action.offer_content
            self.last_proposer_id = agent_id
            self.consecutive_agreements = 0 # Reset on new offer
            
            # Implicit agreement if the proposer proposes
            # In some protocols, proposing counts as agreeing to your own proposal.
            # But here, we wait for the other to agree. 
            pass

        elif action.action_type == "ACCEPT":
            # If the other person proposed it, and I accept, we have agreement.
            # However, the original logic checked for "AGREE" in both messages or consecutive agreements.
            # Original logic: "AGREE" in message1 and "AGREE" in message2 (Round.detect_agreement)
            # OR AllocationTracker.is_round_complete -> agreement_count >= 2
            
            if self.current_proposal:
                 self.consecutive_agreements += 1
                 if self.consecutive_agreements >= 1 and self.last_proposer_id != agent_id:
                     # Agent accepted other's proposal
                     self.agreement_reached = True
                     self.is_terminal_state = True
            
        return True

    def is_agreement(self) -> bool:
        return self.agreement_reached

    def get_outcome(self) -> Dict[str, Any]:
        if not self.agreement_reached or not self.current_proposal:
            return {
                "agreement": False,
                "agent1_utility": 0.0,
                "agent2_utility": 0.0,
                "final_allocation": None
            }
        
        alloc = self.current_proposal.allocation
        
        # Calculate utilities
        u1 = sum(item.agent1Value for item in self.items if item.name in alloc['agent1'])
        u2 = sum(item.agent2Value for item in self.items if item.name in alloc['agent2'])
        
        return {
            "agreement": True,
            "agent1_utility": u1,
            "agent2_utility": u2,
            "final_allocation": alloc
        }
    
    def format_agent_prompt_context(self, agent_id: int) -> str:
        # This duplicates some logic from Negotiation._prepare_agent_contexts but puts it in the domain
        # However, the agent's set_items method handles the main formatting. 
        # Here we just add the closing instruction.
        return "\nWhen you reach an agreement, end your message with \"AGREE\"."
