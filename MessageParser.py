import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedProposal:
    """
    Data class to hold a parsed proposal with validation results.
    """
    agent1_items: List[str]
    agent2_items: List[str]
    is_valid: bool
    error_message: Optional[str] = None
    raw_json: Optional[Dict] = None

class MessageParser:
    """
    Parser for extracting formal proposals from agent messages.
    """
    
    def __init__(self):
        # Regex pattern to find PROPOSAL followed by JSON
        self.proposal_pattern = r'PROPOSAL\s*(\{[^}]*\})'
        
    def extract_proposal(self, message: str, available_items: List[str]) -> Optional[ParsedProposal]:
        """
        Extract and validate a formal proposal from an agent message.
        
        Args:
            message: The agent's message text
            available_items: List of item names available in this round
            
        Returns:
            ParsedProposal object if a valid proposal is found, None otherwise
        """
        # Find PROPOSAL keyword followed by JSON
        match = re.search(self.proposal_pattern, message, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
            
        json_str = match.group(1)
        
        try:
            # Parse the JSON
            proposal_data = json.loads(json_str)
            
            # Extract agent allocations
            agent1_items = proposal_data.get("agent1", [])
            agent2_items = proposal_data.get("agent2", [])
            
            # Validate the proposal
            is_valid, error_message = self._validate_proposal(
                agent1_items, agent2_items, available_items
            )
            
            return ParsedProposal(
                agent1_items=agent1_items,
                agent2_items=agent2_items,
                is_valid=is_valid,
                error_message=error_message,
                raw_json=proposal_data
            )
            
        except json.JSONDecodeError as e:
            return ParsedProposal(
                agent1_items=[],
                agent2_items=[],
                is_valid=False,
                error_message=f"Invalid JSON format: {str(e)}",
                raw_json=None
            )
    
    def _validate_proposal(self, agent1_items: List[str], agent2_items: List[str], 
                          available_items: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that a proposal is complete and correct.
        
        Args:
            agent1_items: Items proposed for agent1
            agent2_items: Items proposed for agent2  
            available_items: All items that should be allocated
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required keys
        if not isinstance(agent1_items, list) or not isinstance(agent2_items, list):
            return False, "Agent allocations must be lists"
        
        # Get all proposed items
        all_proposed = agent1_items + agent2_items
        
        # Check for duplicate items
        if len(all_proposed) != len(set(all_proposed)):
            duplicates = [item for item in set(all_proposed) if all_proposed.count(item) > 1]
            return False, f"Duplicate items found: {duplicates}"
        
        # Check for invalid items (not in available_items)
        invalid_items = [item for item in all_proposed if item not in available_items]
        if invalid_items:
            return False, f"Invalid items (not available this round): {invalid_items}"
        
        # Check for missing items
        missing_items = [item for item in available_items if item not in all_proposed]
        if missing_items:
            return False, f"Missing items (must allocate all items): {missing_items}"
        
        # Check for extra items
        if len(all_proposed) != len(available_items):
            return False, f"Proposal has {len(all_proposed)} items but round has {len(available_items)} items"
        
        return True, None
    
    def contains_agreement(self, message: str) -> bool:
        """
        Check if a message contains an agreement signal.
        
        Args:
            message: The agent's message text
            
        Returns:
            True if message contains "AGREE", False otherwise
        """
        return "AGREE" in message
    
    def extract_items_from_proposal(self, proposal: ParsedProposal, agent_num: int) -> List[str]:
        """
        Extract items allocated to a specific agent from a proposal.
        
        Args:
            proposal: ParsedProposal object
            agent_num: Agent number (1 or 2)
            
        Returns:
            List of item names allocated to the specified agent
        """
        if not proposal.is_valid:
            return []
            
        return proposal.agent1_items if agent_num == 1 else proposal.agent2_items
