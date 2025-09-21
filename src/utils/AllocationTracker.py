from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .MessageParser import ParsedProposal

@dataclass 
class ProposalState:
    """
    Tracks the state of proposals in a negotiation round.
    """
    current_proposal: Optional[ParsedProposal] = None
    proposal_history: List[Tuple[int, ParsedProposal]] = None  # (agent_num, proposal)
    agreement_count: int = 0  # Number of consecutive agreements
    last_proposer: Optional[int] = None  # Which agent made the last proposal
    
    def __post_init__(self):
        if self.proposal_history is None:
            self.proposal_history = []

class AllocationTracker:
    """
    Tracks allocation proposals and agreement status across negotiation rounds.
    """
    
    def __init__(self):
        self.round_states: Dict[int, ProposalState] = {}
    
    def initialize_round(self, round_number: int):
        """
        Initialize tracking for a new round.
        
        Args:
            round_number: The round number to initialize
        """
        self.round_states[round_number] = ProposalState()
    
    def update_proposal(self, round_number: int, agent_num: int, proposal: ParsedProposal):
        """
        Update the current proposal for a round.
        
        Args:
            round_number: The round number
            agent_num: The agent making the proposal (1 or 2)
            proposal: The parsed proposal
        """
        if round_number not in self.round_states:
            self.initialize_round(round_number)
        
        state = self.round_states[round_number]
        
        # Store the proposal
        state.current_proposal = proposal
        state.proposal_history.append((agent_num, proposal))
        state.last_proposer = agent_num
        
        # Reset agreement count when new proposal is made
        state.agreement_count = 0
    
    def record_agreement(self, round_number: int, agent_num: int):
        """
        Record that an agent agreed to the current proposal.
        
        Args:
            round_number: The round number
            agent_num: The agent who agreed (1 or 2)
        """
        if round_number not in self.round_states:
            self.initialize_round(round_number)
        
        state = self.round_states[round_number]
        state.agreement_count += 1
    
    def is_round_complete(self, round_number: int) -> bool:
        """
        Check if a round is complete (both agents agreed to current proposal).
        
        Args:
            round_number: The round number to check
            
        Returns:
            True if round is complete, False otherwise
        """
        if round_number not in self.round_states:
            return False
        
        state = self.round_states[round_number]
        return (state.current_proposal is not None and 
                state.current_proposal.is_valid and 
                state.agreement_count >= 2)
    
    def get_current_proposal(self, round_number: int) -> Optional[ParsedProposal]:
        """
        Get the current proposal for a round.
        
        Args:
            round_number: The round number
            
        Returns:
            Current proposal or None if no valid proposal exists
        """
        if round_number not in self.round_states:
            return None
        
        return self.round_states[round_number].current_proposal
    
    def get_final_allocation(self, round_number: int) -> Optional[Dict[str, List[str]]]:
        """
        Get the final allocation for a completed round.
        
        Args:
            round_number: The round number
            
        Returns:
            Dictionary with agent allocations or None if round not complete
        """
        if not self.is_round_complete(round_number):
            return None
        
        proposal = self.get_current_proposal(round_number)
        if not proposal or not proposal.is_valid:
            return None
        
        return {
            "agent1": proposal.agent1_items,
            "agent2": proposal.agent2_items
        }
    
    def get_proposal_history(self, round_number: int) -> List[Tuple[int, ParsedProposal]]:
        """
        Get the history of proposals for a round.
        
        Args:
            round_number: The round number
            
        Returns:
            List of (agent_num, proposal) tuples
        """
        if round_number not in self.round_states:
            return []
        
        return self.round_states[round_number].proposal_history.copy()
    
    def get_round_summary(self, round_number: int) -> Dict:
        """
        Get a summary of the round state.
        
        Args:
            round_number: The round number
            
        Returns:
            Dictionary with round summary information
        """
        if round_number not in self.round_states:
            return {"status": "not_initialized"}
        
        state = self.round_states[round_number]
        
        return {
            "status": "complete" if self.is_round_complete(round_number) else "in_progress",
            "current_proposal": state.current_proposal,
            "proposal_count": len(state.proposal_history),
            "agreement_count": state.agreement_count,
            "last_proposer": state.last_proposer,
            "final_allocation": self.get_final_allocation(round_number)
        }
    
    def reset_agreement_count(self, round_number: int):
        """
        Reset the agreement count for a round (used when new proposal is made).
        
        Args:
            round_number: The round number
        """
        if round_number in self.round_states:
            self.round_states[round_number].agreement_count = 0
