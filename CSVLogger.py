import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from Item import Item
from Round import Round
from Scoring.ParetoAnalyzer import ParetoAnalyzer

@dataclass
class NegotiationLogEntry:
    """
    Comprehensive data structure for a single negotiation round log entry.
    """
    # Session metadata
    session_id: str
    model_name: str
    num_items: int
    timestamp: str
    date: str
    
    # Round identification
    round_number: int
    total_rounds: int
    
    # Round timing and interaction metrics
    round_duration_seconds: float
    turn_count: int
    max_turns: int
    starting_agent: int
    round_completed: bool
    agreement_reached: bool
    
    # Item details (flattened for CSV)
    agent1_values: str  # comma-separated
    agent2_values: str  # comma-separated
    total_value_pool: float
    
    # Final allocation
    agent1_items: str  # comma-separated
    agent2_items: str  # comma-separated
    agent1_final_value: float
    agent2_final_value: float
    total_welfare: float
    
    # Pareto analysis
    is_pareto_optimal: bool
    welfare_efficiency: float
    max_possible_welfare: float
    welfare_gap: float
    pareto_optimal_count: int
    unique_pareto_combinations: int
    
    # Negotiation dynamics
    proposal_count: int
    valid_proposal_count: int
    invalid_proposal_count: int
    agent1_proposal_count: int
    agent2_proposal_count: int
    final_proposer: Optional[int]

class CSVLogger:
    """
    Handles CSV logging for negotiation sessions with comprehensive data capture.
    """
    
    def __init__(self, model_name: str, num_items: int, base_dir: str = "logs"):
        self.model_name = model_name
        self.num_items = num_items
        self.base_dir = base_dir
        self.session_id = self._generate_session_id()
        self.filename = self._generate_filename()
        self.filepath = os.path.join(base_dir, self.filename)
        
        # Ensure logs directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Track if header has been written
        self.header_written = False
        
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return f"{self.model_name}_{self.num_items}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_filename(self) -> str:
        """Generate filename following MODELNAME_NUMITEMS_DATE.csv convention"""
        date_str = datetime.now().strftime('%Y%m%d')
        return f"{self.model_name}_{self.num_items}_{date_str}.csv"
    
    def create_log_entry(self, 
                        round_obj: Round, 
                        round_duration: float,
                        final_allocation: Dict[str, List[str]],
                        allocation_tracker: Any,
                        total_rounds: int) -> NegotiationLogEntry:
        """
        Create a comprehensive log entry for a negotiation round.
        """
        # Analyze the allocation
        analysis = ParetoAnalyzer.analyze_allocation_efficiency(round_obj.items, final_allocation)
        
        # Calculate max possible values for each agent
        agent1_max_possible = sum(item.agent1Value for item in round_obj.items)
        agent2_max_possible = sum(item.agent2Value for item in round_obj.items)
        
        # Get proposal statistics from allocation tracker
        round_state = allocation_tracker.round_states.get(round_obj.round_number)
        proposal_count = len(round_state.proposal_history) if round_state else 0
        
        # Extract valid and invalid proposals from the proposal history
        valid_proposals = []
        invalid_proposals = []
        agent1_proposals = []
        agent2_proposals = []
        
        if round_state and round_state.proposal_history:
            for agent_num, proposal in round_state.proposal_history:
                if proposal.is_valid:
                    valid_proposals.append((agent_num, proposal))
                    if agent_num == 1:
                        agent1_proposals.append((agent_num, proposal))
                    else:
                        agent2_proposals.append((agent_num, proposal))
                else:
                    invalid_proposals.append((agent_num, proposal))
        
        final_proposer = None
        if round_state and round_state.proposal_history:
            for agent_num, proposal in reversed(round_state.proposal_history):
                if proposal.is_valid:
                    final_proposer = agent_num
                    break
        
        # Create the log entry
        entry = NegotiationLogEntry(
            # Session metadata
            session_id=self.session_id,
            model_name=self.model_name,
            num_items=self.num_items,
            timestamp=datetime.now().isoformat(),
            date=datetime.now().strftime('%Y-%m-%d'),
            
            # Round identification
            round_number=round_obj.round_number,
            total_rounds=total_rounds,
            
            # Round timing and interaction metrics
            round_duration_seconds=round_duration,
            turn_count=len(round_obj.conversation_history),
            max_turns=20,  # TODO: Make this configurable
            starting_agent=round_obj.starting_agent,
            round_completed=round_obj.is_complete,
            agreement_reached=round_obj.is_complete,
            
            # Item details
            agent1_values=",".join([str(item.agent1Value) for item in round_obj.items]),
            agent2_values=",".join([str(item.agent2Value) for item in round_obj.items]),
            total_value_pool=agent1_max_possible + agent2_max_possible,
            
            # Final allocation
            agent1_items=",".join(final_allocation.get('agent1', [])),
            agent2_items=",".join(final_allocation.get('agent2', [])),
            agent1_final_value=analysis['agent1_value'],
            agent2_final_value=analysis['agent2_value'],
            total_welfare=analysis['total_welfare'],
            
            # Pareto analysis
            is_pareto_optimal=analysis['is_pareto_optimal'],
            welfare_efficiency=analysis['welfare_efficiency'],
            max_possible_welfare=analysis['max_possible_welfare'],
            welfare_gap=analysis['max_possible_welfare'] - analysis['total_welfare'],
            pareto_optimal_count=analysis['pareto_optimal_count'],
            unique_pareto_combinations=analysis['unique_pareto_optimal_count'],
            
            # Negotiation dynamics
            proposal_count=proposal_count,
            valid_proposal_count=len(valid_proposals),
            invalid_proposal_count=len(invalid_proposals),
            agent1_proposal_count=len(agent1_proposals),
            agent2_proposal_count=len(agent2_proposals),
            final_proposer=final_proposer
        )
        
        return entry
    
    def log_round(self, entry: NegotiationLogEntry):
        """
        Write a round entry to the CSV file.
        """
        # Convert dataclass to dictionary
        entry_dict = asdict(entry)
        
        # Write header if this is the first entry
        file_exists = os.path.exists(self.filepath)
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(entry_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new or empty
            if not file_exists or os.path.getsize(self.filepath) == 0:
                writer.writeheader()
                
            writer.writerow(entry_dict)
    
    def get_filepath(self) -> str:
        """Return the full path to the CSV file."""
        return self.filepath
    
    def get_filename(self) -> str:
        """Return just the filename."""
        return self.filename