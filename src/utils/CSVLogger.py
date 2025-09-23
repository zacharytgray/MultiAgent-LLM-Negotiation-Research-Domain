import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from ..core.Item import Item
from ..core.Round import Round

# Import configuration constants
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import CSV_FILENAME_TIMESTAMP_FORMAT, DEFAULT_LOG_DIR

@dataclass
class RawNegotiationLogEntry:
    """
    Raw data structure for a single negotiation round log entry.
    Contains only raw data.
    """
    # Session metadata
    session_id: str
    model_name: str
    num_items: int
    timestamp: str
    date: str
    
    # Agent types
    agent1_type: str
    agent2_type: str
    
    # Round identification
    round_number: int
    total_rounds: int
    
    # Round timing and interaction metrics
    round_duration_seconds: float
    turn_count: int
    starting_agent: int
    round_completed: bool
    agreement_reached: bool
    
    # Raw item data (JSON strings for complex data)
    items_data: str  # JSON: [{"name": "ItemA", "agent1_value": 0.8, "agent2_value": 0.3}, ...]
    
    # Final allocation
    final_allocation: str  # JSON: {"agent1": ["ItemA", "ItemB"], "agent2": ["ItemC", "ItemD"]}
    
    # Conversation history
    conversation_history: str  # JSON: [(agent_num, message), ...]
    
    # Negotiation dynamics
    proposal_history: str  # JSON: [(agent_num, proposal_data), ...]
    final_proposer: Optional[int]

class CSVLogger:
    """
    Handles CSV logging for negotiation sessions with comprehensive data capture.
    """
    
    def __init__(self, model_name: str, num_items: int, base_dir: str = DEFAULT_LOG_DIR):
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
        """Generate filename following MODELNAME_NUMITEMS_YYYYMMDD_HHMM.csv convention"""
        timestamp_str = datetime.now().strftime(CSV_FILENAME_TIMESTAMP_FORMAT)
        return f"{self.model_name}_{self.num_items}_{timestamp_str}.csv"
    
    def create_log_entry(self, 
                        round_obj: Round, 
                        round_duration: float,
                        final_allocation: Dict[str, List[str]],
                        allocation_tracker: Any,
                        total_rounds: int,
                        agent1_type: str = "unknown",
                        agent2_type: str = "unknown") -> RawNegotiationLogEntry:
        """
        Create a raw log entry for a negotiation round without analysis metrics.
        """
        # Serialize items data
        items_data = [
            {
                "name": item.name,
                "agent1_value": item.agent1Value,
                "agent2_value": item.agent2Value
            }
            for item in round_obj.items
        ]
        
        # Get proposal statistics from allocation tracker
        round_state = allocation_tracker.round_states.get(round_obj.round_number)
        proposal_history = []
        final_proposer = None
        
        if round_state and round_state.proposal_history:
            # Store proposal history as structured data
            proposal_history = [
                {
                    "agent_num": agent_num,
                    "proposal": {
                        "agent1_items": proposal.agent1_items,
                        "agent2_items": proposal.agent2_items,
                        "is_valid": proposal.is_valid,
                        "error_message": proposal.error_message  # Changed from validation_message
                    }
                }
                for agent_num, proposal in round_state.proposal_history
            ]
            
            # Find final proposer
            for agent_num, proposal in reversed(round_state.proposal_history):
                if proposal.is_valid:
                    final_proposer = agent_num
                    break
        
        # Create the raw log entry
        entry = RawNegotiationLogEntry(
            # Session metadata
            session_id=self.session_id,
            model_name=self.model_name,
            num_items=self.num_items,
            timestamp=datetime.now().isoformat(),
            date=datetime.now().strftime('%Y-%m-%d'),
            
            # Agent types
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            
            # Round identification
            round_number=round_obj.round_number,
            total_rounds=total_rounds,
            
            # Round timing and interaction metrics
            round_duration_seconds=round_duration,
            turn_count=len(round_obj.conversation_history),
            starting_agent=round_obj.starting_agent,
            round_completed=round_obj.is_complete,
            agreement_reached=round_obj.is_complete,
            
            # Raw data as JSON strings
            items_data=json.dumps(items_data),
            final_allocation=json.dumps(final_allocation),
            conversation_history=json.dumps(round_obj.conversation_history),
            proposal_history=json.dumps(proposal_history),
            final_proposer=final_proposer
        )
        
        return entry
    
    def log_round(self, entry: RawNegotiationLogEntry):
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