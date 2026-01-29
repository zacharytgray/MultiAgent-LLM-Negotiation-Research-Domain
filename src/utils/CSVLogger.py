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
    
    # Boulware agent parameters (if applicable)
    boulware_initial_threshold: Optional[float]
    boulware_decrease_rate: Optional[float] 
    boulware_min_threshold: Optional[float]
    boulware_final_threshold: Optional[float]
    
    # Fixed Price agent parameters (if applicable)
    fixed_price_threshold: Optional[float]
    
    # Price Domain Agent parameter
    price_fixed_margin: Optional[float]
    price_boulware_beta: Optional[float]
    
    # Round identification
    round_number: int
    total_rounds: int
    
    # Round timing and interaction metrics
    round_duration_seconds: float
    turn_count: int
    starting_agent: int
    round_completed: bool
    agreement_reached: bool
    reached_consensus: bool
    
    # Raw item data (JSON strings for complex data)
    items_data: str  # JSON: [{"name": "ItemA", "agent1_value": 0.8, "agent2_value": 0.3}, ...]
    
    # Final allocation
    final_allocation: str  # JSON: {"agent1": ["ItemA", "ItemB"], "agent2": ["ItemC", "ItemD"]}
    
    # Conversation history
    conversation_history: str  # JSON: [(agent_num, message), ...]
    
    # Negotiation dynamics
    proposal_history: str  # JSON: [(agent_num, proposal_data), ...]
    final_proposer: Optional[int]
    
    # Domain fields
    domain_name: str = "multi_item"
    domain_public_context: str = ""
    agent1_private_context: str = ""
    agent2_private_context: str = ""
    outcome_details: str = ""

@dataclass
class PriceDomainLogEntry:
    """
    Log entry specifically for Single Issue Price Domain.
    """
    session_id: str
    timestamp: str
    model_name: str
    
    agent1_type: str
    agent2_type: str
    
    round_number: int
    round_duration: float
    turns: int
    
    # Context
    buyer_max: float
    seller_min: float
    zopa_low: float
    zopa_high: float
    
    # Outcome
    agreement: bool
    final_price: Optional[float]
    within_zopa: bool
    
    # Utilities
    agent1_utility: float # Buyer
    agent2_utility: float # Seller
    
    # Transcript
    history_str: str # Simplified "Speaker: Action" string
    json_details: str # Full outcome details JSON


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
                        agent2_type: str = "unknown",
                        boulware_initial_threshold: Optional[float] = None,
                        boulware_decrease_rate: Optional[float] = None,
                        boulware_min_threshold: Optional[float] = None,
                        boulware_final_threshold: Optional[float] = None,
                        fixed_price_threshold: Optional[float] = None,
                        price_fixed_margin: Optional[float] = None,
                        price_boulware_beta: Optional[float] = None,
                        reached_consensus: bool = True,
                        **kwargs) -> RawNegotiationLogEntry:
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
        # Check if allocation_tracker has round_states (it might be None for simple domains depending on runner refactor)
        proposal_history = []
        final_proposer = None
        
        if hasattr(allocation_tracker, "round_states"):
            round_state = allocation_tracker.round_states.get(round_obj.round_number)
            if round_state and round_state.proposal_history:
                # Store proposal history as structured data
                proposal_history = [
                    {
                        "agent_num": agent_num,
                        "proposal": {
                            "agent1_items": proposal.agent1_items,
                            "agent2_items": proposal.agent2_items,
                            "is_valid": proposal.is_valid,
                            "error_message": getattr(proposal, "error_message", None)
                        } if hasattr(proposal, "agent1_items") else str(proposal)
                    }
                    for agent_num, proposal in round_state.proposal_history
                ]
                
                # Find final proposer (logic for multi-item mostly)
                for agent_num, proposal in reversed(round_state.proposal_history):
                    if hasattr(proposal, "is_valid") and proposal.is_valid:
                        final_proposer = agent_num
                        break
                # Override with kwargs if provided
        if "proposal_history" in kwargs:
             proposal_history = kwargs["proposal_history"]
        # Extract domain fields from kwargs
        domain_name = kwargs.get("domain_name", "multi_item")
        domain_public_context = json.dumps(kwargs.get("domain_public_context", {}))
        agent1_private_context = json.dumps(kwargs.get("agent1_private_context", {}))
        agent2_private_context = json.dumps(kwargs.get("agent2_private_context", {}))
        outcome_details = json.dumps(kwargs.get("outcome_details", {}))

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
            
            # Boulware agent parameters (if applicable)
            boulware_initial_threshold=boulware_initial_threshold,
            boulware_decrease_rate=boulware_decrease_rate,
            boulware_min_threshold=boulware_min_threshold,
            boulware_final_threshold=boulware_final_threshold,
            
            # Fixed Price agent parameters (if applicable)
            fixed_price_threshold=fixed_price_threshold,
            
            # Price Domain Agent parameters
            price_fixed_margin=price_fixed_margin,
            price_boulware_beta=price_boulware_beta,
            
            # Round identification
            round_number=round_obj.round_number,
            total_rounds=total_rounds,
            
            # Round timing and interaction metrics
            round_duration_seconds=round_duration,
            turn_count=len(round_obj.conversation_history),
            starting_agent=round_obj.starting_agent,
            round_completed=round_obj.is_complete,
            agreement_reached=round_obj.is_complete,
            reached_consensus=reached_consensus,
            
            # Raw data as JSON strings
            items_data=json.dumps(items_data),
            final_allocation=json.dumps(final_allocation),
            conversation_history=json.dumps(round_obj.conversation_history),
            proposal_history=json.dumps(proposal_history),
            final_proposer=final_proposer,

            # Domain specific
            domain_name=domain_name,
            domain_public_context=domain_public_context,
            agent1_private_context=agent1_private_context,
            agent2_private_context=agent2_private_context,
            outcome_details=outcome_details
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
    
    def log_price_round(self, round_obj: Round, outcome_details: Dict[str, Any], duration: float, 
                       agent1_type: str, agent2_type: str, domain_context: Dict[str, Any]):
        """
        Specialized logging for Price Domain rounds.
        """
        import json
        
        # Determine filename for price domain
        if not self.filename.endswith("_price_domain.csv"):
            base = self.filename.replace(".csv", "")
            self.price_filename = f"{base}_price_domain.csv"
            self.price_filepath = os.path.join(self.base_dir, self.price_filename)
        else:
            self.price_filepath = self.filepath

        # Extract context
        buyer_max = domain_context.get("agent1_private_context", {}).get("max_willingness_to_pay", 0.0)
        seller_min = domain_context.get("agent2_private_context", {}).get("min_acceptable_price", 0.0)
        
        # History string
        hist = []
        for role, text in round_obj.conversation_history:
            # simple summary
            hist.append(f"{role}: {text}")
        history_str = " | ".join(hist)

        entry = PriceDomainLogEntry(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            round_number=round_obj.round_number,
            round_duration=duration,
            turns=len(round_obj.conversation_history),
            buyer_max=float(buyer_max),
            seller_min=float(seller_min),
            zopa_low=float(seller_min),
            zopa_high=float(buyer_max),
            agreement=outcome_details.get("agreement", False),
            final_price=outcome_details.get("price"),
            within_zopa=outcome_details.get("within_zopa", False),
            agent1_utility=outcome_details.get("agent1_utility", 0.0),
            agent2_utility=outcome_details.get("agent2_utility", 0.0),
            history_str=history_str,
            json_details=json.dumps(outcome_details)
        )
        
        # Write
        entry_dict = asdict(entry)
        file_exists = os.path.exists(self.price_filepath)
        
        with open(self.price_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(entry_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(self.price_filepath) == 0:
                writer.writeheader()
            writer.writerow(entry_dict)
            
        print(f"Logged Price Round to: {self.price_filename}")

    def get_filepath(self) -> str:
        """Return the full path to the CSV file."""
        return self.filepath
    
    def get_filename(self) -> str:
        """Return just the filename."""
        return self.filename