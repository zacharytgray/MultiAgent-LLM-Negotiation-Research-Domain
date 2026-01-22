import os
import json
import dataclasses
from typing import List, Dict, Any, Optional
from src.core.price_structures import PriceAction, PriceState

class DatasetWriter:
    """
    Handles logging negotiation episodes to a JSONL file for offline RL/Decision Transformer training.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        # Ensure dir exists if specified
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        # Clear file or append? Usually append is safer if resuming, but overwrite if new run.
        # User said "add to additional", but we are running a script. Let's append if exists.
        
        self.episode_buffer: List[Dict] = []
        
    def add_step(self, 
                 state: PriceState, 
                 action: PriceAction, 
                 reward: float,
                 agent_metadata: Dict,
                 meta: Dict,
                 trajectory_id: str,
                 terminal: bool = False):
        """
        Buffer a single step.
        """
        # Helper for float rounding
        def round_floats(obj):
            if isinstance(obj, float):
                return round(obj, 2)
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [round_floats(x) for x in obj]
            return obj

        step_record = {
            "trajectory_id": trajectory_id,
            "t": state.timestep,
            # "role": state.role, # Removed redundant field (present in state)
            "agent": round_floats(agent_metadata),
            "state": round_floats(state.to_dict()),
            "action": round_floats(dataclasses.asdict(action)),
            "reward": round(reward, 4), # Keep reward slightly more precise
            "terminal": terminal,
            "meta": round_floats(meta)
            # return_to_go will be filled later
        }
        self.episode_buffer.append(step_record)
        
    def flush_episode(self, outcome_meta: Dict, agent_rewards: Optional[Dict[str, float]] = None):
        """
        Post-process the current episode buffer to calculate Rho and write to disk.
        Rho is calculated as rho = (Final Price - Min Acceptable) / (Max Acceptable - Min Acceptable)
        Only calculated at the terminal offer in a trajectory.
        Same rho value is assigned to every offer in that trajectory.
        Requires dataset template compliance.
        """
        if not self.episode_buffer:
            return

        # 1. Check agreement and get Final Price
        if not outcome_meta.get("agreement", False):
            # No agreement => No Final Price => Rho undefined => Skip writing
            self.episode_buffer = [] 
            return

        final_price = outcome_meta.get("price")
        if final_price is None:
            self.episode_buffer = []
            return

        # 2. Extract trajectory constants from the first step's metadata
        # We use zopa_low (Seller Min) and zopa_high (Buyer Max) for calculation
        first_step = self.episode_buffer[0]
        meta_start = first_step["meta"]
        
        min_acceptable = meta_start.get("zopa_low")     # Seller Min
        max_acceptable = meta_start.get("zopa_high")    # Buyer Max

        if min_acceptable is None or max_acceptable is None:
            self.episode_buffer = []
            return

        # 3. Calculate Rho
        denominator = max_acceptable - min_acceptable
        rho = 0.0
        if denominator != 0:
            rho = (final_price - min_acceptable) / denominator
        
        # Round Rho
        rho = round(rho, 4)

        # 4. Write records with new structure
        with open(self.filepath, 'a', encoding='utf-8') as f:
            for step in self.episode_buffer:
                
                # --- Map State to match template ---
                # state {timestep, max turns, role, last_offer_price, offer_history, reservation_price, price_range}
                raw_state = step["state"]
                clean_state = {
                    "timestep": raw_state.get("timestep"),
                    "max_turns": raw_state.get("max_turns"),
                    "role": raw_state.get("role"),
                    "last_offer_price": raw_state.get("last_offer_price"),
                    "offer_history": raw_state.get("offer_history"),
                    "reservation_price": raw_state.get("true_reservation_price"), # Map from true_res
                    "price_range": raw_state.get("public_price_range")      # Map from public_range
                }

                # --- Map Meta to match template ---
                # meta {buyer_max, seller_min, zopa_low, zopa_high, zopa_width, accepted_price, episode_outcome{agreement_bool, price, num_turns}}
                
                raw_meta = step["meta"]
                
                # Ensure episode_outcome structure matches: {agreement_bool, price, num_turns}
                clean_outcome = {
                    "agreement_bool": outcome_meta.get("agreement"),
                    "price": outcome_meta.get("price"),
                    "num_turns": outcome_meta.get("turns")
                }

                clean_meta = {
                    "buyer_max": raw_meta.get("buyer_max"),
                    "seller_min": raw_meta.get("seller_min"),
                    "zopa_low": raw_meta.get("zopa_low"),
                    "zopa_high": raw_meta.get("zopa_high"),
                    "zopa_width": raw_meta.get("zopa_width"),
                    "accepted_price": raw_meta.get("accepted_price"),
                    "episode_outcome": clean_outcome
                }

                # --- Map Agent to match template ---
                # agent {strategy, strategy_params {...}}
                raw_agent = step["agent"]
                clean_agent = {
                    "strategy": raw_agent.get("strategy"),
                    "strategy_params": raw_agent.get("strategy_params")
                }

                # --- Assemble Final Record ---
                record = {
                    "trajectory_id": step["trajectory_id"],
                    "turn": step["t"], # mapped from 't'
                    "agent": clean_agent,
                    "state": clean_state,
                    "action": step["action"],
                    "is_terminal": step["terminal"],
                    "meta": clean_meta,
                    "rho": rho
                }
                
                f.write(json.dumps(record) + "\n")
        
        self.episode_buffer = []
