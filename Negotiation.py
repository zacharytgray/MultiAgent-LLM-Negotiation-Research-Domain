import asyncio
import random
import time
import argparse
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.agents.agent_factory import AgentFactory, AgentConfig
from colorama import Fore, init, Style
from src.core.Item import Item
from src.core.Round import Round
from src.utils.MessageParser import MessageParser
from src.utils.AllocationTracker import AllocationTracker
from src.utils.CSVLogger import CSVLogger
from config.settings import *

# Dataset mode imports
from src.core.price_structures import PriceAction, PriceState
from src.agents.price_strategies import DeterministicPriceAgent, STRATEGY_REGISTRY
from src.logging.dataset_writer import DatasetWriter

# Initialize colorama
init(autoreset=True)

from src.domains.multi_item_allocation_domain import MultiItemAllocationDomain
from src.domains.single_issue_price_domain import SingleIssuePriceDomain
from src.domains.base_domain import ParsedAction

class NegotiationSession:
    """
    Main class that manages the entire negotiation session across multiple rounds.
    """
    def __init__(self, num_rounds: int = DEFAULT_NUM_ROUNDS, items_per_round: int = DEFAULT_ITEMS_PER_ROUND, 
                 model_name: str = DEFAULT_MODEL_NAME, agent1_type: str = "default", 
                 agent2_type: str = "default", agent1_config: Optional[Dict] = None, 
                 agent2_config: Optional[Dict] = None, domain_type: str = "multi_item",
                 dataset_mode: bool = False, dataset_out: str = "datasets/price_domain.jsonl",
                 seed: Optional[int] = None,
                 csv_logger: Optional[CSVLogger] = None,
                 **domain_kwargs):
        self.num_rounds = num_rounds
        self.items_per_round = items_per_round
        self.domain_type = domain_type
        self.domain_kwargs = domain_kwargs
        
        self.dataset_mode = dataset_mode
        self.dataset_out = dataset_out
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)

        # Initialize Domain
        if domain_type == "price":
            self.domain = SingleIssuePriceDomain()
        else:
            self.domain = MultiItemAllocationDomain()
            
        if self.dataset_mode:
            if self.domain_type != "price" and self.domain_type != "single_issue_price":
                raise ValueError("Dataset mode currently only supports single_issue_price domain.")
            print(f"{Fore.MAGENTA}*** DATASET MODE ENABLED ***{Fore.RESET}")
            print(f"Output: {self.dataset_out}")
            self.dataset_writer = DatasetWriter(self.dataset_out)
            # Agents will be initialized per episode in run_dataset_generation
            return

        self.model_name = model_name.replace(":", "_")  # Clean up for filename
        self.agent1_type = agent1_type
        self.agent2_type = agent2_type
        self.rounds = []
        self.total_scores = {"agent1": 0.0, "agent2": 0.0}
        
        # Initialize CSV logger
        if csv_logger:
            self.csv_logger = csv_logger
        else:
            session_label = f"{agent1_type}_vs_{agent2_type}"
            self.csv_logger = CSVLogger(f"{self.model_name}_{session_label}", items_per_round)
            print(f"{Fore.GREEN}[CSV] CSV logging to: {self.csv_logger.get_filename()}{Fore.RESET}")
        
        # Get agent configurations
        if agent1_config is None:
            agent1_config = AgentConfig.get_config_for_type(agent1_type)
        if agent2_config is None:
            agent2_config = AgentConfig.get_config_for_type(agent2_type)
        
        # Determine system instructions files for each agent
        agent1_instructions = self._get_system_instructions_file(agent1_type)
        agent2_instructions = self._get_system_instructions_file(agent2_type)
        
        # Mapping logic for Price Domain:
        # If user asks for "boulware" in price domain, use "price_strategy" wrapper with "boulware_conceding" strategy.
        # If user asks for "default" in price domain, use "basic_price".
        if domain_type in ["price", "single_issue_price"] and not self.dataset_mode:
            
            def map_agent_type(a_type, a_config):
                # Basic
                if a_type in ["default", "basic"]:
                    return "basic_price", a_config
                
                # Strategies
                # If it's a known strategy name in price_strategies (e.g. "boulware_linear"), 
                # use "price_strategy" agent and force that strategy param.
                from src.agents.price_strategies import STRATEGY_REGISTRY
                
                # Direct match?
                if a_type in STRATEGY_REGISTRY:
                    # Pass strategy_name via config
                    new_config = a_config.copy()
                    new_config["strategy_name"] = a_type
                    return "price_strategy", new_config
                
                # Generic "boulware" -> "boulware_conceding" (default)
                if a_type == "boulware":
                     new_config = a_config.copy()
                     if "strategy_name" not in new_config:
                         new_config["strategy_name"] = "boulware_conceding"
                     return "price_strategy", new_config
                
                # Generic "tit_for_tat"
                if a_type == "tit_for_tat":
                     new_config = a_config.copy()
                     new_config["strategy_name"] = "tit_for_tat"
                     return "price_strategy", new_config

                # Already correct type?
                if a_type in ["price_strategy", "basic_price"]:
                    return a_type, a_config

                return a_type, a_config # Fallback

            agent1_type, agent1_config = map_agent_type(agent1_type, agent1_config)
            agent2_type, agent2_config = map_agent_type(agent2_type, agent2_config)
            
            print(f"{Fore.MAGENTA}[Domain Logic] Mapped agents to Price Domain types: {agent1_type}, {agent2_type}{Fore.RESET}")

        # Initialize agents using factory
        self.agent1 = AgentFactory.create_agent(
            agent1_type, 1, model_name, agent1_instructions, **agent1_config
        )
        self.agent2 = AgentFactory.create_agent(
            agent2_type, 2, model_name, agent2_instructions, **agent2_config
        )
        
        # Initialize proposal tracking components
        self.message_parser = MessageParser()
        self.allocation_tracker = AllocationTracker()

    def run(self):
        """
        Run the negotiation session.
        """
        if self.dataset_mode:
            # Delegate to dataset generation loop
            self.run_dataset_generation()
            return

        print(f"Starting Negotiation Session: {self.num_rounds} rounds")
        
        for i in range(1, self.num_rounds + 1):
            asyncio.run(self.run_round(i))
            
        self._print_overall_stats()

    def run_dataset_generation(self):
        """
        Run in dataset generation mode.
        Loops for self.num_rounds (which acts as num_episodes here, or use separate arg).
        Requires single_issue_price domain.
        """
        num_episodes = self.num_rounds
        
        print(f"Generating {num_episodes} episodes...")
        
        strategies = list(STRATEGY_REGISTRY.keys())
        
        stats = {
            "episodes": 0,
            "agreements": 0,
            "total_norm_welfare_buyer": 0.0,
            "total_norm_welfare_seller": 0.0,
            "total_length": 0
        }
        
        # Max turns per episode - use a config or standard value
        MAX_EPISODE_TURNS = 20
        if "max_turns" in self.domain_kwargs:
            MAX_EPISODE_TURNS = int(self.domain_kwargs["max_turns"])
        elif "max_rounds" in self.domain_kwargs: # Backwards compat
            MAX_EPISODE_TURNS = int(self.domain_kwargs["max_rounds"])
        elif hasattr(self, "domain_kwargs") and "rounds" in self.domain_kwargs:
             MAX_EPISODE_TURNS = int(self.domain_kwargs["rounds"])
             
        for ep in range(1, num_episodes + 1):
            trajectory_id = f"run_{ep:06d}"
            
            # 1. Select Strategies
            strat1_name = random.choice(strategies)
            strat2_name = random.choice(strategies)
            
            # 2. Instantiate Agents
            agent1 = DeterministicPriceAgent(1, strat1_name)
            agent2 = DeterministicPriceAgent(2, strat2_name)
            
            # 3. Reset Domain and Override Bounds for Dataset Balance (Paper Replicating Mode)
            self.domain.reset(ep)
            
            # Paper Setup:
            # Buyer Max ~ N(900, 50)
            # Seller Min = Buyer Max - 500
            # ZOPA = 500
            
            buyer_max_val = round(random.gauss(900, 50), 2)
            seller_min_val = buyer_max_val - 500.0
            
            # Apply overrides to domain object
            self.domain.buyer_max = buyer_max_val
            self.domain.seller_min = seller_min_val
            
            # Effective reservation prices for agents
            # Agents should not accept deals worse than fallback.
            # Buyer Indifference: B_max - P = 0 -> P = B_max
            # Seller Indifference: P - S_min = 0 -> P = S_min 
            fallback_utility = 0.0
            
            buyer_effective_res = buyer_max_val
            seller_effective_res = seller_min_val
            
            # Inject ZOPA info if strategy needs it (Oracle strategies)
            # For random_zopa, we should restrict to effective ZOPA? 
            # Or true ZOPA? Random ZOPA usually implies random valid DEAL.
            # Valid deal must be better than fallback.
            
            buyer_max = self.domain.buyer_max
            seller_min = self.domain.seller_min
            
            # Update params if needed (Oracle injection)
            if strat1_name == "random_zopa":
                agent1.params.update({"zopa_min": seller_min, "zopa_max": buyer_max})
            if strat2_name == "random_zopa":
                agent2.params.update({"zopa_min": seller_min, "zopa_max": buyer_max})
                
            # 4. Run Episode
            # Randomize who starts?
            starting_agent_id = random.choice([1, 2])
            
            offer_history: List[Tuple[str, float]] = []
            last_offer_price = None
            
            # Setup State Meta
            zopa_exists = True # Always 500
            zopa_width = 500.0
            
            meta_template = {
                "buyer_max": buyer_max,
                "seller_min": seller_min,
                # "buyer_effective_max": buyer_effective_res, # Removed
                # "seller_effective_min": seller_effective_res, # Removed
                "zopa_exists": True,
                "zopa_low": seller_min,
                "zopa_high": buyer_max,
                "zopa_width": zopa_width,
                # "fallback_utility": fallback_utility # Removed
            }
            
            t = 0
            done = False
            agreement_reached = False
            final_price = None
            
            # We need to alternate
            current_agent_id = starting_agent_id
            
            while t < MAX_EPISODE_TURNS and not done:
                t += 1
                
                # Determine Role
                current_role = "buyer" if current_agent_id == 1 else "seller"
                current_agent = agent1 if current_agent_id == 1 else agent2
                
                # Pass effective reservation to agent (equals true reservation now)
                current_true_res = buyer_max if current_role == "buyer" else seller_min
                
                # Construct State
                state = PriceState(
                    timestep=t,
                    max_turns=MAX_EPISODE_TURNS, # Updated param name
                    role=current_role,
                    last_offer_price=last_offer_price,
                    offer_history=list(offer_history), # copy
                    effective_reservation_price=current_true_res, # Deprecated field
                    true_reservation_price=current_true_res,
                    public_price_range=(200.0, 1500.0)
                )
                
                # Get Action
                try:
                    action = current_agent.propose_action(state)
                except Exception as e:
                    print(f"Error in strategy {current_agent.strategy_name}: {e}")
                    break
                    
                # Validate Action
                if action.type == "OFFER":
                    if action.price is None:
                        action.price = 0.0 # Error state
                    
                    # Update tracking
                    last_offer_price = action.price
                    offer_history.append((current_role, action.price))
                    
                elif action.type == "ACCEPT":
                    if last_offer_price is None:
                         pass 
                    else:
                        done = True
                        agreement_reached = True
                        final_price = last_offer_price
                        
                # Determine Reward
                # Normalize rewards based on ZOPA=500 and Fallback=100 logic
                # Scale utility: 500 = 1.0, 0 = 0.0
                # Fallback Utility = 100 -> 0.2
                
                lambda_time = 1.0 / MAX_EPISODE_TURNS
                reward = -lambda_time
                
                # Shape terminal reward
                outcome_meta = {}
                if done and agreement_reached:
                    # Utility
                    if current_role == "buyer":
                         final_util = buyer_max - final_price
                         norm_util = max(0, min(1, final_util / zopa_width))
                         reward += norm_util
                         stats["total_norm_welfare_buyer"] += norm_util
                         
                         seller_util = final_price - seller_min
                         s_norm = max(0, min(1, seller_util / zopa_width))
                         stats["total_norm_welfare_seller"] += s_norm
                         
                    else: # Seller Agreed
                         final_util = final_price - seller_min
                         norm_util = max(0, min(1, final_util / zopa_width))
                         reward += norm_util
                         stats["total_norm_welfare_seller"] += norm_util
                         
                         buyer_util = buyer_max - final_price
                         b_norm = max(0, min(1, buyer_util / zopa_width))
                         stats["total_norm_welfare_buyer"] += b_norm
                         
                    outcome_meta = {
                        "agreement": True,
                        "price": final_price,
                        "turns": t
                    }
                    stats["agreements"] += 1
                elif t >= MAX_EPISODE_TURNS:
                     done = True
                     # Terminal Impasse: Fallback Reward
                     # Reward = 100 (fallback) / 500 (ZOPA) = 0.2
                     impasse_reward = 100.0 / 500.0
                     reward += impasse_reward
                     
                     outcome_meta = {"agreement": False}
                     
                # Prepare normalized rewards for both agents for correct RTG
                # Calculate what the normalized utility WOULD be / IS for meaningful logging
                # If agreement:
                if done and agreement_reached:
                     # Buyer
                     b_u = buyer_max - final_price
                     b_norm_util = max(0.0, min(1.0, b_u / zopa_width))
                     
                     # Seller
                     s_u = final_price - seller_min
                     s_norm_util = max(0.0, min(1.0, s_u / zopa_width))
                     
                     agent_rewards = {"buyer": b_norm_util, "seller": s_norm_util}
                elif done and not agreement_reached:
                     # Impasse
                     impasse_val = 100.0 / 500.0
                     agent_rewards = {"buyer": impasse_val, "seller": impasse_val}
                else:
                     agent_rewards = None

                # Calculate Meta for this step
                step_meta = meta_template.copy()
                step_meta["accepted_price"] = final_price if (done and agreement_reached) else None
                
                if last_offer_price is not None:
                    # offer_outside_zopa_gap
                    gap = 0.0
                    if last_offer_price < seller_min:
                        gap += (seller_min - last_offer_price)
                    if last_offer_price > buyer_max:
                        gap += (last_offer_price - buyer_max)
                    step_meta["offer_outside_zopa_gap"] = gap
                else:
                    step_meta["offer_outside_zopa_gap"] = None
                    
                # Log Step
                # Note: 'agent_metadata' should refer to the agent acting in this step
                self.dataset_writer.add_step(
                    state=state,
                    action=action,
                    reward=reward,
                    agent_metadata=current_agent.get_metadata(),
                    meta=step_meta,
                    trajectory_id=trajectory_id,
                    terminal=done
                )
                
                # Switch turn
                current_agent_id = 3 - current_agent_id
                
            # End of Episode
            self.dataset_writer.flush_episode(outcome_meta, agent_rewards)
            stats["episodes"] += 1
            stats["total_length"] += t
            
            if ep % 100 == 0:
                 print(f"Gen: {ep}/{num_episodes} ... Agreement Rate: {stats['agreements']/ep:.2f}")

        # Summary
        print("\n=== Dataset Generation Complete ===")
        print(f"Episodes: {stats['episodes']}")
        if stats['episodes'] > 0:
            print(f"Agreement Rate: {stats['agreements'] / stats['episodes']:.2%}")
            print(f"Avg Length: {stats['total_length'] / stats['episodes']:.2f}")
        if stats['agreements'] > 0:
            print(f"Avg Norm. Buyer Utility: {stats['total_norm_welfare_buyer'] / stats['agreements']:.4f}")
            print(f"Avg Norm. Seller Utility: {stats['total_norm_welfare_seller'] / stats['agreements']:.4f}")
        print(f"Saved to: {self.dataset_out}")

    def _get_system_instructions_file(self, agent_type: str) -> str:
        """
        Get the appropriate system instructions file for the given agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            str: Path to system instructions file
        """
        if self.domain_type in ["price", "single_issue_price"]:
            if agent_type in ["boulware", "price_strategy", "price_boulware"]: # Deterministic Price Agents
                 return PRICE_DETERMINISTIC_INSTRUCTIONS_FILE
            else: # Basic Price Agents
                 return PRICE_SYSTEM_INSTRUCTIONS_FILE
        
        # Default Multi-Item Logic
        if agent_type in ["boulware"]:  # Deterministic agents
            return DETERMINISTIC_AGENT_INSTRUCTIONS_FILE
        elif agent_type == "charming":
            return CHARMING_AGENT_INSTRUCTIONS_FILE
        elif agent_type == "rude":
            return RUDE_AGENT_INSTRUCTIONS_FILE
        else:
            return SYSTEM_INSTRUCTIONS_FILE
        
    def generate_random_items(self, round_number: int) -> List[Item]:
        """
        Generate random items for a round with random values for each agent.
        """
        items = []
        
        for i in range(self.items_per_round):
            name = ITEM_NAMES[i] if i < len(ITEM_NAMES) else f"Item{i+1}"
            agent1_value = round(random.uniform(MIN_ITEM_VALUE, MAX_ITEM_VALUE), ITEM_VALUE_PRECISION)
            agent2_value = round(random.uniform(MIN_ITEM_VALUE, MAX_ITEM_VALUE), ITEM_VALUE_PRECISION)
            items.append(Item(name, agent1_value, agent2_value))
            
        return items
    
    def _prepare_round_setup(self, round_number: int) -> Tuple[Any, int, Round]:
        """
        Prepare the basic setup for a round: domain reset, starting agent, and round object.
        """
        # Reset agent memories to start fresh each round (preserving only system instructions)
        self.agent1.reset_memory()
        self.agent2.reset_memory()
        
        # Reset domain for this round
        domain_state = self.domain.reset(round_number, items_per_round=self.items_per_round)
        items = domain_state.get("items", []) # For backward compatibility with Round object requiring items
        
        # Determine starting agent (alternates each round)
        starting_agent = 1 if round_number % 2 == 1 else 2
        
        # Create round object (Legacy: Round expects items list)
        # If price domain, items might be empty or dummy.
        round_obj = Round(round_number, items, self.agent1, self.agent2, starting_agent)
        
        return items, starting_agent, round_obj
    
    def _display_round_info(self, round_number: int, items: List[Item], starting_agent: int):
        """
        Display round information to the console.
        """
        print(f"\n{Fore.CYAN}{'='*SEPARATOR_LENGTH}")
        print(f"Round {round_number} Starting")
        print(f"{'='*SEPARATOR_LENGTH}{Fore.RESET}")
        
        if self.domain_type in ["price", "single_issue_price"] and hasattr(self.domain, "buyer_max"):
             print(f"\n{Fore.YELLOW}Price Negotiation Stats:{Fore.RESET}")
             print(f"  Buyer Max (Willingness to Pay): ${self.domain.buyer_max}")
             print(f"  Seller Min (Acceptable Price): ${self.domain.seller_min}")
             if self.domain.buyer_max >= self.domain.seller_min:
                 print(f"  ZOPA: [${self.domain.seller_min}, ${self.domain.buyer_max}]")
             else:
                 print(f"  ZOPA: None (Negative Bargaining Zone!)")
        
        if items:
            print(f"\n{Fore.YELLOW}Items for Round {round_number}:")
            for item in items:
                print(f"  {item.name}: Agent1={item.agent1Value}, Agent2={item.agent2Value}")
        
        start_name = self._get_agent_name(starting_agent)
        print(f"Starting Agent: {start_name}{Fore.RESET}\n")
    
    def _prepare_agent_contexts(self, round_obj: Round):
        """
        Prepare and set initial context for both agents.
        """
        # Set contexts for both agents
        # Private context
        ctx1_private = self.domain.get_private_context(1)
        ctx2_private = self.domain.get_private_context(2)
        
        # Public context
        ctx_public = self.domain.get_public_context()
        
        self.agent1.set_domain_context(ctx1_private, ctx_public)
        self.agent2.set_domain_context(ctx2_private, ctx_public)
        
        # Legacy support: set items if available (for internal logic of some agents like Boulware)
        if "items" in ctx1_private:
            self.agent1.set_items(ctx1_private["items"])
        if "items" in ctx2_private:
            self.agent2.set_items(ctx2_private["items"])

        # Create system prompt additions
        # We rely on format_agent_prompt_context from domain
        msg1 = self.domain.format_agent_prompt_context(1)
        msg2 = self.domain.format_agent_prompt_context(2)
        
        agent1_context = f"""
--Round Start--
You are Agent 1 in a negotiation. Your goal is to maximize your own value.
{msg1}
"""
        
        agent2_context = f"""
--Round Start--
You are Agent 2 in a negotiation. Your goal is to maximize your own value.
{msg2}
"""
        
        # Add context to agents
        self.agent1.add_to_memory('system', agent1_context)
        self.agent2.add_to_memory('system', agent2_context)
        
        # Give the starting agent an initial prompt to begin the negotiation
        if round_obj.starting_agent == 1:
            self.agent1.add_to_memory('user', "Please begin the negotiation by making your opening proposal.")
        else:
            self.agent2.add_to_memory('user', "Please begin the negotiation by making your opening proposal.")
    
    async def _execute_negotiation_loop(self, round_obj: Round, available_items: List[str]) -> bool:
        """
        Execute the main negotiation loop between agents with improved error handling.
        Returns True if agreement reached, False if max turns exceeded.
        """
        max_turns = MAX_TURNS_PER_ROUND  # Prevent infinite loops
        turn_count = 0
        current_agent_num = round_obj.starting_agent
        max_retries_per_turn = MAX_RETRIES_PER_INVALID_PROPOSAL  # Maximum retries for invalid proposals
        pressure_message_sent = {1: False, 2: False}  # Track which agents have received pressure message
        
        while turn_count < max_turns and not round_obj.is_complete:
            turn_count += 1
            
            # Get references to current and other agents
            if current_agent_num == 1:
                current_agent = self.agent1
                other_agent = self.agent2
                current_color = Fore.GREEN
                other_agent_num = 2
            else:
                current_agent = self.agent2
                other_agent = self.agent1
                current_color = Fore.BLUE
                other_agent_num = 1
            
            # Update agent context with latest history before turn
            # This is critical for Price Strategy Agents that need history to compute moves
            if self.domain_type in ["price", "single_issue_price"]:
                 p_ctx = self.domain.get_private_context(current_agent_num)
                 pub_ctx = self.domain.get_public_context()
                 current_agent.set_domain_context(p_ctx, pub_ctx)

            # Add time pressure message for non-deterministic agents when 5 turns remain
            turns_remaining = max_turns - turn_count
            if turns_remaining <= 5:
                # --- Fallback allocation calculation (Only for Multi-Item) ---
                msg = None
                
                if self.domain_type == "multi_item" and round_obj.items:
                    def all_possible_allocations(items):
                        n = len(items)
                        allocations = []
                        for bits in range(2**n):
                            a1, a2 = [], []
                            for i in range(n):
                                if (bits >> i) & 1:
                                    a1.append(items[i].name)
                                else:
                                    a2.append(items[i].name)
                            allocations.append({'agent1': a1, 'agent2': a2})
                        return allocations

                    def score_allocation_welfare(allocation, items):
                        score = 0.0
                        for item in items:
                            if item.name in allocation['agent1']:
                                score += item.agent1Value
                            elif item.name in allocation['agent2']:
                                score += item.agent2Value
                        return score

                    def score_agent_value(allocation, items, agent):
                        score = 0.0
                        for item in items:
                            if item.name in allocation[agent]:
                                score += getattr(item, f'{agent}Value')
                        return score

                    items = round_obj.items
                    allAllocations = all_possible_allocations(items)
                    rankedAllocations = sorted(allAllocations, key=lambda alloc: score_allocation_welfare(alloc, items))
                    fallbackIndexPercent = 0.10
                    fallbackIndex = int(fallbackIndexPercent * len(rankedAllocations))
                    fallbackIndex = min(max(fallbackIndex, 0), len(rankedAllocations)-1)
                    fallbackAlloc = rankedAllocations[fallbackIndex]
                    fallbackScore1 = score_agent_value(fallbackAlloc, items, 'agent1')
                    fallbackScore2 = score_agent_value(fallbackAlloc, items, 'agent2')
                    # Prepare agent-specific pressure messages
                    pressure_message_agent1 = (
                        f"⚠️  TIME PRESSURE ALERT: You have only {turns_remaining} turns remaining to reach an agreement! "
                        "There is a strict time limit in this negotiation. If you do not come to an agreement soon, "
                        f"you will receive a fallback allocation.\n"
                        f"Your fallback allocation value: {fallbackScore1:.2f}\n"
                        "You must prioritize reaching an agreement quickly to avoid this penalty. "
                        "Make item sacrifices as necessary to avoid this penalty."
                    )
                    pressure_message_agent2 = (
                        f"⚠️  TIME PRESSURE ALERT: You have only {turns_remaining} turns remaining to reach an agreement! "
                        "There is a strict time limit in this negotiation. If you do not come to an agreement soon, "
                        f"you will receive a fallback allocation.\n"
                        f"Your fallback allocation value: {fallbackScore2:.2f}\n"
                        "You must prioritize reaching an agreement quickly to avoid this penalty. "
                        "Make item sacrifices as necessary to avoid this penalty."
                    )
                    msg = pressure_message_agent1 if current_agent_num == 1 else pressure_message_agent2
                
                else:
                    # Generic pressure message
                    msg = (f"⚠️  TIME PRESSURE ALERT: You have only {turns_remaining} turns remaining to reach an agreement! "
                           "There is a strict time limit. If you do not agree soon, the negotiation will end in impasse with 0 utility.")

                # Send pressure message and update pressure message sent dict
                if (msg and not current_agent.should_make_deterministic_proposal(turn_count) and 
                    not pressure_message_sent[current_agent_num]):
                    agent_name = self._get_agent_name(current_agent_num)
                    print(f"{Fore.YELLOW}[ALERT] Sending time pressure alert to {agent_name} ({current_agent.agent_type if hasattr(current_agent, 'agent_type') else 'Unknown'}):{Fore.RESET}")
                    current_agent.add_to_memory('user', msg)
                    pressure_message_sent[current_agent_num] = True
            
            # Process agent turn with retry logic for invalid proposals
            response, turn_successful = await self._process_agent_turn_with_retry(
                current_agent, current_agent_num, current_color, turn_count, 
                round_obj, available_items, max_retries_per_turn
            )
            
            if not turn_successful:
                agent_name = self._get_agent_name(current_agent_num)
                print(f"{Fore.RED}[WARN] {agent_name} failed to provide a valid response after {max_retries_per_turn} retries. Ending round.{Fore.RESET}")
                break
            
            # Add system message with turns remaining to the agent's response before sending to the other agent
            system_msg = f"\n\n**SYSTEM MESSAGE: YOU HAVE {turns_remaining} TURNS REMAINING TO COME TO AN AGREEMENT**"
            response_with_system = f"{response}{system_msg}"
            agent_name = self._get_agent_name(current_agent_num)
            other_agent.add_to_memory('user', f"{agent_name}: {response_with_system}")
            
            # Store conversation history
            round_obj.conversation_history.append((current_agent_num, response))
            
            # Check if round completed due to agreement
            if round_obj.is_complete:
                break
                
            # Switch to other agent
            current_agent_num = other_agent_num
        
        return round_obj.is_complete
    
    async def _process_agent_turn_with_retry(self, current_agent: BaseAgent, current_agent_num: int, 
                                           current_color: str, turn_count: int, round_obj: Round, 
                                           available_items: List[str], max_retries: int) -> Tuple[str, bool]:
        """
        Process a single agent's turn with retry logic for invalid proposals.
        Now includes support for deterministic agents.
        Returns (agent_response, turn_successful).
        """
        # Check if this agent should make a deterministic proposal
        if current_agent.should_make_deterministic_proposal(turn_count):
            return await self._process_deterministic_agent_turn(
                current_agent, current_agent_num, current_color, turn_count, 
                round_obj, available_items, max_retries
            )
        
        # Regular agent processing with retry logic
        retry_count = 0
        agent_name = self._get_agent_name(current_agent_num)
        
        while retry_count <= max_retries:
            # Generate response from current agent
            if retry_count == 0:
                print(f"{current_color}{agent_name}'s turn (Turn {turn_count}):{Fore.RESET}")
            else:
                print(f"{current_color}{agent_name} retry {retry_count} (Turn {turn_count}):{Fore.RESET}")
            
            response = await current_agent.generate_response()
            # Windows console safety
            safe_response = response.encode("cp1252", "replace").decode("cp1252")
            print(f"{current_color}{agent_name}: {safe_response}{Fore.RESET}\n")
            
            # Use Domain to parse action
            action = self.domain.parse_agent_action(current_agent_num, response)
            
            # Validity Check
            if self.domain.is_valid_action(action):
                 # Apply action to domain state
                 self.domain.apply_action(action, current_agent_num)

                 # Update legacy components if needed (for multi_item only)
                 # Original log expected proposal extraction.
                 if self.domain_type == "multi_item":
                     if action.action_type == "OFFER" and action.offer_content:
                          print(f"{Fore.YELLOW}[OK] Valid proposal detected from {agent_name}:{Fore.RESET}")
                          self.allocation_tracker.update_proposal(round_obj.round_number, current_agent_num, action.offer_content)
                     elif action.action_type == "ACCEPT":
                          print(f"{Fore.CYAN}{agent_name} agreed (ACCEPT)!{Fore.RESET}")
                          # In multi-item, explicit AGREE keyword check was used too.
                          # If domain says ACCEPT, we record generic agreement count?
                          # AllocationTracker relies on `record_agreement` which bumps counter.
                          # Domain logic already handles agreement logic (consecutive agreements).
                          pass
                 else:
                     if action.action_type == "OFFER":
                         print(f"{Fore.YELLOW}[OK] Offer detected: {action.offer_content}{Fore.RESET}")
                     elif action.action_type == "ACCEPT":
                         print(f"{Fore.CYAN}{agent_name} Accepted!{Fore.RESET}")

                 # Check Agreement
                 if self.domain.is_agreement():
                      # Update round status
                      round_obj.is_complete = True
                      outcome = self.domain.get_outcome()
                      if outcome["agreement"]:
                           print(f"{Fore.CYAN}Agreement Reached in Round {round_obj.round_number}!{Fore.RESET}")
                           print(f"{Fore.CYAN}Outcome: {outcome}{Fore.RESET}")
                           if self.domain_type == "multi_item":
                                round_obj.final_allocation = outcome["final_allocation"]
                                # Sync tracker for log
                                self.allocation_tracker.record_agreement(round_obj.round_number, current_agent_num)

                 break # Valid and applied

            else:
                # Invalid Action logic
                if self.domain_type == "multi_item":
                     if action.action_type == "OFFER" and hasattr(action.offer_content, "is_valid") and not action.offer_content.is_valid:
                          print(f"{Fore.RED}[INVALID] Invalid proposal from {agent_name}: {action.offer_content.error_message}{Fore.RESET}")
                          if retry_count < max_retries:
                                feedback = self._generate_proposal_feedback(action.offer_content.error_message, available_items)
                                current_agent.add_to_memory('user', feedback)
                                print(f"{Fore.YELLOW}[RETRY] Providing feedback and retrying...{Fore.RESET}")
                                retry_count += 1
                                continue
                          else:
                                print(f"{Fore.RED}[FAIL] {agent_name} exceeded maximum retries for valid proposal{Fore.RESET}")
                                return response, False
                     
                     # Check for legacy string-based agreement if domain didn't catch it as ACCEPT action
                     # (MultiItemAllocationDomain parses "AGREE" as ACCEPT, so we should be good)
                     pass

                elif self.domain_type == "price":
                     # No retry for now, just continue (chatting)
                     pass
                
                break # Default break for no-op/chat messages
            
        return response, True
    
    async def _process_deterministic_agent_turn(self, current_agent: BaseAgent, current_agent_num: int,
                                              current_color: str, turn_count: int, round_obj: Round,
                                              available_items: List[str], max_retries: int) -> Tuple[str, bool]:
        """
        Process a turn for a deterministic agent (like Boulware).
        Includes validation that the output matches the intended deterministic proposal.
        """
        # Update agent strategy state
        current_agent.update_strategy_state(turn_count)
        
        # Check if agent should accept current proposal
        current_proposal_obj = None  # Could be ParsedProposal or price float
        should_accept = False
        
        # Domain-agnostic approach
        if self.domain_type == "multi_item":
             current_proposal_obj = self.allocation_tracker.get_current_proposal(round_obj.round_number)
             if current_proposal_obj and current_agent.should_accept_proposal(current_proposal_obj, turn_count):
                 should_accept = True
        elif self.domain_type == "price":
             # Price domain: check if there is an offer
             if self.domain.current_offer is not None:
                  current_proposal_obj = ParsedAction("OFFER", self.domain.current_offer, "") 
                  if current_agent.should_accept_proposal(current_proposal_obj, turn_count):
                      should_accept = True
        
        # EXECUTE ACCEPT
        if should_accept:
            agent_name = self._get_agent_name(current_agent_num)
            print(f"{current_color}{agent_name}'s turn (Turn {turn_count}) - Should Accept:{Fore.RESET}")
            
            # Instruct agent to accept
            # Use domain specific accept instruction? Or generic.
            accept_instruction = "The current proposal is acceptable to you. Please respond by agreeing to it and end your message with 'AGREE' or 'ACCEPT'."
            current_agent.add_to_memory('user', accept_instruction)
            
            response = await current_agent.generate_response()
            # Windows console safety
            safe_response = response.encode("cp1252", "replace").decode("cp1252")
            print(f"{current_color}{agent_name}: {safe_response}{Fore.RESET}\n")
            
            # Use Domain to parse and apply
            action = self.domain.parse_agent_action(current_agent_num, response)
            if self.domain.is_valid_action(action) and action.action_type == "ACCEPT":
                 self.domain.apply_action(action, current_agent_num)
                 print(f"{Fore.CYAN}{agent_name} agreed!{Fore.RESET}")
                 
                 # Check agreement
                 if self.domain.is_agreement():
                     print(f"{Fore.CYAN}Agreement Reached! Round {round_obj.round_number} complete.{Fore.RESET}")
                     round_obj.is_complete = True
                     # Sync tracker for legacy logging if needed
                     if self.domain_type == "multi_item":
                        self.allocation_tracker.record_agreement(round_obj.round_number, current_agent_num)
                        # Sync final allocation
                        outcome = self.domain.get_outcome()
                        if outcome["agreement"]:
                            round_obj.final_allocation = outcome["final_allocation"]

            return response, True
        
        # Agent should make a deterministic proposal
        intended_proposal = current_agent.get_deterministic_proposal(turn_count)
        
        agent_name = self._get_agent_name(current_agent_num)
        
        if not intended_proposal:
            print(f"{Fore.RED}❌ {agent_name} could not generate deterministic proposal{Fore.RESET}")
            return "I'm unable to make a proposal at this time.", False
        
        # Prepare instruction for the agent
        proposal_instruction = self._create_deterministic_proposal_instruction(intended_proposal)
        
        # Retry logic for deterministic agents
        retry_count = 0
        
        while retry_count <= max_retries:
            if retry_count == 0:
                print(f"{current_color}{agent_name}'s turn (Turn {turn_count}) - Deterministic:{Fore.RESET}")
                # Give agent the deterministic proposal instruction
                current_agent.add_to_memory('user', proposal_instruction)
            else:
                print(f"{current_color}{agent_name} deterministic retry {retry_count} (Turn {turn_count}):{Fore.RESET}")
                feedback = "Your previous response was not parsed correctly. Please ensure you strictly include the required format (e.g., 'OFFER 100') and be brief."
                current_agent.add_to_memory('user', feedback)
            
            response = await current_agent.generate_response()
            # Windows console safety
            safe_response = response.encode("cp1252", "replace").decode("cp1252")
            print(f"{current_color}{agent_name}: {safe_response}{Fore.RESET}\n")
            
            # Validate that the output matches the intended proposal
            if current_agent.validate_output_matches_intent(response, intended_proposal):
                print(f"{Fore.GREEN}[OK] Deterministic agent output validated{Fore.RESET}")
                
                # Use domain to parse and register
                action = self.domain.parse_agent_action(current_agent_num, response)
                
                # Extra check: does action match intent?
                # Ideally yes if validation passed.
                
                if self.domain.is_valid_action(action):
                    print(f"{Fore.YELLOW}[OK] Valid deterministic action from {agent_name}:{Fore.RESET}")
                    res = self.domain.apply_action(action, current_agent_num)
                    
                    # Sync legacy tracker
                    if self.domain_type == "multi_item" and hasattr(action, "offer_content") and hasattr(action.offer_content, "is_valid"):
                        self.allocation_tracker.update_proposal(round_obj.round_number, current_agent_num, action.offer_content)
                        
                    return response, True
                else:
                    print(f"{Fore.RED}[FAIL] Deterministic action parsing failed or invalid in domain{Fore.RESET}")
            else:
                print(f"{Fore.RED}[FAIL] Deterministic agent output validation failed{Fore.RESET}")
            
            retry_count += 1
            if retry_count <= max_retries:
                print(f"{Fore.YELLOW}[RETRY] Retrying deterministic agent...{Fore.RESET}")
        
        print(f"{Fore.RED}[FAIL] Deterministic {agent_name} failed validation after {max_retries} retries{Fore.RESET}")
        return response, False
        return response, False
    
    def _create_deterministic_proposal_instruction(self, intended_proposal: Dict) -> str:
        """
        Create instruction for deterministic agent to make specific proposal.
        """
        if self.domain_type == "price":
            # Single Issue Price Domain
            action = intended_proposal.get("action")
            if action == "OFFER":
                price = intended_proposal.get("price")
                return f"""Please make the following offer: "OFFER {price}"
                
Present this offer naturally but be extremely brief (1-2 sentences)."""
            elif action == "ACCEPT":
                 return """Please accept the current offer by saying "ACCEPT". Be extremely brief."""
            return ""

        # Default: Multi-Item
        agent1_items = intended_proposal.get("agent1", [])
        agent2_items = intended_proposal.get("agent2", [])
        
        instruction = f"""Please make the following proposal in your negotiation style:

PROPOSAL {{
  "agent1": {agent1_items},
  "agent2": {agent2_items}
}}

Present this proposal naturally as if you determined it through your own strategic thinking. Explain briefly why you think this allocation makes sense."""
        
        return instruction
    
    def _generate_proposal_feedback(self, error_message: str, available_items: List[str]) -> str:
        """
        Generate helpful feedback for agents when they make invalid proposals.
        """
        feedback = f"Your previous proposal was invalid: {error_message}\n\n"
        feedback += "Please make a new proposal using the correct format:\n"
        feedback += f"Available items for this round: {available_items}\n"
        feedback += "Use this exact format:\n\n"
        feedback += "PROPOSAL {\n"
        feedback += '  "agent1": ["ItemA", "ItemB"],\n'
        feedback += '  "agent2": ["ItemC", "ItemD"]\n'
        feedback += "}\n\n"
        feedback += "Remember:\n"
        feedback += "- Every item must be allocated to exactly one agent\n"
        feedback += "- Use the exact item names provided\n"
        feedback += "- The JSON must be valid and complete"
        
        return feedback

    async def _process_agent_turn(self, current_agent: BaseAgent, current_agent_num: int, 
                                current_color: str, turn_count: int, round_obj: Round, 
                                available_items: List[str]) -> str:
        """
        Process a single agent's turn in the negotiation (legacy method for compatibility).
        Returns the agent's response.
        """
        response, _ = await self._process_agent_turn_with_retry(
            current_agent, current_agent_num, current_color, turn_count, 
            round_obj, available_items, max_retries=0  # No retries in legacy mode
        )
        return response
    
    def _log_round_completion(self, round_obj: Round, round_duration: float, success: bool):
        """
        Log the round completion to CSV and display status.
        Always logs rounds, even when consensus wasn't reached.
        """
        try:
            # Extract agent parameters if any agent is deterministic
            agent_params = self._extract_agent_parameters()
            
            # Determine final allocation and consensus status
            final_allocation = {}
            reached_consensus = False
            
            if success:
                if self.domain_type == "multi_item" and round_obj.final_allocation:
                    final_allocation = round_obj.final_allocation
                    reached_consensus = True
                    final_proposer = getattr(round_obj, 'final_proposer', None)
                elif self.domain_type in ["price", "single_issue_price"]:
                    outcome = self.domain.get_outcome()
                    if outcome.get("agreement"):
                        reached_consensus = True
                        final_allocation = {"price": outcome.get("agreement_price")}

            if not reached_consensus:
                if self.domain_type == "multi_item" and round_obj.items:
                    # No consensus reached - use fallback allocation
                    # Recompute fallback allocation using welfare-based ranking
                    items = round_obj.items
                    def all_possible_allocations(items):
                        n = len(items)
                        allocations = []
                        for bits in range(2**n):
                            a1, a2 = [], []
                            for i in range(n):
                                if (bits >> i) & 1:
                                    a1.append(items[i].name)
                                else:
                                    a2.append(items[i].name)
                            allocations.append({'agent1': a1, 'agent2': a2})
                        return allocations

                    def score_allocation_welfare(allocation, items):
                        score = 0.0
                        for item in items:
                            if item.name in allocation['agent1']:
                                score += item.agent1Value
                            elif item.name in allocation['agent2']:
                                score += item.agent2Value
                        return score

                    allAllocations = all_possible_allocations(items)
                    rankedAllocations = sorted(allAllocations, key=lambda alloc: score_allocation_welfare(alloc, items))
                    fallbackIndexPercent = 0.10
                    fallbackIndex = int(fallbackIndexPercent * len(rankedAllocations))
                    fallbackIndex = min(max(fallbackIndex, 0), len(rankedAllocations)-1)
                    fallbackAlloc = rankedAllocations[fallbackIndex]
                    final_allocation = fallbackAlloc
                    reached_consensus = False
                    final_proposer = None
                    round_obj.final_allocation = final_allocation
                    round_obj.final_proposer = final_proposer
                else:
                    # Price domain impasse or other
                    reached_consensus = False
                    final_allocation = {}
            
            # Prepare domain log fields
            # Helper to allow serialization of Item objects
            def make_serializable(ctx):
                import dataclasses
                if not isinstance(ctx, dict): return ctx
                new_ctx = {}
                for k, v in ctx.items():
                    if isinstance(v, list):
                        new_list = []
                        for item in v:
                            if dataclasses.is_dataclass(item):
                                new_list.append(dataclasses.asdict(item))
                            else:
                                new_list.append(item)
                        new_ctx[k] = new_list
                    elif dataclasses.is_dataclass(v):
                        new_ctx[k] = dataclasses.asdict(v)
                    else:
                         new_ctx[k] = v
                return new_ctx

            extra_kwargs = {
                "domain_name": self.domain_type,
                "domain_public_context": make_serializable(self.domain.get_public_context()),
                "agent1_private_context": make_serializable(self.domain.get_private_context(1)),
                "agent2_private_context": make_serializable(self.domain.get_private_context(2)),
                "outcome_details": self.domain.get_outcome()
            }
            if self.domain_type == "price":
                 # Use Specialized Price Logger
                 self.csv_logger.log_price_round(
                     round_obj=round_obj, 
                     outcome_details=self.domain.get_outcome(),
                     duration=round_duration,
                     agent1_type=self.agent1_type,
                     agent2_type=self.agent2_type,
                     domain_context=extra_kwargs
                 )
            else:
                # Default Multi-Item Logger
                log_entry = self.csv_logger.create_log_entry(
                    round_obj=round_obj,
                    round_duration=round_duration,
                    final_allocation=final_allocation,
                    allocation_tracker=self.allocation_tracker,
                    total_rounds=self.num_rounds,
                    agent1_type=self.agent1_type,
                    agent2_type=self.agent2_type,
                    reached_consensus=reached_consensus,
                    **agent_params,
                    **extra_kwargs
                )
                self.csv_logger.log_round(log_entry)
            
            if reached_consensus:
                print(f"{Fore.GREEN}[CSV] Round {round_obj.round_number} logged to CSV (Consensus: YES, Duration: {round_duration:.2f}s, Turns: {len(round_obj.conversation_history)}){Fore.RESET}")
            else:
                print(f"{Fore.YELLOW}[CSV] Round {round_obj.round_number} logged to CSV (Consensus: NO - Turn limit reached, Duration: {round_duration:.2f}s, Turns: {len(round_obj.conversation_history)}){Fore.RESET}")
                
        except Exception as e:
            print(f"{Fore.RED}[LOGGING ERROR] Failed to log round {round_obj.round_number}: {e}{Fore.RESET}")
        
        print(f"\n{Fore.CYAN}--End Round {round_obj.round_number}--{Fore.RESET}\n")

    def _extract_agent_parameters(self) -> dict:
        """
        Extract agent parameters for logging from all agent types.
        Returns a dict with all agent parameters for logging.
        """
        # Import here to avoid circular import
        from src.agents.boulware_agent import BoulwareAgent
        from src.agents.fixed_price_agent import FixedPriceAgent
        from src.agents.price_boulware_agent import PriceBoulwareAgent
        from src.agents.price_fixed_agent import PriceFixedAgent
        
        # Initialize all parameters as None
        agent_params = {
            'boulware_initial_threshold': None,
            'boulware_min_threshold': None,
            'boulware_final_threshold': None,
            'fixed_price_threshold': None,
            'price_fixed_margin': None,
            'price_boulware_beta': None
        }
        
        # Check both agents for parameters
        for agent in [self.agent1, self.agent2]:
            if isinstance(agent, BoulwareAgent):
                # Get Boulware parameters from the first Boulware agent found
                agent_params['boulware_initial_threshold'] = agent.initial_threshold
                agent_params['boulware_min_threshold'] = agent.min_threshold
                agent_params['boulware_final_threshold'] = agent.current_threshold
            elif isinstance(agent, FixedPriceAgent):
                # Get Fixed Price parameters from the first Fixed Price agent found
                agent_params['fixed_price_threshold'] = agent.fixed_threshold
            elif isinstance(agent, PriceFixedAgent):
                # Get PriceFixed parameters
                agent_params['price_fixed_margin'] = agent.fixed_margin
            elif isinstance(agent, PriceBoulwareAgent):
                # Get PriceBoulware parameters
                agent_params['price_boulware_beta'] = agent.beta

        return agent_params
    
    def _get_agent_name(self, agent_num: int) -> str:
        """
        Get the display name for an agent based on domain type.
        """
        if self.domain_type in ["price", "single_issue_price"]:
            if agent_num == 1:
                return "Buyer (Agent 1)"
            elif agent_num == 2:
                return "Seller (Agent 2)"
        return f"Agent {agent_num}"

    async def run_round(self, round_number: int) -> Round:
        """
        Execute a single round of negotiation.
        """
        # Start timing the round
        round_start_time = time.time()
        
        # Prepare round setup
        items, starting_agent, round_obj = self._prepare_round_setup(round_number)
        
        # Display round information
        self._display_round_info(round_number, items, starting_agent)
        
        # Prepare agent contexts
        self._prepare_agent_contexts(round_obj)
        
        # Initialize allocation tracking for this round
        self.allocation_tracker.initialize_round(round_number)
        available_items = [item.name for item in items]
        
        # Execute negotiation loop
        success = await self._execute_negotiation_loop(round_obj, available_items)
        
        if not success:
            print(f"{Fore.RED}Round {round_number} ended without agreement (max turns reached).{Fore.RESET}")
        
        # Calculate round duration and log completion
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        self._log_round_completion(round_obj, round_duration, success)
        
        return round_obj
    
    async def run_negotiation(self):
        """
        Run the complete negotiation session with all rounds.
        """
        print(f"{Fore.MAGENTA}Starting Negotiation Session: {self.num_rounds} rounds{Fore.RESET}")
        
        for round_num in range(1, self.num_rounds + 1):
            round_obj = await self.run_round(round_num)
            self.rounds.append(round_obj)
            
            # TODO: Extract and score final allocations
            # For now, just store the round
            
        print(f"\n{Fore.MAGENTA}{'='*50}")
        print(f"Negotiation Session Complete!")
        print(f"{'='*50}{Fore.RESET}")
        print(f"{Fore.GREEN}[CSV] Session data logged to: {self.csv_logger.get_filepath()}{Fore.RESET}")
        
        # TODO: Calculate and display final scores
        self.display_results()
    
    def display_results(self):
        """
        Display the results of all rounds with comprehensive analysis.
        """
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"=== NEGOTIATION SESSION RESULTS ===")
        print(f"{'='*60}{Fore.RESET}")
        
        completed_rounds = [r for r in self.rounds if r.is_complete]
        incomplete_rounds = [r for r in self.rounds if not r.is_complete]
        
        print(f"\n{Fore.CYAN}SESSION SUMMARY:{Fore.RESET}")
        print(f"Total Rounds: {len(self.rounds)}")
        print(f"Completed Rounds: {len(completed_rounds)}")
        print(f"Incomplete Rounds: {len(incomplete_rounds)}")
        
        if incomplete_rounds:
            print(f"{Fore.RED}Incomplete rounds: {[r.round_number for r in incomplete_rounds]}{Fore.RESET}")
        
        # Calculate total scores across all completed rounds
        total_agent1_value = 0.0
        total_agent2_value = 0.0
        
        print(f"\n{Fore.CYAN}ROUND-BY-ROUND ANALYSIS:{Fore.RESET}")
        print("="*60)
        
        for round_obj in completed_rounds:
            print(f"\n{Fore.GREEN}Round {round_obj.round_number}:{Fore.RESET}")
            
            # Get final allocation from allocation tracker
            final_allocation = self.allocation_tracker.get_final_allocation(round_obj.round_number)
            
            if final_allocation:
                # Calculate basic values for display
                agent1_value = sum(
                    item.agent1Value for item in round_obj.items 
                    if item.name in final_allocation.get('agent1', [])
                )
                agent2_value = sum(
                    item.agent2Value for item in round_obj.items 
                    if item.name in final_allocation.get('agent2', [])
                )
                
                # Add to totals
                total_agent1_value += agent1_value
                total_agent2_value += agent2_value
                
                # Display round allocation (without detailed analysis)
                print(f"  Final Allocation:")
                print(f"    Agent 1: {final_allocation['agent1']} (Value: {agent1_value:.2f})")
                print(f"    Agent 2: {final_allocation['agent2']} (Value: {agent2_value:.2f})")
                print(f"  Total Welfare: {agent1_value + agent2_value:.2f}")
            else:
                print(f"  {Fore.RED}No final allocation recorded{Fore.RESET}")
        
        # Display overall session statistics
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"=== OVERALL SESSION STATISTICS ===")
        print(f"{'='*60}{Fore.RESET}")
        
        if completed_rounds:
            print(f"\n{Fore.CYAN}CUMULATIVE SCORES:{Fore.RESET}")
            print(f"Agent 1 Total Value: {total_agent1_value:.2f}")
            print(f"Agent 2 Total Value: {total_agent2_value:.2f}")
            print(f"Total Session Welfare: {total_agent1_value + total_agent2_value:.2f}")
            print(f"Average Welfare per Round: {(total_agent1_value + total_agent2_value) / len(completed_rounds):.2f}")
            
            print(f"\n{Fore.CYAN}EFFICIENCY METRICS:{Fore.RESET}")
            print(f"Run analyze_results.py for detailed Pareto optimality analysis")
        
        # Detailed analysis for each completed round
        # Summary message
        print(f"\n{Fore.MAGENTA}Analysis complete! Use analyze_results.py to calculate detailed metrics.{Fore.RESET}")
        print(f"Raw data saved to: {self.csv_logger.get_filename()}")
        
        self.total_scores = {"agent1": total_agent1_value, "agent2": total_agent2_value}

def parse_arguments():
    """
    Parse command line arguments for the negotiation system.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Negotiation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Agent configuration
    available_agents = AgentFactory.get_available_types()
    parser.add_argument(
        "--agent1", 
        type=str, 
        choices=available_agents,
        default="default",
        help="Type of agent 1"
    )
    parser.add_argument(
        "--agent2", 
        type=str, 
        choices=available_agents,
        default="default", 
        help="Type of agent 2"
    )
    
    # Negotiation parameters
    parser.add_argument(
        "--domain",
        type=str,
        choices=["multi_item", "price"],
        default="multi_item",
        help="Negotiation domain type"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=DEFAULT_NUM_ROUNDS,
        help="Number of negotiation rounds"
    )
    parser.add_argument(
        "--items", 
        type=int, 
        default=DEFAULT_ITEMS_PER_ROUND,
        help="Number of items per round"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Model name to use for agents"
    )
    
    return parser.parse_args()

# Main execution
async def main(agent1_type="default", agent2_type="default", num_rounds=DEFAULT_NUM_ROUNDS, 
              items_per_round=DEFAULT_ITEMS_PER_ROUND, model_name=DEFAULT_MODEL_NAME, 
              domain="multi_item", **kwargs):
    """
    Entry point for the negotiation system.
    Run a single negotiation session with the specified parameters.
    """
    print(f"{Fore.MAGENTA}Multi-Agent Negotiation System{Fore.RESET}")
    print(f"{Fore.CYAN}Available agent types: {AgentFactory.get_available_types()}{Fore.RESET}")
    
    # Display current configuration
    print(f"\n{Fore.YELLOW}=== Configuration ==={Fore.RESET}")
    print(f"Domain: {domain}")
    print(f"Agent 1: {agent1_type}")
    print(f"Agent 2: {agent2_type}") 
    print(f"Rounds: {num_rounds}")
    print(f"Items per round: {items_per_round}")
    print(f"Model: {model_name}")
    print(f"Extra Args: {kwargs}")
    
    # Run negotiation session
    print(f"\n{Fore.YELLOW}=== Running Negotiation: {agent1_type} vs {agent2_type} ==={Fore.RESET}")
    session = NegotiationSession(
        num_rounds=num_rounds,
        items_per_round=items_per_round,
        model_name=model_name,
        agent1_type=agent1_type,
        agent2_type=agent2_type,
        domain_type=domain,
        **kwargs
    )
    await session.run_negotiation()
    return session


async def run_specific_matchup(agent1_type: str, agent2_type: str, num_rounds: int = 3,
                              agent1_config: Optional[Dict] = None, agent2_config: Optional[Dict] = None):
    """
    Run a specific agent matchup for testing.
    
    Args:
        agent1_type: Type of agent 1
        agent2_type: Type of agent 2
        num_rounds: Number of rounds to run
        agent1_config: Configuration for agent 1
        agent2_config: Configuration for agent 2
    """
    session = NegotiationSession(
        num_rounds=num_rounds,
        items_per_round=DEFAULT_ITEMS_PER_ROUND,
        agent1_type=agent1_type,
        agent2_type=agent2_type,
        agent1_config=agent1_config,
        agent2_config=agent2_config
    )
    await session.run_negotiation()
    return session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Agent Negotiation System")
    parser.add_argument("--domain", type=str, default="multi_item", choices=["multi_item", "price"], help="Negotiation domain type")
    parser.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of rounds to simulate")
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS_PER_ROUND, help="Number of items per round")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="LLM model name")
    parser.add_argument("--agent1", type=str, default="default", help="Type of Agent 1")
    parser.add_argument("--agent2", type=str, default="default", help="Type of Agent 2")
    parser.add_argument("--dataset_mode", action="store_true", help="Enable offline RL dataset generation (no LLM)")
    parser.add_argument("--no_llm", action="store_true", help="Alias for dataset_mode")
    parser.add_argument("--dataset_out", type=str, default="datasets/price_domain.jsonl", help="Output path for dataset")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes (rounds) for dataset generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max_turns", type=int, default=20, help="Max turns per episode in dataset mode")

    args, unknown = parser.parse_known_args()
    
    # Handle known args that were previously in parse_arguments or implicit
    # Note: Previous code used parse_arguments function, replacing main block entirely
    
    # Handle unknown args safely
    extra_args = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip('-')
        if i + 1 < len(unknown):
            extra_args[key] = unknown[i+1]

    # Combine flags
    dataset_mode = args.dataset_mode or args.no_llm
    
    # Determine number of rounds/episodes
    if dataset_mode:
        num_rounds = args.num_episodes if args.num_episodes is not None else args.rounds
    else:
        num_rounds = args.rounds
        
    print("Multi-Agent Negotiation System")
    if not dataset_mode:
        print(f"Available agent types: {AgentFactory.get_available_types()}")
    
    print(f"\n=== Configuration ===")
    print(f"Domain: {args.domain}")
    
    if dataset_mode:
        print(f"Mode: DATASET GENERATION (No LLM)")
        print(f"Episodes: {num_rounds}")
        print(f"Output: {args.dataset_out}")
    else:
        print(f"Agent 1: {args.agent1}")
        print(f"Agent 2: {args.agent2}")
        print(f"Rounds: {num_rounds}")
        print(f"Items per round: {args.items}")
        print(f"Model: {args.model}")
        
    print(f"Extra Args: {extra_args}")
    
    # Agent configurations
    agent1_config = {}
    agent2_config = {}
    
    # Setup domain kwargs including max_turns
    domain_kwargs = extra_args.copy()
    domain_kwargs["max_turns"] = args.max_turns

    session = NegotiationSession(
        num_rounds=num_rounds,
        items_per_round=args.items,
        model_name=args.model,
        agent1_type=args.agent1,
        agent2_type=args.agent2,
        agent1_config=agent1_config,
        agent2_config=agent2_config,
        domain_type=args.domain,
        dataset_mode=dataset_mode,
        dataset_out=args.dataset_out,
        seed=args.seed,
        **domain_kwargs
    )
    
    if dataset_mode:
        session.run()
    else:
        asyncio.run(session.run_negotiation())
