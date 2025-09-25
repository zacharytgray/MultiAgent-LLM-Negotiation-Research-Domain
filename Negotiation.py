import asyncio
import random
import time
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.agents.agent_factory import AgentFactory, AgentConfig
from colorama import Fore, init
from src.core.Item import Item
from src.core.Round import Round
from src.utils.MessageParser import MessageParser
from src.utils.AllocationTracker import AllocationTracker
from src.utils.CSVLogger import CSVLogger
from config.settings import *

# Initialize colorama
init(autoreset=True)

class NegotiationSession:
    """
    Main class that manages the entire negotiation session across multiple rounds.
    """
    def __init__(self, num_rounds: int = DEFAULT_NUM_ROUNDS, items_per_round: int = DEFAULT_ITEMS_PER_ROUND, 
                 model_name: str = DEFAULT_MODEL_NAME, agent1_type: str = "default", 
                 agent2_type: str = "default", agent1_config: Optional[Dict] = None, 
                 agent2_config: Optional[Dict] = None):
        self.num_rounds = num_rounds
        self.items_per_round = items_per_round
        self.model_name = model_name.replace(":", "_")  # Clean up for filename
        self.agent1_type = agent1_type
        self.agent2_type = agent2_type
        self.rounds = []
        self.total_scores = {"agent1": 0.0, "agent2": 0.0}
        
        # Initialize CSV logger
        session_label = f"{agent1_type}_vs_{agent2_type}"
        self.csv_logger = CSVLogger(f"{self.model_name}_{session_label}", items_per_round)
        print(f"{Fore.GREEN}📊 CSV logging to: {self.csv_logger.get_filename()}{Fore.RESET}")
        
        # Get agent configurations
        if agent1_config is None:
            agent1_config = AgentConfig.get_config_for_type(agent1_type)
        if agent2_config is None:
            agent2_config = AgentConfig.get_config_for_type(agent2_type)
        
        # Determine system instructions files for each agent
        agent1_instructions = self._get_system_instructions_file(agent1_type)
        agent2_instructions = self._get_system_instructions_file(agent2_type)
        
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
        
    def _get_system_instructions_file(self, agent_type: str) -> str:
        """
        Get the appropriate system instructions file for the given agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            str: Path to system instructions file
        """
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
    
    def _prepare_round_setup(self, round_number: int) -> Tuple[List[Item], int, Round]:
        """
        Prepare the basic setup for a round: items, starting agent, and round object.
        """
        # Reset agent memories to start fresh each round (preserving only system instructions)
        self.agent1.reset_memory()
        self.agent2.reset_memory()
        
        # Generate items for this round
        items = self.generate_random_items(round_number) 
        
        # Determine starting agent (alternates each round)
        starting_agent = 1 if round_number % 2 == 1 else 2
        
        # Create round object
        round_obj = Round(round_number, items, self.agent1, self.agent2, starting_agent)
        
        return items, starting_agent, round_obj
    
    def _display_round_info(self, round_number: int, items: List[Item], starting_agent: int):
        """
        Display round information to the console.
        """
        print(f"\n{Fore.CYAN}{'='*SEPARATOR_LENGTH}")
        print(f"Round {round_number} Starting")
        print(f"{'='*SEPARATOR_LENGTH}{Fore.RESET}")
        
        print(f"\n{Fore.YELLOW}Items for Round {round_number}:")
        for item in items:
            print(f"  {item.name}: Agent1={item.agent1Value}, Agent2={item.agent2Value}")
        print(f"Starting Agent: Agent {starting_agent}{Fore.RESET}\n")
    
    def _prepare_agent_contexts(self, round_obj: Round):
        """
        Prepare and set initial context for both agents.
        """
        # Set items for both agents
        self.agent1.set_items(round_obj.items)
        self.agent2.set_items(round_obj.items)
        
        agent1_context = f"""
--Round Start--
You are Agent 1 in a negotiation. Your goal is to maximize your own value.
Items: {self.agent1.get_agent_items_context()}

You can only see your own values (agent1Value). The other agent's values are unknown to you.
Negotiate with Agent 2 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
"""
        
        agent2_context = f"""
--Round Start--
You are Agent 2 in a negotiation. Your goal is to maximize your own value.
Items: {self.agent2.get_agent_items_context()}

You can only see your own values (agent2Value). The other agent's values are unknown to you.
Negotiate with Agent 1 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
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
            
            # Add time pressure message for non-deterministic agents when 5 turns remain
            turns_remaining = max_turns - turn_count
            if turns_remaining <= 5:
                pressure_message = (
                    f"⚠️  TIME PRESSURE ALERT: You have only {turns_remaining} turns remaining to reach an agreement! "
                    "There is a strict time limit in this negotiation. If you do not come to an agreement soon, "
                    "you will receive a fallback reward that is significantly worse than any outcome you would "
                    "receive if you make a deal. You must prioritize reaching an agreement quickly to avoid this penalty."
                )
                
                # Add pressure message to current agent if non-deterministic and not already sent
                if (not current_agent.should_make_deterministic_proposal(turn_count) and 
                    not pressure_message_sent[current_agent_num]):
                    print(f"{Fore.YELLOW}📢 Sending time pressure alert to Agent {current_agent_num} ({current_agent.agent_type if hasattr(current_agent, 'agent_type') else 'Unknown'}):{Fore.RESET}")
                    print(f"{Fore.YELLOW}    {pressure_message}{Fore.RESET}")
                    current_agent.add_to_memory('system', pressure_message)
                    pressure_message_sent[current_agent_num] = True
                
                # Add pressure message to other agent if non-deterministic and not already sent
                if (not other_agent.should_make_deterministic_proposal(turn_count) and 
                    not pressure_message_sent[other_agent_num]):
                    print(f"{Fore.YELLOW}📢 Sending time pressure alert to Agent {other_agent_num} ({other_agent.agent_type if hasattr(other_agent, 'agent_type') else 'Unknown'}):{Fore.RESET}")
                    print(f"{Fore.YELLOW}    {pressure_message}{Fore.RESET}")
                    other_agent.add_to_memory('system', pressure_message)
                    pressure_message_sent[other_agent_num] = True
            
            # Process agent turn with retry logic for invalid proposals
            response, turn_successful = await self._process_agent_turn_with_retry(
                current_agent, current_agent_num, current_color, turn_count, 
                round_obj, available_items, max_retries_per_turn
            )
            
            if not turn_successful:
                print(f"{Fore.RED}⚠️  Agent {current_agent_num} failed to provide a valid response after {max_retries_per_turn} retries. Ending round.{Fore.RESET}")
                break
            
            # Add current agent's response to other agent's memory
            other_agent.add_to_memory('user', f"Agent {current_agent_num}: {response}")
            
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
        
        while retry_count <= max_retries:
            # Generate response from current agent
            if retry_count == 0:
                print(f"{current_color}Agent {current_agent_num}'s turn (Turn {turn_count}):{Fore.RESET}")
            else:
                print(f"{current_color}Agent {current_agent_num} retry {retry_count} (Turn {turn_count}):{Fore.RESET}")
            
            response = await current_agent.generate_response()
            print(f"{current_color}Agent {current_agent_num}: {response}{Fore.RESET}\n")
            
            # Parse the response for formal proposals
            proposal = self.message_parser.extract_proposal(response, available_items)
            
            if proposal:
                if proposal.is_valid:
                    print(f"{Fore.YELLOW}✓ Valid proposal detected from Agent {current_agent_num}:{Fore.RESET}")
                    self.allocation_tracker.update_proposal(round_obj.round_number, current_agent_num, proposal)
                    break  # Valid proposal found, exit retry loop
                else:
                    print(f"{Fore.RED}✗ Invalid proposal from Agent {current_agent_num}: {proposal.error_message}{Fore.RESET}")
                    if retry_count < max_retries:
                        # Provide feedback to help the agent correct their mistake
                        feedback = self._generate_proposal_feedback(proposal.error_message, available_items)
                        current_agent.add_to_memory('user', feedback)
                        print(f"{Fore.YELLOW}🔄 Providing feedback and retrying...{Fore.RESET}")
                        retry_count += 1
                        continue
                    else:
                        print(f"{Fore.RED}❌ Agent {current_agent_num} exceeded maximum retries for valid proposal{Fore.RESET}")
                        return response, False
            else:
                # No proposal detected, this is acceptable (agent might just be negotiating)
                break
            
            retry_count += 1
        
        # Check for agreement (regardless of proposal validity)
        if self.message_parser.contains_agreement(response):
            print(f"{Fore.CYAN}Agent {current_agent_num} agreed!{Fore.RESET}")
            self.allocation_tracker.record_agreement(round_obj.round_number, current_agent_num)
            
            # Check if round is complete (both agents agreed to a valid proposal)
            if self.allocation_tracker.is_round_complete(round_obj.round_number):
                final_allocation = self.allocation_tracker.get_final_allocation(round_obj.round_number)
                print(f"{Fore.CYAN}Both agents have agreed! Round {round_obj.round_number} complete.{Fore.RESET}")
                print(f"{Fore.CYAN}Final allocation: {final_allocation}{Fore.RESET}")
                round_obj.is_complete = True
                round_obj.final_allocation = final_allocation
        
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
        
        # Get current proposal on the table
        current_proposal = self.allocation_tracker.get_current_proposal(round_obj.round_number)
        
        # Check if agent should accept current proposal
        if current_proposal and current_agent.should_accept_proposal(current_proposal, turn_count):
            print(f"{current_color}Agent {current_agent_num}'s turn (Turn {turn_count}) - Should Accept:{Fore.RESET}")
            
            # Instruct agent to accept
            accept_instruction = "The current proposal is acceptable to you. Please respond by agreeing to it and end your message with 'AGREE'."
            current_agent.add_to_memory('user', accept_instruction)
            
            response = await current_agent.generate_response()
            print(f"{current_color}Agent {current_agent_num}: {response}{Fore.RESET}\n")
            
            # Check for agreement
            if self.message_parser.contains_agreement(response):
                print(f"{Fore.CYAN}Agent {current_agent_num} agreed!{Fore.RESET}")
                self.allocation_tracker.record_agreement(round_obj.round_number, current_agent_num)
                
                # Check if round is complete
                if self.allocation_tracker.is_round_complete(round_obj.round_number):
                    final_allocation = self.allocation_tracker.get_final_allocation(round_obj.round_number)
                    print(f"{Fore.CYAN}Both agents have agreed! Round {round_obj.round_number} complete.{Fore.RESET}")
                    print(f"{Fore.CYAN}Final allocation: {final_allocation}{Fore.RESET}")
                    round_obj.is_complete = True
                    round_obj.final_allocation = final_allocation
            
            return response, True
        
        # Agent should make a deterministic proposal
        intended_proposal = current_agent.get_deterministic_proposal(turn_count)
        
        if not intended_proposal:
            print(f"{Fore.RED}❌ Agent {current_agent_num} could not generate deterministic proposal{Fore.RESET}")
            return "I'm unable to make a proposal at this time.", False
        
        # Prepare instruction for the agent
        proposal_instruction = self._create_deterministic_proposal_instruction(intended_proposal)
        
        # Retry logic for deterministic agents
        retry_count = 0
        
        while retry_count <= max_retries:
            if retry_count == 0:
                print(f"{current_color}Agent {current_agent_num}'s turn (Turn {turn_count}) - Deterministic:{Fore.RESET}")
            else:
                print(f"{current_color}Agent {current_agent_num} deterministic retry {retry_count} (Turn {turn_count}):{Fore.RESET}")
            
            # Give agent the deterministic proposal instruction
            current_agent.add_to_memory('user', proposal_instruction)
            
            response = await current_agent.generate_response()
            print(f"{current_color}Agent {current_agent_num}: {response}{Fore.RESET}\n")
            
            # Validate that the output matches the intended proposal
            if current_agent.validate_output_matches_intent(response, intended_proposal):
                print(f"{Fore.GREEN}✓ Deterministic agent output validated{Fore.RESET}")
                
                # Extract and register the proposal
                proposal = self.message_parser.extract_proposal(response, available_items)
                if proposal and proposal.is_valid:
                    print(f"{Fore.YELLOW}✓ Valid deterministic proposal from Agent {current_agent_num}:{Fore.RESET}")
                    self.allocation_tracker.update_proposal(round_obj.round_number, current_agent_num, proposal)
                    return response, True
                else:
                    print(f"{Fore.RED}✗ Deterministic proposal extraction failed{Fore.RESET}")
            else:
                print(f"{Fore.RED}✗ Deterministic agent output validation failed{Fore.RESET}")
            
            retry_count += 1
            if retry_count <= max_retries:
                print(f"{Fore.YELLOW}🔄 Retrying deterministic agent...{Fore.RESET}")
        
        print(f"{Fore.RED}❌ Deterministic Agent {current_agent_num} failed validation after {max_retries} retries{Fore.RESET}")
        return response, False
    
    def _create_deterministic_proposal_instruction(self, intended_proposal: Dict) -> str:
        """
        Create instruction for deterministic agent to make specific proposal.
        
        Args:
            intended_proposal: The allocation the agent should propose
            
        Returns:
            str: Instruction text for the agent
        """
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
            if success and round_obj.final_allocation:
                final_allocation = round_obj.final_allocation
                reached_consensus = True
                final_proposer = getattr(round_obj, 'final_proposer', None)
            else:
                # No consensus reached - use empty allocation
                final_allocation = {"agent1": [], "agent2": []}
                reached_consensus = False
                final_proposer = None
                # Set empty final_allocation on round_obj if it doesn't exist
                if not hasattr(round_obj, 'final_allocation') or not round_obj.final_allocation:
                    round_obj.final_allocation = final_allocation
                if not hasattr(round_obj, 'final_proposer'):
                    round_obj.final_proposer = final_proposer
            
            log_entry = self.csv_logger.create_log_entry(
                round_obj=round_obj,
                round_duration=round_duration,
                final_allocation=final_allocation,
                allocation_tracker=self.allocation_tracker,
                total_rounds=self.num_rounds,
                agent1_type=self.agent1_type,
                agent2_type=self.agent2_type,
                reached_consensus=reached_consensus,
                **agent_params
            )
            self.csv_logger.log_round(log_entry)
            
            if reached_consensus:
                print(f"{Fore.GREEN}📊 Round {round_obj.round_number} logged to CSV (Consensus: YES, Duration: {round_duration:.2f}s, Turns: {len(round_obj.conversation_history)}){Fore.RESET}")
            else:
                print(f"{Fore.YELLOW}📊 Round {round_obj.round_number} logged to CSV (Consensus: NO - Turn limit reached, Duration: {round_duration:.2f}s, Turns: {len(round_obj.conversation_history)}){Fore.RESET}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to log round {round_obj.round_number}: {e}{Fore.RESET}")
        
        print(f"\n{Fore.CYAN}--End Round {round_obj.round_number}--{Fore.RESET}\n")

    def _extract_agent_parameters(self) -> dict:
        """
        Extract agent parameters for logging from all agent types.
        Returns a dict with all agent parameters for logging.
        """
        # Import here to avoid circular import
        from src.agents.boulware_agent import BoulwareAgent
        from src.agents.fixed_price_agent import FixedPriceAgent
        
        # Initialize all parameters as None
        agent_params = {
            'boulware_initial_threshold': None,
            'boulware_decrease_rate': None,
            'boulware_min_threshold': None,
            'boulware_final_threshold': None,
            'fixed_price_threshold': None
        }
        
        # Check both agents for parameters
        for agent in [self.agent1, self.agent2]:
            if isinstance(agent, BoulwareAgent):
                # Get Boulware parameters from the first Boulware agent found
                agent_params['boulware_initial_threshold'] = agent.initial_threshold
                agent_params['boulware_decrease_rate'] = agent.decrease_rate
                agent_params['boulware_min_threshold'] = agent.min_threshold
                agent_params['boulware_final_threshold'] = agent.current_threshold
                break  # Use parameters from first Boulware agent found
            elif isinstance(agent, FixedPriceAgent):
                # Get Fixed Price parameters from the first Fixed Price agent found
                agent_params['fixed_price_threshold'] = agent.fixed_threshold
                break  # Use parameters from first Fixed Price agent found
                
        return agent_params
    
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
        print(f"{Fore.GREEN}📊 Session data logged to: {self.csv_logger.get_filepath()}{Fore.RESET}")
        
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
    available_agents = ["default", "boulware", "fixed_price", "charming", "rude"]
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
              items_per_round=DEFAULT_ITEMS_PER_ROUND, model_name=DEFAULT_MODEL_NAME):
    """
    Entry point for the negotiation system.
    Run a single negotiation session with the specified parameters.
    """
    print(f"{Fore.MAGENTA}🤖 Multi-Agent Negotiation System{Fore.RESET}")
    print(f"{Fore.CYAN}Available agent types: {AgentFactory.get_available_types()}{Fore.RESET}")
    
    # Display current configuration
    print(f"\n{Fore.YELLOW}=== Configuration ==={Fore.RESET}")
    print(f"Agent 1: {agent1_type}")
    print(f"Agent 2: {agent2_type}") 
    print(f"Rounds: {num_rounds}")
    print(f"Items per round: {items_per_round}")
    print(f"Model: {model_name}")
    
    # Run negotiation session
    print(f"\n{Fore.YELLOW}=== Running Negotiation: {agent1_type} vs {agent2_type} ==={Fore.RESET}")
    session = NegotiationSession(
        num_rounds=num_rounds,
        items_per_round=items_per_round,
        model_name=model_name,
        agent1_type=agent1_type,
        agent2_type=agent2_type
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
    # Parse command line arguments
    args = parse_arguments()
    
    # Run negotiation with CLI arguments
    asyncio.run(main(
        agent1_type=args.agent1,
        agent2_type=args.agent2,
        num_rounds=args.rounds,
        items_per_round=args.items,
        model_name=args.model
    ))
