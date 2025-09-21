import asyncio
import random
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.agents.ollamaAgentModule import Agent
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
    def __init__(self, num_rounds: int, items_per_round: int = DEFAULT_ITEMS_PER_ROUND, model_name: str = DEFAULT_MODEL_NAME):
        self.num_rounds = num_rounds
        self.items_per_round = items_per_round
        self.model_name = model_name.replace(":", "_")  # Clean up for filename
        self.rounds = []
        self.total_scores = {"agent1": 0.0, "agent2": 0.0}
        
        # Initialize CSV logger
        self.csv_logger = CSVLogger(self.model_name, items_per_round)
        print(f"{Fore.GREEN}📊 CSV logging to: {self.csv_logger.get_filename()}{Fore.RESET}")
        
        # Initialize agents
        self.agent1 = Agent(DEFAULT_MODEL_NAME, SYSTEM_INSTRUCTIONS_FILE)
        self.agent2 = Agent(DEFAULT_MODEL_NAME, SYSTEM_INSTRUCTIONS_FILE)
        
        # Initialize proposal tracking components
        self.message_parser = MessageParser()
        self.allocation_tracker = AllocationTracker()
        
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
        agent1_context = f"""
--Round Start--
You are Agent 1 in a negotiation. Your goal is to maximize your own value.
Items: {round_obj.get_items_for_agent(1)}

You can only see your own values (agent1Value). The other agent's values are unknown to you.
Negotiate with Agent 2 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
"""
        
        agent2_context = f"""
--Round Start--
You are Agent 2 in a negotiation. Your goal is to maximize your own value.
Items: {round_obj.get_items_for_agent(2)}

You can only see your own values (agent2Value). The other agent's values are unknown to you.
Negotiate with Agent 1 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
"""
        
        # Add context to agents
        self.agent1.addToMemory('system', agent1_context)
        self.agent2.addToMemory('system', agent2_context)
        
        # Give the starting agent an initial prompt to begin the negotiation
        if round_obj.starting_agent == 1:
            self.agent1.addToMemory('user', "Please begin the negotiation by making your opening proposal.")
        else:
            self.agent2.addToMemory('user', "Please begin the negotiation by making your opening proposal.")
    
    async def _execute_negotiation_loop(self, round_obj: Round, available_items: List[str]) -> bool:
        """
        Execute the main negotiation loop between agents with improved error handling.
        Returns True if agreement reached, False if max turns exceeded.
        """
        max_turns = MAX_TURNS_PER_ROUND  # Prevent infinite loops
        turn_count = 0
        current_agent_num = round_obj.starting_agent
        max_retries_per_turn = MAX_RETRIES_PER_INVALID_PROPOSAL  # Maximum retries for invalid proposals
        
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
            
            # Process agent turn with retry logic for invalid proposals
            response, turn_successful = await self._process_agent_turn_with_retry(
                current_agent, current_agent_num, current_color, turn_count, 
                round_obj, available_items, max_retries_per_turn
            )
            
            if not turn_successful:
                print(f"{Fore.RED}⚠️  Agent {current_agent_num} failed to provide a valid response after {max_retries_per_turn} retries. Ending round.{Fore.RESET}")
                break
            
            # Add current agent's response to other agent's memory
            other_agent.addToMemory('user', f"Agent {current_agent_num}: {response}")
            
            # Store conversation history
            round_obj.conversation_history.append((current_agent_num, response))
            
            # Check if round completed due to agreement
            if round_obj.is_complete:
                break
                
            # Switch to other agent
            current_agent_num = other_agent_num
        
        return round_obj.is_complete
    
    async def _process_agent_turn_with_retry(self, current_agent: Agent, current_agent_num: int, 
                                           current_color: str, turn_count: int, round_obj: Round, 
                                           available_items: List[str], max_retries: int) -> Tuple[str, bool]:
        """
        Process a single agent's turn with retry logic for invalid proposals.
        Returns (agent_response, turn_successful).
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            # Generate response from current agent
            if retry_count == 0:
                print(f"{current_color}Agent {current_agent_num}'s turn (Turn {turn_count}):{Fore.RESET}")
            else:
                print(f"{current_color}Agent {current_agent_num} retry {retry_count} (Turn {turn_count}):{Fore.RESET}")
            
            response = await current_agent.generateResponse()
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
                        current_agent.addToMemory('user', feedback)
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

    async def _process_agent_turn(self, current_agent: Agent, current_agent_num: int, 
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
        """
        if success and round_obj.final_allocation:
            try:
                log_entry = self.csv_logger.create_log_entry(
                    round_obj=round_obj,
                    round_duration=round_duration,
                    final_allocation=round_obj.final_allocation,
                    allocation_tracker=self.allocation_tracker,
                    total_rounds=self.num_rounds
                )
                self.csv_logger.log_round(log_entry)
                print(f"{Fore.GREEN}📊 Round {round_obj.round_number} logged to CSV (Duration: {round_duration:.2f}s, Turns: {len(round_obj.conversation_history)}){Fore.RESET}")
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to log round {round_obj.round_number}: {e}{Fore.RESET}")
        else:
            print(f"{Fore.YELLOW}⚠️  Round {round_obj.round_number} not logged (incomplete or no allocation){Fore.RESET}")
        
        print(f"\n{Fore.CYAN}--End Round {round_obj.round_number}--{Fore.RESET}\n")
    
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

# Main execution
async def main():
    """
    Entry point for the negotiation system.
    """
    # Create and run a negotiation session
    session = NegotiationSession(num_rounds=DEFAULT_NUM_ROUNDS, items_per_round=DEFAULT_ITEMS_PER_ROUND)
    await session.run_negotiation()

if __name__ == "__main__":
    asyncio.run(main())
