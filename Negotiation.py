import asyncio
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ollamaAgentModule import Agent
from colorama import Fore, init
from Item import Item
from Round import Round
from MessageParser import MessageParser
from AllocationTracker import AllocationTracker
from Scoring.ParetoAnalyzer import ParetoAnalyzer

# Initialize colorama
init(autoreset=True)

class NegotiationSession:
    """
    Main class that manages the entire negotiation session across multiple rounds.
    """
    def __init__(self, num_rounds: int, items_per_round: int = 4):
        self.num_rounds = num_rounds
        self.items_per_round = items_per_round
        self.rounds = []
        self.total_scores = {"agent1": 0.0, "agent2": 0.0}
        
        # Initialize agents
        self.agent1 = Agent("gemma3:12b", "systemInstructions.txt")
        self.agent2 = Agent("gemma3:12b", "systemInstructions.txt")
        
        # Initialize proposal tracking components
        self.message_parser = MessageParser()
        self.allocation_tracker = AllocationTracker()
        
    def generate_random_items(self, round_number: int) -> List[Item]:
        """
        Generate random items for a round with random values for each agent.
        """
        items = []
        item_names = ["ItemA", "ItemB", "ItemC", "ItemD", "ItemE", "ItemF", "ItemG", "ItemH"]
        
        for i in range(self.items_per_round):
            name = item_names[i] if i < len(item_names) else f"Item{i+1}"
            agent1_value = round(random.uniform(0.0, 1.0), 1)
            agent2_value = round(random.uniform(0.0, 1.0), 1)
            items.append(Item(name, agent1_value, agent2_value))
            
        return items
    
    async def run_round(self, round_number: int) -> Round:
        """
        Execute a single round of negotiation.
        """
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"Round {round_number} Starting")
        print(f"{'='*50}{Fore.RESET}")
        
        # Reset agent memories to start fresh each round (preserving only system instructions)
        self.agent1.reset_memory()
        self.agent2.reset_memory()
        
        # Generate items for this round
        items = self.generate_random_items(round_number) 
        
        # Determine starting agent (alternates each round)
        starting_agent = 1 if round_number % 2 == 1 else 2
        
        # Create round
        round_obj = Round(round_number, items, self.agent1, self.agent2, starting_agent)
        
        # Display items to console (for debugging)
        print(f"\n{Fore.YELLOW}Items for Round {round_number}:")
        for item in items:
            print(f"  {item.name}: Agent1={item.agent1Value}, Agent2={item.agent2Value}")
        print(f"Starting Agent: Agent {starting_agent}{Fore.RESET}\n")
        
        # Prepare initial context for agents
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
        if starting_agent == 1:
            self.agent1.addToMemory('user', "Please begin the negotiation by making your opening proposal.")
        else:
            self.agent2.addToMemory('user', "Please begin the negotiation by making your opening proposal.")
        
        # Start the negotiation
        max_turns = 20  # Prevent infinite loops
        turn_count = 0
        
        # Initialize allocation tracking for this round
        self.allocation_tracker.initialize_round(round_number)
        available_items = [item.name for item in items]
        
        current_agent_num = starting_agent
        
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
            
            # Generate response from current agent
            print(f"{current_color}Agent {current_agent_num}'s turn (Turn {turn_count}):{Fore.RESET}")
            response = await current_agent.generateResponse()
            print(f"{current_color}Agent {current_agent_num}: {response}{Fore.RESET}\n")
            
            # Parse the response for formal proposals
            proposal = self.message_parser.extract_proposal(response, available_items)
            if proposal:
                if proposal.is_valid:
                    print(f"{Fore.YELLOW}✓ Valid proposal detected from Agent {current_agent_num}:{Fore.RESET}")
                    print(f"{Fore.YELLOW}  Agent 1: {proposal.agent1_items}{Fore.RESET}")
                    print(f"{Fore.YELLOW}  Agent 2: {proposal.agent2_items}{Fore.RESET}")
                    self.allocation_tracker.update_proposal(round_number, current_agent_num, proposal)
                else:
                    print(f"{Fore.RED}✗ Invalid proposal from Agent {current_agent_num}: {proposal.error_message}{Fore.RESET}")
            
            # Check for agreement
            if self.message_parser.contains_agreement(response):
                print(f"{Fore.CYAN}Agent {current_agent_num} agreed!{Fore.RESET}")
                self.allocation_tracker.record_agreement(round_number, current_agent_num)
                
                # Check if round is complete (both agents agreed to a valid proposal)
                if self.allocation_tracker.is_round_complete(round_number):
                    final_allocation = self.allocation_tracker.get_final_allocation(round_number)
                    print(f"{Fore.CYAN}Both agents have agreed! Round {round_number} complete.{Fore.RESET}")
                    print(f"{Fore.CYAN}Final allocation: {final_allocation}{Fore.RESET}")
                    round_obj.is_complete = True
                    round_obj.final_allocation = final_allocation
                    break
            
            # Add current agent's response to other agent's memory
            other_agent.addToMemory('user', f"Agent {current_agent_num}: {response}")
            
            # Store conversation history
            round_obj.conversation_history.append((current_agent_num, response))
            
            # Switch to other agent
            current_agent_num = other_agent_num
        
        if not round_obj.is_complete:
            print(f"{Fore.RED}Round {round_number} ended without agreement (max turns reached).{Fore.RESET}")
        
        print(f"\n{Fore.CYAN}--End Round {round_number}--{Fore.RESET}\n")
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
        pareto_optimal_count = 0
        
        print(f"\n{Fore.CYAN}ROUND-BY-ROUND ANALYSIS:{Fore.RESET}")
        print("="*60)
        
        for round_obj in completed_rounds:
            print(f"\n{Fore.GREEN}Round {round_obj.round_number}:{Fore.RESET}")
            
            # Get final allocation from allocation tracker
            final_allocation = self.allocation_tracker.get_final_allocation(round_obj.round_number)
            
            if final_allocation:
                # Analyze this round's allocation
                analysis = ParetoAnalyzer.analyze_allocation_efficiency(round_obj.items, final_allocation)
                
                # Add to totals
                total_agent1_value += analysis['agent1_value']
                total_agent2_value += analysis['agent2_value']
                
                if analysis['is_pareto_optimal']:
                    pareto_optimal_count += 1
                
                # Display round analysis
                print(f"  Final Allocation:")
                print(f"    Agent 1: {final_allocation['agent1']} (Value: {analysis['agent1_value']:.2f})")
                print(f"    Agent 2: {final_allocation['agent2']} (Value: {analysis['agent2_value']:.2f})")
                print(f"  Total Welfare: {analysis['total_welfare']:.2f}")
                print(f"  Welfare Efficiency: {analysis['welfare_efficiency']:.1%}")
                
                if analysis['is_pareto_optimal']:
                    print(f"  {Fore.GREEN}✅ Pareto Optimal{Fore.RESET}")
                else:
                    print(f"  {Fore.RED}❌ Not Pareto Optimal{Fore.RESET}")
                    if analysis['potential_improvements']:
                        best_improvement = analysis['potential_improvements'][0]
                        print(f"  {Fore.YELLOW}💡 Best improvement: Total welfare {best_improvement['total_welfare']:.2f}{Fore.RESET}")
                
                print(f"  Available Pareto allocations: {analysis['pareto_optimal_count']}")
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
            print(f"Pareto Optimal Rounds: {pareto_optimal_count}/{len(completed_rounds)}")
            print(f"Pareto Optimality Rate: {pareto_optimal_count/len(completed_rounds):.1%}")
            
            # Determine winner
            if total_agent1_value > total_agent2_value:
                winner = "Agent 1"
                margin = total_agent1_value - total_agent2_value
                print(f"\n{Fore.GREEN}🏆 SESSION WINNER: {winner} (margin: +{margin:.2f}){Fore.RESET}")
            elif total_agent2_value > total_agent1_value:
                winner = "Agent 2"
                margin = total_agent2_value - total_agent1_value
                print(f"\n{Fore.GREEN}🏆 SESSION WINNER: {winner} (margin: +{margin:.2f}){Fore.RESET}")
            else:
                print(f"\n{Fore.YELLOW}🤝 SESSION RESULT: TIE{Fore.RESET}")
        
        # Detailed analysis for each completed round
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"=== DETAILED PARETO ANALYSIS ===")
        print(f"{'='*60}{Fore.RESET}")
        
        for round_obj in completed_rounds:
            final_allocation = self.allocation_tracker.get_final_allocation(round_obj.round_number)
            if final_allocation:
                print(f"\n{Fore.CYAN}Round {round_obj.round_number} Detailed Analysis:{Fore.RESET}")
                analysis = ParetoAnalyzer.analyze_allocation_efficiency(round_obj.items, final_allocation)
                formatted_report = ParetoAnalyzer.format_analysis_report(analysis)
                
                # Color code the report lines
                for line in formatted_report.split('\n'):
                    if '✅' in line:
                        print(f"{Fore.GREEN}{line}{Fore.RESET}")
                    elif '❌' in line:
                        print(f"{Fore.RED}{line}{Fore.RESET}")
                    elif '📊' in line or '═' in line:
                        print(f"{Fore.CYAN}{line}{Fore.RESET}")
                    elif '📈' in line:
                        print(f"{Fore.YELLOW}{line}{Fore.RESET}")
                    else:
                        print(line)
        
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"=== END SESSION ANALYSIS ===")
        print(f"{'='*60}{Fore.RESET}")
        
        self.total_scores = {"agent1": total_agent1_value, "agent2": total_agent2_value}

# Main execution
async def main():
    """
    Entry point for the negotiation system.
    """
    # Create and run a negotiation session
    session = NegotiationSession(num_rounds=3, items_per_round=6)
    await session.run_negotiation()

if __name__ == "__main__":
    asyncio.run(main())
