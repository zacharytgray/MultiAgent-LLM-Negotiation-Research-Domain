import asyncio
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ollamaAgentModule import Agent
from colorama import Fore, init
from Item import Item
from Round import Round

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
        self.agent1 = Agent("gpt-oss:20b", "systemInstructions.txt")
        self.agent2 = Agent("gpt-oss:20b", "systemInstructions.txt")
        
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
--New Round Start--
You are Agent 1 in a negotiation. Your goal is to maximize your own value.
Items: {round_obj.get_items_for_agent(1)}

You can only see your own values (agent1Value). The other agent's values are unknown to you.
Negotiate with Agent 2 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
"""
        
        agent2_context = f"""
--New Round Start--
You are Agent 2 in a negotiation. Your goal is to maximize your own value.
Items: {round_obj.get_items_for_agent(2)}

You can only see your own values (agent2Value). The other agent's values are unknown to you.
Negotiate with Agent 1 to get the best items for yourself.
When you reach an agreement, end your message with "AGREE".
"""
        
        # Add context to agents
        self.agent1.addToMemory('system', agent1_context)
        self.agent2.addToMemory('system', agent2_context)
        
        # Start the negotiation
        max_turns = 20  # Prevent infinite loops
        turn_count = 0
        last_messages = ["", ""]
        
        current_agent = starting_agent
        
        while turn_count < max_turns and not round_obj.is_complete:
            turn_count += 1
            
            if current_agent == 1:
                print(f"{Fore.GREEN}Agent 1's turn (Turn {turn_count}):{Fore.RESET}")
                response = await self.agent1.generateResponse()
                print(f"{Fore.GREEN}Agent 1: {response}{Fore.RESET}\n")
                
                # Add agent 1's response to agent 2's memory
                self.agent2.addToMemory('user', f"Agent 1: {response}")
                last_messages[0] = response
                
                # Check for agreement
                if turn_count > 1 and round_obj.detect_agreement(last_messages[0], last_messages[1]):
                    print(f"{Fore.CYAN}Both agents have agreed! Round {round_number} complete.{Fore.RESET}")
                    round_obj.is_complete = True
                    break
                
                current_agent = 2
                
            else:
                print(f"{Fore.BLUE}Agent 2's turn (Turn {turn_count}):{Fore.RESET}")
                response = await self.agent2.generateResponse()
                print(f"{Fore.BLUE}Agent 2: {response}{Fore.RESET}\n")
                
                # Add agent 2's response to agent 1's memory
                self.agent1.addToMemory('user', f"Agent 2: {response}")
                last_messages[1] = response
                
                # Check for agreement
                if turn_count > 1 and round_obj.detect_agreement(last_messages[0], last_messages[1]):
                    print(f"{Fore.CYAN}Both agents have agreed! Round {round_number} complete.{Fore.RESET}")
                    round_obj.is_complete = True
                    break
                
                current_agent = 1
            
            round_obj.conversation_history.append((current_agent, response))
        
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
        Display the results of all rounds.
        """
        print(f"\n{Fore.YELLOW}=== FINAL RESULTS ==={Fore.RESET}")
        for round_obj in self.rounds:
            status = "Completed" if round_obj.is_complete else "Incomplete"
            print(f"Round {round_obj.round_number}: {status}")
        
        print(f"\nTotal Rounds: {len(self.rounds)}")
        print(f"Completed Rounds: {sum(1 for r in self.rounds if r.is_complete)}")

# Main execution
async def main():
    """
    Entry point for the negotiation system.
    """
    # Create and run a negotiation session
    session = NegotiationSession(num_rounds=3, items_per_round=4)
    await session.run_negotiation()

if __name__ == "__main__":
    asyncio.run(main())
