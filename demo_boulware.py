"""
Demo script for testing the Boulware agent against a default agent.
This script runs a focused test of the new agent architecture.
"""

import asyncio
from Negotiation import run_specific_matchup
from src.agents.agent_factory import AgentFactory
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)


async def demo_boulware_agent():
    """
    Demo the Boulware agent in action.
    """
    print(f"{Fore.MAGENTA}🎯 Boulware Agent Demo{Fore.RESET}")
    print(f"{Fore.CYAN}Testing Boulware strategy vs Default agent{Fore.RESET}")
    
    # Show available agent types
    print(f"\n{Fore.YELLOW}Available agent types: {AgentFactory.get_available_types()}{Fore.RESET}")
    
    # Test configuration: Default vs Boulware
    print(f"\n{Fore.GREEN}=== Running: Default Agent vs Boulware Agent ==={Fore.RESET}")
    print(f"- Agent 1: Default (pure LLM)")
    print(f"- Agent 2: Boulware (deterministic strategy, threshold=0.80)")
    print(f"- Rounds: 3")
    print(f"- Items per round: 4")
    
    # Configure Boulware agent
    boulware_config = {
        "initial_threshold": 0.80,
        "use_tools": False
    }
    
    # Run the matchup
    session = await run_specific_matchup(
        agent1_type="default",
        agent2_type="boulware", 
        num_rounds=3,
        agent1_config=None,
        agent2_config=boulware_config
    )
    
    print(f"\n{Fore.MAGENTA}=== Demo Complete ==={Fore.RESET}")
    print(f"Final scores:")
    print(f"- Agent 1 (Default): {session.total_scores['agent1']:.2f}")
    print(f"- Agent 2 (Boulware): {session.total_scores['agent2']:.2f}")


async def demo_agent_combinations():
    """
    Demo different agent combinations.
    """
    print(f"\n{Fore.MAGENTA}🔄 Agent Combinations Demo{Fore.RESET}")
    
    combinations = [
        ("default", "default", "Pure LLM vs Pure LLM"),
        ("default", "boulware", "Pure LLM vs Boulware Strategy"),
        ("boulware", "default", "Boulware Strategy vs Pure LLM"),
        ("boulware", "boulware", "Boulware vs Boulware")
    ]
    
    for agent1_type, agent2_type, description in combinations:
        print(f"\n{Fore.GREEN}=== {description} ==={Fore.RESET}")
        
        # Configure agents
        config1 = {"initial_threshold": 0.75} if agent1_type == "boulware" else None
        config2 = {"initial_threshold": 0.80} if agent2_type == "boulware" else None
        
        session = await run_specific_matchup(
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            num_rounds=1,  # Short demo rounds
            agent1_config=config1,
            agent2_config=config2
        )
        
        print(f"Results: Agent1={session.total_scores['agent1']:.2f}, Agent2={session.total_scores['agent2']:.2f}")


async def main():
    """
    Main demo entry point.
    """
    try:
        # First demo: focused Boulware test
        await demo_boulware_agent()
        
        # Optional: Extended demo with multiple combinations
        # Uncomment the line below to run more extensive testing
        # await demo_agent_combinations()
        
    except Exception as e:
        print(f"{Fore.RED}❌ Demo failed: {e}{Fore.RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())