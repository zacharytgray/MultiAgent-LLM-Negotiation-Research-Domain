from typing import List, Dict, Optional, Tuple
from itertools import combinations
from Item import Item

class ParetoAnalyzer:
    """
    Analyzer for determining Pareto optimality of allocations in negotiation rounds.
    """
    
    @staticmethod
    def calculate_agent_value(items_dict: Dict[str, float], agent_items: List[str]) -> float:
        """
        Calculate total value for an agent given their allocated items.
        
        Args:
            items_dict: Dictionary mapping item names to values for this agent
            agent_items: List of item names allocated to this agent
            
        Returns:
            Total value for this agent
        """
        return sum(items_dict.get(item, 0) for item in agent_items)
    
    @staticmethod
    def is_pareto_optimal(items: List[Item], allocation: Dict[str, List[str]]) -> bool:
        """
        Check if an allocation is Pareto optimal.
        
        Args:
            items: List of Item objects with values for both agents
            allocation: Dictionary with 'agent1' and 'agent2' keys containing item lists
            
        Returns:
            True if allocation is Pareto optimal, False otherwise
        """
        # Create value dictionaries for each agent
        agent1_values = {item.name: item.agent1Value for item in items}
        agent2_values = {item.name: item.agent2Value for item in items}
        
        # Current allocation values
        current_agent1_value = ParetoAnalyzer.calculate_agent_value(
            agent1_values, allocation['agent1']
        )
        current_agent2_value = ParetoAnalyzer.calculate_agent_value(
            agent2_values, allocation['agent2']
        )
        
        # Try all possible alternative allocations
        all_items = [item.name for item in items]
        
        # Generate all possible ways to split items between agents
        for r in range(len(all_items) + 1):
            for agent1_items in combinations(all_items, r):
                agent2_items = [item for item in all_items if item not in agent1_items]
                
                alt_agent1_value = ParetoAnalyzer.calculate_agent_value(
                    agent1_values, list(agent1_items)
                )
                alt_agent2_value = ParetoAnalyzer.calculate_agent_value(
                    agent2_values, agent2_items
                )
                
                # Check if this alternative is strictly better for one agent
                # without being worse for the other (Pareto improvement)
                if ((alt_agent1_value > current_agent1_value and alt_agent2_value >= current_agent2_value) or
                    (alt_agent2_value > current_agent2_value and alt_agent1_value >= current_agent1_value)):
                    return False  # Found a Pareto improvement
        
        return True  # No Pareto improvements found
    
    @staticmethod
    def find_all_pareto_optimal_allocations(items: List[Item]) -> List[Dict[str, List[str]]]:
        """
        Find all Pareto optimal allocations for a given set of items.
        
        Args:
            items: List of Item objects with values for both agents
            
        Returns:
            List of dictionaries, each representing a Pareto optimal allocation
        """
        pareto_optimal = []
        all_items = [item.name for item in items]
        
        # Check all possible allocations
        for r in range(len(all_items) + 1):
            for agent1_items in combinations(all_items, r):
                agent2_items = [item for item in all_items if item not in agent1_items]
                allocation = {
                    'agent1': list(agent1_items),
                    'agent2': agent2_items
                }
                
                if ParetoAnalyzer.is_pareto_optimal(items, allocation):
                    pareto_optimal.append(allocation)
        
        return pareto_optimal
    
    @staticmethod
    def find_unique_pareto_optimal_allocations(items: List[Item]) -> List[Dict[str, List[str]]]:
        """
        Find all unique Pareto optimal allocations, avoiding double-counting of symmetric allocations.
        
        This method eliminates allocations that are equivalent when agents are swapped,
        useful for analyzing the fundamental outcome space rather than agent-specific assignments.
        
        Args:
            items: List of Item objects with values for both agents
            
        Returns:
            List of dictionaries representing unique Pareto optimal allocations
        """
        all_pareto = ParetoAnalyzer.find_all_pareto_optimal_allocations(items)
        unique_allocations = []
        seen_value_pairs = set()
        
        # Create value dictionaries
        agent1_values = {item.name: item.agent1Value for item in items}
        agent2_values = {item.name: item.agent2Value for item in items}
        
        for allocation in all_pareto:
            # Calculate values for this allocation
            agent1_value = ParetoAnalyzer.calculate_agent_value(agent1_values, allocation['agent1'])
            agent2_value = ParetoAnalyzer.calculate_agent_value(agent2_values, allocation['agent2'])
            
            # Create a canonical representation of the value pair
            # Sort the values to eliminate order dependency
            value_pair = tuple(sorted([agent1_value, agent2_value]))
            
            # Only include if we haven't seen this value combination before
            if value_pair not in seen_value_pairs:
                seen_value_pairs.add(value_pair)
                unique_allocations.append(allocation)
        
        return unique_allocations
    
    @staticmethod
    def analyze_allocation_efficiency(items: List[Item], allocation: Dict[str, List[str]]) -> Dict:
        """
        Comprehensive analysis of an allocation's efficiency.
        
        Args:
            items: List of Item objects with values for both agents
            allocation: Dictionary with 'agent1' and 'agent2' keys containing item lists
            
        Returns:
            Dictionary with analysis results
        """
        # Create value dictionaries for each agent
        agent1_values = {item.name: item.agent1Value for item in items}
        agent2_values = {item.name: item.agent2Value for item in items}
        
        # Calculate current values
        agent1_value = ParetoAnalyzer.calculate_agent_value(agent1_values, allocation['agent1'])
        agent2_value = ParetoAnalyzer.calculate_agent_value(agent2_values, allocation['agent2'])
        
        # Find all Pareto optimal allocations
        pareto_allocations = ParetoAnalyzer.find_all_pareto_optimal_allocations(items)
        unique_pareto_allocations = ParetoAnalyzer.find_unique_pareto_optimal_allocations(items)
        
        # Check if current allocation is Pareto optimal
        is_pareto_optimal = ParetoAnalyzer.is_pareto_optimal(items, allocation)
        
        # Calculate total welfare (sum of both agents' values)
        total_welfare = agent1_value + agent2_value
        
        # Find the maximum possible total welfare
        max_welfare = 0
        best_welfare_allocation = None
        for pareto_alloc in pareto_allocations:
            welfare = (ParetoAnalyzer.calculate_agent_value(agent1_values, pareto_alloc['agent1']) +
                      ParetoAnalyzer.calculate_agent_value(agent2_values, pareto_alloc['agent2']))
            if welfare > max_welfare:
                max_welfare = welfare
                best_welfare_allocation = pareto_alloc
        
        # Calculate welfare efficiency (actual welfare / maximum welfare)
        welfare_efficiency = total_welfare / max_welfare if max_welfare > 0 else 0
        
        # Find potential improvements if not Pareto optimal
        potential_improvements = []
        if not is_pareto_optimal:
            for pareto_alloc in pareto_allocations:
                pareto_agent1_value = ParetoAnalyzer.calculate_agent_value(agent1_values, pareto_alloc['agent1'])
                pareto_agent2_value = ParetoAnalyzer.calculate_agent_value(agent2_values, pareto_alloc['agent2'])
                
                # Check if this Pareto allocation is better than current
                if (pareto_agent1_value >= agent1_value and pareto_agent2_value >= agent2_value and
                    (pareto_agent1_value > agent1_value or pareto_agent2_value > agent2_value)):
                    potential_improvements.append({
                        'allocation': pareto_alloc,
                        'agent1_value': pareto_agent1_value,
                        'agent2_value': pareto_agent2_value,
                        'total_welfare': pareto_agent1_value + pareto_agent2_value
                    })
        
        return {
            'current_allocation': allocation,
            'agent1_value': agent1_value,
            'agent2_value': agent2_value,
            'total_welfare': total_welfare,
            'is_pareto_optimal': is_pareto_optimal,
            'welfare_efficiency': welfare_efficiency,
            'max_possible_welfare': max_welfare,
            'best_welfare_allocation': best_welfare_allocation,
            'pareto_optimal_count': len(pareto_allocations),
            'unique_pareto_optimal_count': len(unique_pareto_allocations),
            'potential_improvements': potential_improvements
        }
    
    @staticmethod
    def format_analysis_report(analysis: Dict) -> str:
        """
        Format an analysis result into a readable report.
        
        Args:
            analysis: Analysis dictionary from analyze_allocation_efficiency
            
        Returns:
            Formatted string report
        """
        report = []
        report.append(f"📊 ALLOCATION ANALYSIS")
        report.append(f"═══════════════════════")
        report.append(f"Current Allocation:")
        report.append(f"  Agent 1: {analysis['current_allocation']['agent1']} (Value: {analysis['agent1_value']:.2f})")
        report.append(f"  Agent 2: {analysis['current_allocation']['agent2']} (Value: {analysis['agent2_value']:.2f})")
        report.append(f"")
        report.append(f"Total Welfare: {analysis['total_welfare']:.2f}")
        report.append(f"Welfare Efficiency: {analysis['welfare_efficiency']:.1%}")
        report.append(f"")
        
        if analysis['is_pareto_optimal']:
            report.append(f"✅ This allocation is PARETO OPTIMAL")
            report.append(f"   (Cannot improve one agent without hurting the other)")
        else:
            report.append(f"❌ This allocation is NOT Pareto optimal")
            if analysis['potential_improvements']:
                report.append(f"   Potential improvements found:")
                for i, improvement in enumerate(analysis['potential_improvements'][:3]):  # Show top 3
                    report.append(f"   {i+1}. Agent1: {improvement['allocation']['agent1']} (Value: {improvement['agent1_value']:.2f})")
                    report.append(f"      Agent2: {improvement['allocation']['agent2']} (Value: {improvement['agent2_value']:.2f})")
                    report.append(f"      Total Welfare: {improvement['total_welfare']:.2f}")
        
        report.append(f"")
        report.append(f"📈 There are {analysis['pareto_optimal_count']} total Pareto optimal allocations")
        report.append(f"📈 ({analysis['unique_pareto_optimal_count']} unique value combinations)")
        report.append(f"📈 Maximum possible welfare: {analysis['max_possible_welfare']:.2f}")
        
        return "\n".join(report)
