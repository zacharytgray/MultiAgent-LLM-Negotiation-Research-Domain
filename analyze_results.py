#!/usr/bin/env python3
"""
Analysis Script for Multi-Agent Negotiation Results

This script loads raw negotiation CSV files and calculates all metrics including:
- Pareto optimality analysis
- Welfare efficiency
- Negotiation dynamics
- Future metrics can be added here

Usage:
    python analyze_results.py <csv_filepath>
    python analyze_results.py logs/gpt-oss_20b_4_20250921_1425.csv
"""

import argparse
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from src.analysis.ParetoAnalyzer import ParetoAnalyzer
from src.core.Item import Item
from config.settings import DEFAULT_RESULTS_DIR


class NegotiationAnalyzer:
    """
    Analyzes raw negotiation data and calculates comprehensive metrics.
    """
    
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.df = pd.read_csv(csv_filepath)
        self.results = []
        
        # Create results directory
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        
        # Generate output filename based on input
        base_filename = os.path.splitext(os.path.basename(csv_filepath))[0]
        self.output_filename = f"{base_filename}_analyzed.csv"
        self.output_filepath = os.path.join(DEFAULT_RESULTS_DIR, self.output_filename)
        
    def parse_items_data(self, items_json: str) -> List[Item]:
        """Parse JSON items data back into Item objects."""
        try:
            items_data = json.loads(items_json)
            return [
                Item(
                    name=item['name'],
                    agent1Value=item['agent1_value'],
                    agent2Value=item['agent2_value']
                )
                for item in items_data
            ]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing items data: {e}")
            return []
    
    def parse_allocation(self, allocation_json: str) -> Dict[str, List[str]]:
        """Parse JSON allocation data back into allocation dictionary."""
        try:
            return json.loads(allocation_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing allocation data: {e}")
            return {"agent1": [], "agent2": []}
    
    def parse_conversation_history(self, conversation_json: str) -> List[tuple]:
        """Parse JSON conversation history."""
        try:
            return json.loads(conversation_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing conversation history: {e}")
            return []
    
    def parse_proposal_history(self, proposal_json: str) -> List[Dict]:
        """Parse JSON proposal history."""
        try:
            return json.loads(proposal_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing proposal history: {e}")
            return []
    
    def calculate_basic_metrics(self, items: List[Item], allocation: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate basic allocation metrics."""
        # Calculate individual agent values
        agent1_value = sum(
            item.agent1Value for item in items 
            if item.name in allocation.get('agent1', [])
        )
        agent2_value = sum(
            item.agent2Value for item in items 
            if item.name in allocation.get('agent2', [])
        )
        
        # Calculate maximum possible values
        agent1_max_possible = sum(item.agent1Value for item in items)
        agent2_max_possible = sum(item.agent2Value for item in items)
        
        return {
            'agent1_final_value': agent1_value,
            'agent2_final_value': agent2_value,
            'total_welfare': agent1_value + agent2_value,
            'agent1_max_possible': agent1_max_possible,
            'agent2_max_possible': agent2_max_possible,
            'total_max_possible': agent1_max_possible + agent2_max_possible,
            'agent1_efficiency': agent1_value / agent1_max_possible if agent1_max_possible > 0 else 0,
            'agent2_efficiency': agent2_value / agent2_max_possible if agent2_max_possible > 0 else 0
        }
    
    def calculate_negotiation_dynamics(self, conversation_history: List[tuple], 
                                     proposal_history: List[Dict]) -> Dict[str, Any]:
        """Calculate negotiation dynamics metrics."""
        total_proposals = len(proposal_history)
        valid_proposals = sum(1 for p in proposal_history if p.get('proposal', {}).get('is_valid', False))
        invalid_proposals = total_proposals - valid_proposals
        
        agent1_proposals = sum(1 for p in proposal_history if p.get('agent_num') == 1)
        agent2_proposals = sum(1 for p in proposal_history if p.get('agent_num') == 2)
        
        # Find final proposer
        final_proposer = None
        for proposal in reversed(proposal_history):
            if proposal.get('proposal', {}).get('is_valid', False):
                final_proposer = proposal.get('agent_num')
                break
        
        return {
            'total_proposals': total_proposals,
            'valid_proposals': valid_proposals,
            'invalid_proposals': invalid_proposals,
            'agent1_proposals': agent1_proposals,
            'agent2_proposals': agent2_proposals,
            'final_proposer': final_proposer,
            'conversation_turns': len(conversation_history)
        }
    
    def analyze_round(self, row: pd.Series) -> Dict[str, Any]:
        """Analyze a single round and return all metrics."""
        # Parse raw data
        items = self.parse_items_data(row['items_data'])
        allocation = self.parse_allocation(row['final_allocation'])
        conversation_history = self.parse_conversation_history(row['conversation_history'])
        proposal_history = self.parse_proposal_history(row['proposal_history'])
        
        # Start with original round data
        round_metrics = {
            'session_id': row['session_id'],
            'model_name': row['model_name'],
            'round_number': row['round_number'],
            'total_rounds': row['total_rounds'],
            'round_duration_seconds': row['round_duration_seconds'],
            'starting_agent': row['starting_agent'],
            'round_completed': row['round_completed'],
            'agreement_reached': row['agreement_reached'],
            'timestamp': row['timestamp'],
            'date': row['date']
        }
        
        # Calculate basic metrics
        basic_metrics = self.calculate_basic_metrics(items, allocation)
        round_metrics.update(basic_metrics)
        
        # Calculate Pareto analysis
        if items and allocation:
            pareto_analysis = ParetoAnalyzer.analyze_allocation_efficiency(items, allocation)
            round_metrics.update({
                'is_pareto_optimal': pareto_analysis['is_pareto_optimal'],
                'welfare_efficiency': pareto_analysis['welfare_efficiency'],
                'max_possible_welfare': pareto_analysis['max_possible_welfare'],
                'welfare_gap': pareto_analysis['max_possible_welfare'] - pareto_analysis['total_welfare'],
                'pareto_optimal_count': pareto_analysis['pareto_optimal_count'],
                'unique_pareto_combinations': pareto_analysis['unique_pareto_optimal_count']
            })
        else:
            # Default values if analysis fails
            round_metrics.update({
                'is_pareto_optimal': False,
                'welfare_efficiency': 0.0,
                'max_possible_welfare': 0.0,
                'welfare_gap': 0.0,
                'pareto_optimal_count': 0,
                'unique_pareto_combinations': 0
            })
        
        # Calculate negotiation dynamics
        dynamics = self.calculate_negotiation_dynamics(conversation_history, proposal_history)
        round_metrics.update(dynamics)
        
        # Add item allocation details (comma-separated for CSV compatibility)
        round_metrics.update({
            'agent1_items': ','.join(allocation.get('agent1', [])),
            'agent2_items': ','.join(allocation.get('agent2', [])),
            'agent1_values': ','.join([str(item.agent1Value) for item in items]),
            'agent2_values': ','.join([str(item.agent2Value) for item in items])
        })
        
        return round_metrics
    
    def analyze_all_rounds(self) -> pd.DataFrame:
        """Analyze all rounds in the dataset."""
        print(f"Analyzing {len(self.df)} rounds from {self.csv_filepath}")
        
        for idx, row in self.df.iterrows():
            print(f"Analyzing round {row['round_number']}...")
            try:
                round_analysis = self.analyze_round(row)
                self.results.append(round_analysis)
            except Exception as e:
                print(f"Error analyzing round {row['round_number']}: {e}")
                # Add a basic entry to maintain row count
                self.results.append({
                    'round_number': row['round_number'],
                    'error': str(e),
                    **{col: row[col] for col in ['session_id', 'model_name', 'timestamp']}
                })
        
        return pd.DataFrame(self.results)
    
    def save_results(self, results_df: pd.DataFrame):
        """Save analyzed results to CSV."""
        results_df.to_csv(self.output_filepath, index=False)
        print(f"✅ Analysis complete! Results saved to: {self.output_filepath}")
    
    def display_summary(self, results_df: pd.DataFrame):
        """Display summary statistics."""
        total_rounds = len(results_df)
        completed_rounds = results_df['round_completed'].sum() if 'round_completed' in results_df.columns else 0
        
        if 'is_pareto_optimal' in results_df.columns:
            pareto_optimal_rounds = results_df['is_pareto_optimal'].sum()
            avg_welfare_efficiency = results_df['welfare_efficiency'].mean()
        else:
            pareto_optimal_rounds = 0
            avg_welfare_efficiency = 0
            
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total Rounds: {total_rounds}")
        print(f"Completed Rounds: {completed_rounds}")
        print(f"Pareto Optimal Rounds: {pareto_optimal_rounds}")
        print(f"Average Welfare Efficiency: {avg_welfare_efficiency:.3f}")
        print(f"Output File: {self.output_filename}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze negotiation results from CSV file')
    parser.add_argument('csv_file', help='Path to the raw negotiation CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: File {args.csv_file} does not exist")
        return 1
    
    # Create analyzer and run analysis
    analyzer = NegotiationAnalyzer(args.csv_file)
    
    try:
        results_df = analyzer.analyze_all_rounds()
        analyzer.save_results(results_df)
        analyzer.display_summary(results_df)
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())