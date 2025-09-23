# Multi-Agent Negotiation System - Agent Architecture Refactor

## Overview

The codebase has been successfully refactored to support multiple agent types while maintaining backward compatibility with the existing default agent behavior. The new architecture enables easy addition of deterministic negotiation strategies like the Boulware agent.

## Architecture Changes

### 1. Base Agent Architecture

- **BaseAgent**: Abstract base class defining the interface for all agents
- **DefaultAgent**: Wrapper around the existing `ollamaAgentModule` for backward compatibility
- **BoulwareAgent**: Deterministic agent implementing the Boulware negotiation strategy
- **AgentFactory**: Factory pattern for creating different agent types
- **AgentConfig**: Configuration helpers for different agent types

### 2. Agent Types Implemented

#### Default Agent
- Pure LLM-based negotiation
- No deterministic strategy
- Original system behavior

#### Boulware Agent
- **Strategy**: Starts with high demands and gradually concedes
- **Threshold**: Begins at 80% (configurable) and decreases each turn
- **Deterministic**: Makes calculated proposals based on welfare rankings
- **LLM Wrapper**: Uses LLM to present proposals naturally
- **Validation**: Ensures LLM output matches intended deterministic proposal

### 3. Key Features

#### Deterministic Agent Support
- `should_make_deterministic_proposal()`: Identifies agents that need deterministic behavior
- `get_deterministic_proposal()`: Gets the calculated proposal based on strategy
- `should_accept_proposal()`: Evaluates whether to accept opponent's proposal
- `validate_output_matches_intent()`: Ensures LLM outputs the intended proposal
- `update_strategy_state()`: Updates strategy parameters based on turn number

#### Output Validation
- Deterministic agents validate that LLM output matches intended proposal
- Retry logic (up to 5 attempts) if validation fails
- Special system instructions for deterministic agents

#### Allocation Calculation
- Generates all possible item allocations
- Ranks allocations by agent welfare
- Supports threshold-based proposal selection
- Welfare-based acceptance decisions

## Files Added/Modified

### New Files
- `src/agents/base_agent.py` - Base agent interface
- `src/agents/default_agent.py` - Default agent implementation
- `src/agents/boulware_agent.py` - Boulware strategy implementation
- `src/agents/agent_factory.py` - Agent factory and configuration
- `config/deterministic_agent_instructions.txt` - System instructions for deterministic agents
- `demo_boulware.py` - Demo script for testing Boulware agent
- `test_agents.py` - Unit tests for agent system

### Modified Files
- `Negotiation.py` - Updated to support multiple agent types
- `src/utils/MessageParser.py` - Added allocation property to ParsedProposal

## Usage Examples

### Basic Usage
```python
# Create a session with different agent types
session = NegotiationSession(
    num_rounds=3,
    agent1_type="default",
    agent2_type="boulware",
    agent2_config={"initial_threshold": 0.80}
)
await session.run_negotiation()
```

### Available Agent Types
- `"default"` - Pure LLM agent
- `"boulware"` - Boulware strategy agent

### Running Specific Matchups
```python
# Test specific agent combinations
await run_specific_matchup(
    agent1_type="default",
    agent2_type="boulware",
    num_rounds=3
)
```

## Boulware Agent Details

### Strategy Implementation
1. **Initialization**: Calculates all possible allocations and ranks by welfare
2. **Opening Proposal**: Makes best possible allocation for itself
3. **Counter Proposals**: Uses threshold-based selection from ranked allocations
4. **Threshold Decay**: Decreases threshold by 10% each turn (minimum 0.1)
5. **Acceptance Logic**: Accepts proposals that meet current threshold welfare

### Validation Process
1. Agent calculates deterministic proposal
2. LLM receives instruction to present specific allocation
3. LLM generates natural language response with proposal
4. System validates that extracted proposal matches intended allocation
5. Retries up to 5 times if validation fails

## Testing

### Unit Tests (`test_agents.py`)
- Agent factory functionality
- Agent context generation
- Allocation calculations
- Boulware strategy logic
- Welfare calculations

### Integration Demo (`demo_boulware.py`)
- Full negotiation sessions
- Agent type combinations
- Real LLM interaction
- Output validation

## Results

The demo shows successful operation:
- **Agent 1 (Default)**: 4.70 total value
- **Agent 2 (Boulware)**: 4.50 total value
- **3 rounds completed** with natural conversation flow
- **Deterministic proposals validated** correctly
- **Acceptance logic working** as expected

## Next Steps

The architecture is now ready for additional agent types:

### Planned Agent Types
- **Fixed Price Agent**: Always proposes same allocation
- **Charming Agent**: Uses persuasive language patterns
- **Rude/Bullying Agent**: Aggressive negotiation style

### Easy Extension
```python
# Adding new agent type
class NewAgent(BaseAgent):
    def should_make_deterministic_proposal(self):
        return True  # or False for pure LLM
    
    def get_deterministic_proposal(self, current_proposal):
        # Implement strategy logic
        return calculated_allocation

# Register with factory
AgentFactory.register_agent_type("new_type", NewAgent)
```

The system now provides a robust foundation for multi-agent negotiation research with both deterministic and LLM-based strategies.