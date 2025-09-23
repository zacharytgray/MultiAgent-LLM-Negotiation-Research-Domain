# Multi-Agent LLM Negotiation Research Domain

A sophisticated research framework for studying negotiation dynamics between Large Language Model (LLM) agents in competitive multi-round scenarios. This system enables comprehensive analysis of AI-to-AI negotiations with detailed logging, Pareto optimality analysis, and configurable experimental parameters.

**🎯 NEW:** Multi-Agent Architecture with Boulware Strategy Support - Run negotiations between different agent types with configurable strategies!

## 🎯 Project Overview

This framework simulates competitive negotiation environments where two LLM agents must negotiate the allocation of valuable items across multiple rounds. Each agent only knows their own item valuations and must strategically negotiate to maximize their total utility without knowledge of their opponent's preferences.

The system now supports **multiple agent types** that can be mixed and matched to study different negotiation strategies and behaviors in combination.

## 🏗️ Architecture Overview

The system follows a modular, two-stage architecture designed for research scalability and analytical flexibility:

```
MultiAgent-LLM-Negotiation-Research-Domain/
├── src/                          # Source code organized by responsibility
│   ├── core/                     # Domain models and business logic
│   ├── agents/                   # LLM agent implementations
│   ├── analysis/                 # Research analysis tools
│   └── utils/                    # Utility functions and helpers
├── config/                       # Configuration management
├── logs/                         # Raw negotiation data (CSV)
├── results/                      # Analyzed results with metrics
├── Negotiation.py               # Main experiment runner
└── analyze_results.py           # Post-negotiation analysis script
```

## 🔄 Two-Stage Design

### Stage 1: Data Collection (`Negotiation.py`)
- **Fast execution**: No heavy analysis during negotiations
- **Raw data storage**: Complete conversation history, proposals, and allocations
- **Real-time feedback**: Basic round summaries and progress tracking
- **Output**: Raw CSV files in `logs/` directory

### Stage 2: Analysis (`analyze_results.py`)
- **Comprehensive metrics**:
  - **Pareto Optimality**: Identifies allocations where no agent can be made better off without making another worse off.
  - **Welfare**: Measures the total combined utility achieved by all agents, calculated as the sum of each agent’s value for their allocated items.
  - **Welfare Efficiency**: Expresses efficiency as the ratio of achieved welfare to the maximum possible welfare, where the maximum is determined by the best Pareto optimal allocation.
  - **Negotiation Dynamics**: Analyzes patterns such as proposal frequency, agreement timing, and strategy shifts throughout the negotiation.
- **Flexible analysis**: Add new metrics without re-running negotiations
- **Reproducible results**: Same raw data can be analyzed multiple ways
- **Output**: Analyzed CSV files in `results/` directory

## 🧩 Component Documentation

### Core Domain Models (`src/core/`)

#### `Item.py`
- **Purpose**: Represents negotiable items with agent-specific valuations
- **Key Features**: 
  - Dual valuation system (agent1Value, agent2Value)
  - Value range: 0.0 (lowest) to 1.0 (highest priority)
  - Encapsulates item metadata and utility calculations

#### `Round.py`
- **Purpose**: Manages individual negotiation rounds
- **Key Features**:
  - Tracks round state and completion status
  - Stores conversation history and final allocations
  - Manages agent turn sequences and round metadata

### Agent System (`src/agents/`)

The system now features a **modular multi-agent architecture** that supports different negotiation strategies and behaviors:

#### Agent Types Available

##### **Default Agent** (`default_agent.py`)
- **Purpose**: Standard LLM negotiator using conversation-based strategy
- **Behavior**: Relies on model's natural negotiation capabilities
- **Best for**: Baseline comparisons and natural negotiation dynamics

##### **Boulware Agent** (`boulware_agent.py`) 🆕
- **Purpose**: Implements the classic Boulware negotiation strategy with deterministic proposals
- **Strategy**: Starts with high demands (80% threshold) and gradually decreases over turns
- **Behavior**: 
  - Calculates welfare-ranked allocations for strategic decision making
  - Uses LLM as a natural language wrapper around deterministic proposals
  - Validates that LLM output matches intended algorithmic strategy
- **Configurable Parameters**:
  - `BOULWARE_INITIAL_THRESHOLD`: Starting demand level (default: 0.80)
  - `BOULWARE_DECREASE_RATE`: Threshold reduction per turn (default: 0.05)  
  - `BOULWARE_MIN_THRESHOLD`: Minimum acceptable threshold (default: 0.1)
- **Best for**: Studying competitive vs. cooperative dynamics

##### **More Agent Types Coming Soon** 🚀
- Fixed Price Agent (rigid pricing strategies)
- Charming Agent (socially persuasive tactics)
- Rude/Bullying Agent (aggressive negotiation styles)

#### Agent Architecture

##### **Base Agent** (`base_agent.py`)
- **Purpose**: Abstract interface defining all agent behaviors
- **Features**: Deterministic agent support, validation methods, memory management
- **Extensibility**: Easy framework for adding new agent types

##### **Agent Factory** (`agent_factory.py`)
- **Purpose**: Factory pattern for creating and configuring different agent types
- **Features**: 
  - Type registration system for easy extension
  - Configuration helpers for each agent type
  - Support for custom hyperparameters

#### Core LLM Integration

##### `ollamaAgentModule.py`
- **Purpose**: LLM agent implementation using Ollama framework
- **Key Features**:
  - Memory management for conversation context
  - Integration with LangChain for prompt engineering
  - Configurable model parameters and behavior
  - Tool integration capabilities for enhanced reasoning

##### `ollamaTools.py`
- **Purpose**: Dynamic tool collection for agent capabilities
- **Key Features**:
  - Extensible tool framework for agent enhancement
  - Integration with LangChain's tool ecosystem

### Analysis Engine (`src/analysis/`)

#### `ParetoAnalyzer.py`
- **Purpose**: Advanced game-theoretic analysis of negotiation outcomes
- **Key Features**:
  - Pareto optimality detection and measurement
  - Welfare efficiency calculations
  - Alternative allocation generation and ranking
  - Comprehensive efficiency metrics and reporting

### Utility System (`src/utils/`)

#### `CSVLogger.py`
- **Purpose**: Raw data logging for post-processing analysis
- **Key Features**:
  - Stores raw negotiation data
  - JSON fields for complex data structures
  - Complete conversation and proposal history preservation
  - Unique timestamped filenames (`MODEL_ITEMS_YYYYMMDD_HHMM.csv`)

#### `MessageParser.py`
- **Purpose**: Natural language processing for negotiation communication
- **Key Features**:
  - Formal proposal extraction using structured JSON format
  - Agreement detection and validation
  - Error handling for malformed proposals
  - Support for `PROPOSAL {"agent1": [...], "agent2": [...]}` syntax

#### `AllocationTracker.py`
- **Purpose**: State management for negotiation progress
- **Key Features**:
  - Proposal history tracking across conversation turns
  - Agreement validation and finalization logic
  - Round completion detection
  - Final allocation determination and conflict resolution

### Configuration Management (`config/`)

#### `settings.py`
- **Purpose**: Centralized configuration for all system parameters
- **Key Features**:
  - Model and agent configuration constants
  - Negotiation parameters (max turns, item constraints)
  - Analysis thresholds and display settings
  - File directory and naming conventions
  - **Agent-Specific Settings**: 🆕
    - `BOULWARE_INITIAL_THRESHOLD`: Starting threshold for Boulware agents
    - `BOULWARE_DECREASE_RATE`: Rate of threshold decrease per turn
    - `BOULWARE_MIN_THRESHOLD`: Minimum threshold value

#### `system_instructions.txt`
- **Purpose**: Agent behavior guidelines and negotiation rules
- **Key Features**:
  - Strategic guidance for competitive negotiations
  - Formal proposal syntax requirements
  - Communication protocols and format specifications

## 🚀 Getting Started

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
# Default: http://localhost:11434
```

### Multi-Agent Configuration 🆕

The system now supports different agent types and combinations:

#### Creating Different Agent Types
```python
from src.agents.agent_factory import AgentFactory, AgentConfig

# Create a default agent (standard LLM behavior)
default_agent = AgentFactory.create_agent(
    agent_type="default",
    agent_id=1,
    model_name="gpt-oss:20b",
    system_instructions_file="config/system_instructions.txt"
)

# Create a Boulware agent with custom parameters
boulware_config = AgentConfig.boulware_config(
    initial_threshold=0.90,  # Start more aggressively 
    decrease_rate=0.03,      # Decrease more slowly
    min_threshold=0.15       # Don't go too low
)

boulware_agent = AgentFactory.create_agent(
    agent_type="boulware",
    agent_id=2,
    model_name="gpt-oss:20b", 
    system_instructions_file="config/system_instructions.txt",
    **boulware_config
)
```

#### Tuning Boulware Agent Behavior
Edit `config/settings.py` to change default Boulware parameters:
```python
# Aggressive strategy: high initial demands, fast concessions
BOULWARE_INITIAL_THRESHOLD = 0.95
BOULWARE_DECREASE_RATE = 0.10
BOULWARE_MIN_THRESHOLD = 0.20

# Conservative strategy: moderate demands, slow concessions  
BOULWARE_INITIAL_THRESHOLD = 0.70
BOULWARE_DECREASE_RATE = 0.02
BOULWARE_MIN_THRESHOLD = 0.05
```

### Two-Stage Workflow

#### Stage 1: Run Negotiations
```bash
python Negotiation.py
```
**Output**: Raw data saved to `logs/MODEL_ITEMS_YYYYMMDD_HHMM.csv`

#### Stage 2: Analyze Results
```bash
python analyze_results.py logs/gpt-oss_20b_4_20250921_1425.csv
```
**Output**: Analyzed data saved to `results/gpt-oss_20b_4_20250921_1425_analyzed.csv`

### Configuration Options
All system parameters can be modified in `config/settings.py`:
- **Model Settings**: Default model, temperature, timeout
- **Negotiation Parameters**: Max turns, item value ranges, round counts
- **Analysis Settings**: Pareto thresholds, efficiency metrics
- **Logging Configuration**: Output directories, filename formats
- **Agent-Specific Settings**: 🆕
  - **Boulware Agent**: Initial threshold, decrease rate, minimum threshold
  - **Future Agents**: Fixed Price, Charming, Rude/Bullying (coming soon)

## 📊 Research Data Output

### Raw Data (logs/)
Each negotiation session generates a CSV with:
- **Session Metadata**: Model, timestamp, configuration
- **Agent Information**: 🆕 Agent types for both participants
- **Round Data**: Items, allocations, timing
- **Conversation History**: Complete agent interactions
- **Proposal History**: All formal proposals with validation
- **Outcome Data**: Final allocations and completion status

### Analyzed Results (results/)
Post-processing analysis adds:
- **Pareto Optimality**: Efficiency classification and measurements
- **Welfare Analysis**: Total welfare, individual agent efficiency
- **Negotiation Dynamics**: Proposal patterns, agreement timing
- **Strategy Metrics**: Turn counts, proposal validation rates
- **Agent Interaction Analysis**: 🆕 Performance comparison between different agent types

## 🤝 Negotiation Protocol

### Formal Proposal System
Agents use structured JSON format for proposals:
```
Agent: "I think this allocation works well for both of us.

PROPOSAL {
  "agent1": ["ItemA", "ItemC"],
  "agent2": ["ItemB", "ItemD"]  
}

This gives us both some high-value items. What do you think?"
```

### Agreement Protocol
- Agents type "AGREE" to accept current proposal
- Both agents must agree consecutively to finalize round
- System validates all proposals for completeness and correctness

## Adding New Analysis Metrics

1. **Add calculation functions** to `analyze_results.py`
2. **Update analysis pipeline** in the `analyze_round()` method  
3. **Re-run analysis** on existing CSV files (no re-negotiation needed)
4. **Raw data contains everything**: Items, allocations, conversations, proposals

## 📄 Example Usage

```bash
# Run a standard negotiation (default vs default agents)
python Negotiation.py  

# Run mixed agent negotiations by modifying Negotiation.py:
# - Change agent1_type = "default" and agent2_type = "boulware" 
# - Or create custom agent configurations in code

# Analyze the results with all metrics
python analyze_results.py logs/log_file_path.csv

# Check results directory for analyzed output
ls results/
```

## 🧪 Research Applications

This framework enables studying:
- **Strategy Comparison**: Default vs Boulware vs future agent types
- **Behavioral Analysis**: How different strategies affect negotiation outcomes
- **Efficiency Studies**: Which agent combinations achieve better Pareto outcomes
- **Competitive Dynamics**: Aggressive vs cooperative strategy interactions
- **Parameter Sensitivity**: How Boulware threshold tuning affects performance

## 🚀 Extending the Framework

### Adding New Agent Types
1. **Create agent class** inheriting from `BaseAgent`
2. **Implement required methods**: `should_make_deterministic_proposal()`, `get_deterministic_proposal()`, etc.
3. **Register with factory**: `AgentFactory.register_agent_type("your_type", YourAgentClass)`
4. **Add configuration helpers**: Update `AgentConfig` class
5. **Document hyperparameters**: Add settings to `config/settings.py`

The framework is designed for easy extension - more agent types coming soon!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built for advancing research in AI negotiation dynamics and multi-agent system behavior.*
