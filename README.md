# Multi-Agent LLM Negotiation Research Domain

A sophisticated research framework for studying negotiation dynamics between Large Language Model (LLM) agents in competitive multi-round scenarios. This system enables comprehensive analysis of AI-to-AI negotiations with detailed logging, Pareto optimality analysis, and configurable experimental parameters.

## 🎯 Project Overview

This framework simulates competitive negotiation environments where two LLM agents must negotiate the allocation of valuable items across multiple rounds. Each agent only knows their own item valuations and must strategically negotiate to maximize their total utility without knowledge of their opponent's preferences.

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

#### `ollamaAgentModule.py`
- **Purpose**: LLM agent implementation using Ollama framework
- **Key Features**:
  - Memory management for conversation context
  - Integration with LangChain for prompt engineering
  - Configurable model parameters and behavior
  - Tool integration capabilities for enhanced reasoning

#### `ollamaTools.py`
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

## 📊 Research Data Output

### Raw Data (logs/)
Each negotiation session generates a CSV with:
- **Session Metadata**: Model, timestamp, configuration
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

## 🎓 Research Applications

This framework enables:
- **Behavioral AI Research**: Understanding LLM negotiation strategies
- **Game Theory Studies**: Empirical analysis of multi-agent interactions  
- **Efficiency Analysis**: Measuring Pareto optimality in AI negotiations
- **Longitudinal Studies**: Strategy evolution across multiple rounds
- **Comparative Analysis**: Different models, parameters, and conditions

## 🤝 Enhanced Negotiation Protocol

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
# Run a 5-round negotiation session
python Negotiation.py  # Uses config/settings.py defaults

# Analyze the results with all metrics
python analyze_results.py logs/log_file_path.csv

# Check results directory for analyzed output
ls results/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built for advancing research in AI negotiation dynamics and multi-agent system behavior.*