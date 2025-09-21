# Multi-Agent LLM Negotiation Research Domain

A sophisticated research framework for studying negotiation dynamics between Large Language Model (LLM) agents in competitive multi-round scenarios. This system enables comprehensive analysis of AI-to-AI negotiations with detailed logging, Pareto optimality analysis, and configurable experimental parameters.

## 🎯 Project Overview

This framework simulates competitive negotiation environments where two LLM agents must negotiate the allocation of valuable items across multiple rounds. Each agent only knows their own item valuations and must strategically negotiate to maximize their total utility without knowledge of their opponent's preferences.

### Key Research Questions Addressed:
- How do LLM agents develop negotiation strategies?
- What factors influence Pareto-optimal outcomes in AI negotiations?
- How does negotiation behavior evolve across multiple rounds?
- What are the efficiency characteristics of AI-mediated negotiations?

## 🏗️ Architecture Overview

The system follows a modular, professional package structure designed for research scalability and maintainability:

```
MultiAgent-LLM-Negotiation-Research-Domain/
├── src/                          # Source code organized by responsibility
│   ├── core/                     # Domain models and business logic
│   ├── agents/                   # LLM agent implementations
│   ├── analysis/                 # Research analysis tools
│   └── utils/                    # Utility functions and helpers
├── config/                       # Configuration management
├── logs/                         # Generated experiment data
└── Negotiation.py               # Main experiment runner
```

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
- **Purpose**: Comprehensive data logging for research analysis
- **Key Features**:
  - Structured CSV output with 40+ metrics per round
  - Unique timestamped filenames (`MODEL_ITEMS_YYYYMMDD_HHMM.csv`)
  - Integration with Pareto analysis for outcome classification
  - Configurable logging parameters and output directories

#### `MessageParser.py`
- **Purpose**: Natural language processing for negotiation communication
- **Key Features**:
  - Proposal extraction from free-form text
  - Agreement detection and validation
  - Structured parsing of allocation statements
  - Error handling for malformed proposals

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
  - CSV logging and file naming conventions

#### `system_instructions.txt`
- **Purpose**: Agent behavior guidelines and negotiation rules
- **Key Features**:
  - Strategic guidance for LLM agents
  - Negotiation protocol definitions
  - Communication format specifications

## 🚀 Getting Started

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
# Default: http://localhost:11434
```

### Basic Usage
```python
# Run a complete negotiation session
python Negotiation.py

# Or customize parameters
from Negotiation import NegotiationSession

session = NegotiationSession(
    num_rounds=5,           # Number of negotiation rounds
    items_per_round=4,      # Items to negotiate per round
    model_name="gemma3:12b" # LLM model to use
)
await session.run_negotiation()
```

### Configuration Options
All system parameters can be modified in `config/settings.py`:
- **Model Settings**: Default model, temperature, timeout
- **Negotiation Parameters**: Max turns, item value ranges, round counts
- **Analysis Settings**: Pareto thresholds, efficiency metrics
- **Logging Configuration**: Output directories, filename formats

## 📊 Research Data Output

### CSV Logging
Each experiment generates a comprehensive CSV file with metrics including:
- **Session Metadata**: Model, timestamp, configuration parameters
- **Round Performance**: Duration, turn count, completion status
- **Agent Behavior**: Proposal patterns, agreement timing
- **Outcome Analysis**: Final allocations, utility scores
- **Efficiency Metrics**: Pareto optimality, welfare measurements

### Real-time Analysis
The system provides live console output with:
- Color-coded agent conversations
- Proposal validation and feedback
- Pareto optimality detection
- Round completion summaries
- Session-wide performance statistics

## 🎓 Research Applications

This framework is designed for:
- **Behavioral AI Research**: Understanding LLM negotiation strategies
- **Game Theory Studies**: Empirical analysis of multi-agent interactions
- **Efficiency Analysis**: Measuring Pareto optimality in AI negotiations
- **Strategic Learning**: Investigating adaptation across multiple rounds
- **Comparative Studies**: Evaluating different models and parameters

## 🤝 Example Negotiation Flow

```
Round 1 Starting
================
Items for Round 1:
  ItemA: Agent1=0.9, Agent2=0.3
  ItemB: Agent1=0.2, Agent2=0.8
  ItemC: Agent1=0.7, Agent2=0.5
  ItemD: Agent1=0.4, Agent2=0.9
Starting Agent: Agent 1

Agent 1's turn (Turn 1):
Agent 1: I propose taking ItemA and ItemC, which leaves ItemB and ItemD for you.

Agent 2's turn (Turn 2):
Agent 2: That works well for me since ItemB and ItemD are valuable to me. AGREE.

Agent 1's turn (Turn 3):
Agent 1: Excellent! AGREE.

Both agents have agreed! Round 1 complete.
Final allocation: {'agent1': ['ItemA', 'ItemC'], 'agent2': ['ItemB', 'ItemD']}
✅ Pareto Optimal allocation achieved!
📊 Round 1 logged to CSV (Duration: 12.3s, Turns: 3)
```

## 📈 Future Enhancements

- **Multi-Model Comparisons**: Framework for testing different LLM architectures
- **Advanced Analysis**: Machine learning-based strategy classification
- **Interactive Visualization**: Real-time negotiation analysis dashboards
- **Extended Protocols**: Support for more complex negotiation scenarios
- **Benchmarking Suite**: Standardized evaluation metrics and test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built for advancing research in AI negotiation dynamics and multi-agent system behavior.*