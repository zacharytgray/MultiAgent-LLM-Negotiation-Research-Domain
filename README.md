# Multi-Agent LLM Negotiation Research Domain

A sophisticated research framework for studying negotiation dynamics between Large Language Model (LLM) agents. This system enables comprehensive analysis of AI-to-AI negotiations with detailed logging, Pareto optimality analysis, and configurable experimental parameters. It supports multiple domains (Multi-Item Allocation, Single Issue Price) and training pipelines (HyperLoRA, Dataset Generation).

## 🌍 Negotiation Domains

### 1. Multi-Item Allocation (`multi_item`)
The agents must split a set of items with different private values.
- **Goal**: Allocate a set of items between two agents.
- **Constraints**: Items are indivisible.
- **Utilities**: Sum of values of items received.
- **Agent Types**: `default`, `boulware`, `fixed_price`, `charming`, `rude`.

### 2. Single Issue Price (`price`)
A negotiation over a single scalar price.
- **Goal**: Agree on a numeric price.
- **Roles**: Buyer (wants low price) vs Seller (wants high price).
- **Constraints**: Buyer Max Willingness vs Seller Min Acceptable.
- **Agent Types**:
  - `basic_price`: Free-form LLM negotiator.
  - `price_strategy`: LLM wrapper around a deterministic strategy (e.g., Boulware).
  - `price_fixed`, `price_boulware`: Legacy deterministic agents.

---

## 🚀 Usage Guide

### 1. Interactive LLM Negotiation (Online Mode)
Run a live negotiation between two LLM agents.

**Multi-Item Domain:**  
```bash
python Negotiation.py --domain multi_item --rounds 5 --agent1 default --agent2 boulware --model llama3
```

**Price Domain:**  
In the price domain, agents can be "wrappers" that enforce a specific mathematical strategy (like Boulware) while letting the LLM generate the natural language.
```bash
# Agent 1 is a Conceding Boulware (Deterministic Strategy wrapped in LLM)
# Agent 2 is a Basic Agent (Free-form LLM)
python Negotiation.py --domain price --rounds 5 --agent1 boulware --agent2 basic --model llama3
```
*Note: In price domain, `boulware` automatically maps to a Conceding Boulware strategy wrapper.*

To use specific strategies (e.g., Tit-for-Tat):
```bash
python Negotiation.py --domain price --agent1 tit_for_tat --agent2 hardliner
```

### 2. Dataset Generation (Offline Mode)
Generate large-scale offline datasets for training without invoking any LLMs. This is strictly for the `price` domain.

- **Generates**: `datasets/price_domain.jsonl`
- **Format**: JSONL (Trajectory ID, State, Action, Rho, Meta)
- **Features**: 
  - Calculates `rho` (Price Orientation) for Janus training.
  - `rho = (FinalPrice - Min) / (Max - Min)`

```bash
python Negotiation.py --domain price --dataset_mode --num_episodes 10000 --max_turns 20 --dataset_out datasets/price_domain.jsonl
```

### 3. Janus Training Pipeline (HyperLoRA)
**"Janus"** is a specialized architecture designed to train a single LLM to master the entire spectrum of negotiation strategies. Instead of training separate models for "aggressive" vs "conceding" behavior, Janus conditions a frozen base model (Qwen2) using a continuous scalar control variable $\rho$ (rho).

#### A. Architecture
Janus uses **HyperLoRA** (Hypernetwork-Gated Low-Rank Adaptation). 
- A small MLP (HyperNetwork) takes the scalar $\rho$ as input.
- It outputs a gating vector that modulates the LoRA adapters injected into the base model's Linear layers.
- **$\rho$ Semantics:**
    - $\rho \in [0, 1]$: Represents the normalized outcome of a successful deal (0=Seller Min, 1=Buyer Max).
    - $\rho = -1.0$: Represents a "Failure" mode (Impasse).
- **Goal**: By changing $\rho$ at inference time, you can control whether the agent fights for every penny, concedes easily, or purposefully causes a deadlock.

#### B. Workflow

**Step 1: Generate Raw Data**
Generate thousands of episodes using deterministic strategies (Boulware, Linear, Tit-for-Tat, etc.).
```bash
python Negotiation.py --domain price --dataset_mode --num_episodes 10000 --max_turns 20 --dataset_out datasets/price_domain.jsonl
```

**Step 2: Prepare Training Tables**
Convert the raw JSONL logs into a structured Parquet dataset (`decision_steps`) optimized for training. This calculates normalized prices (-1..1) and history features.
```bash
python src/data_prep/prepare_data.py --jsonl_path datasets/price_domain.jsonl --output_dir datasets/processed_tables
```
*Outputs: `datasets/processed_tables/decision_steps.parquet`*

**Step 3: Train Janus**
Run the Supervised Fine-Tuning (SFT) loop. This freezes the base model and trains the HyperLoRA parameters to predict the next action (`OFFER <price>` or `ACCEPT`).
```bash
python src/training/train_janus_hyperlora.py \
  --decision_steps_path datasets/processed_tables/decision_steps.parquet \
  --output_dir checkpoints/janus_v1 \
  --model_name "Qwen/Qwen2-7B-Instruct" \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 2e-4 \
  --max_steps 5000 \
  --use_qlora
```

**Training Arguments:**
- `--include_failures`: (Default: true) Whether to train on failed trajectories (rho=-1.0).
- `--use_qlora`: Use 4-bit quantization (BitsAndBytes) to safe VRAM (Requires ~6GB VRAM for 7B model).
- `--rank` / `--alpha`: LoRA configuration (Default: r=16, alpha=32).
- `--k_history`: Number of history items to include in the context window.

### 4. Visualization
Visualize the behavior of deterministic strategies.

**Concession Curves:**
Plots the price offers over time for all registered strategies (Boulware, TFT, Hardliner, etc.).
```bash
python visualize_concessions.py
```
*Output: `concession_plots/`*

**Interaction Plots:**
Simulates specific matchups (e.g., Tit-for-Tat vs Boulware) to see how they react to each other.
```bash
python visualize_interactions.py
```
*Output: `interaction_plots/`*

---

## 🤖 Agent Architectures

### Price Strategy Wrapper (`price_strategy_agent.py`)
Used when you want an LLM to behave predictably.
1.  **Backend**: Calculates the exact mathematical offer (e.g., $500).
2.  **Prompting**: Instructs the LLM, "You must offer $500. Wrap this in natural language."
3.  **Validation**: Checks if the LLM's output actually contains "500". If not, it retries the prompt.

### Basic Price Agent (`basic_price_agent.py`)
A standard LLM agent given a role and goals (e.g., "Buy cheap"), but with no mathematical constraints.

---

## 📂 Project Structure

```
src/
├── agents/             # Agent implementations
│   ├── price_strategies.py       # Mathematical strategies (Boulware, TFT)
│   ├── price_strategy_agent.py   # LLM Wrapper for strategies
│   ├── basic_price_agent.py      # Free-form Price Agent
│   └── ...
├── analysis/           # Analysis tools
├── core/               # Domain logic (PriceState, Item)
├── logging/            # Dataset logging (JSONL)
└── training/           # Training code
    ├── hyper_lora.py         # Hypernetwork + LoRA implementation
    └── dataset_loader.py     # Data pipeline
```

## 🛠️ Requirements
```bash
pip install -r requirements.txt
```
Includes: `torch`, `transformers`, `peft`, `ollama`, `langchain`, `pandas`, `matplotlib`.
