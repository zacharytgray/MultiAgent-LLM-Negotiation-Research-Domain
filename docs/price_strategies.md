# Price Domain Strategy Registry

This document lists the deterministic strategies available in the `price` domain for offline RL dataset generation.

These strategies are implemented in `src/agents/price_strategies.py` and are registered in `STRATEGY_REGISTRY`.

## Overview

The dataset generation mode randomly selects two strategies from this pool for each episode. This ensures the dataset contains a diverse set of negotiation behaviors, from cooperative to competitive.

## Strategies

### Boulware Agents (Spectrum)

Agents that follow a time-dependent concession curve $P(t) = Start + (Reservation - Start) \times (t/T)^\beta$.

1.  **boulware_very_conceding** (`beta=0.2`): Rapidly concedes early in the negotiation.
2.  **boulware_conceding** (`beta=0.5`): Concedes moderately early.
3.  **boulware_linear** (`beta=1.0`): Concedes linearly over time (equivalent to "Linear" strategy).
4.  **boulware_firm** (`beta=2.0`): Concedes slowly, holding value until late.
5.  **boulware_hard** (`beta=4.0`): Concedes very slowly, effectively a hardliner until the very end.

### Fixed Strategies

1.  **price_fixed_strict** (`margin=5.0`): Offers exactly (Reservation +/- 5). Very rigid.
2.  **price_fixed_loose** (`margin=25.0`): Offers exactly (Reservation +/- 25). More cooperative but static.

### Adaptive Strategies

1.  **tit_for_tat**: Mirrors the opponent's last concession magnitude. Starts with a standard opening if opponent hasn't conceded yet.
2.  **split_difference**: Proposes the midpoint between the opponent's last offer and the agent's own previous offer (or reservation).
3.  **time_dependent**: Does not follow a concession curve for *offers* but relaxes its *acceptance threshold* linearly over time. If forced to offer, it offers its current threshold.
4.  **hardliner**: Maintains a static hardline offer (Reservation +/- 40) until the final round, where it concedes to its reservation value.

### Training-Only (Oracle)

1.  **random_zopa**: Randomly selects a price within the true ZOPA (Zone of Possible Agreement). This strategy has access to oracle information (both buyer max and seller min) and is used to generate diverse valid data coverage.

## Usage

To generate a dataset using these strategies:

```bash
python Negotiation.py --domain price --dataset_mode --num_episodes 1000 --max_rounds 10 --seed 42 --dataset_out datasets/price_train.jsonl
```

The resulting JSONL file will contain:
- Agent metadata (strategy name, params)
- State (history, time, role)
- Action (OFFER/ACCEPT)
- Reward (Shaped for Decision Transformer)
- Return-to-Go (calculated post-episode)
