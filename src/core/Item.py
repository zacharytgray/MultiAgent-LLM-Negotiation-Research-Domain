from dataclasses import dataclass

@dataclass
class Item:
    """
    Represents an item in the negotiation with values for each agent.
    Values are between 0.0 (least appealing) and 1.0 (highest priority).
    """
    name: str
    agent1Value: float
    agent2Value: float
    
    def __post_init__(self):
        # Validate values are in the correct range
        if not (0.0 <= self.agent1Value <= 1.0):
            raise ValueError(f"agent1Value must be between 0.0 and 1.0, got {self.agent1Value}")
        if not (0.0 <= self.agent2Value <= 1.0):
            raise ValueError(f"agent2Value must be between 0.0 and 1.0, got {self.agent2Value}")