from dataclasses import dataclass


@dataclass
class JNCFCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim_rl: list
    hidden_dim_ml: list
    dropout: float