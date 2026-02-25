from dataclasses import dataclass


@dataclass
class JNCFCfg:
    num_users: int
    num_items: int
    params: dict

