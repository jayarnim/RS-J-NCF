from ..config.model import (
    JNCFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="jncf":
        return jncf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def jncf(cfg):
    return JNCFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim_rl=cfg["model"]["hidden_dim_rl"],
        hidden_dim_ml=cfg["model"]["hidden_dim_ml"],
        dropout=cfg["model"]["dropout"],
    )