from ..config.model import JNCFCfg


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="jncf":
        return jncf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def jncf(cfg):
    return JNCFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )
