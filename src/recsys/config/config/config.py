from dataclasses import dataclass
from typing import Literal
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import JNCFCfg


@dataclass
class Config:
    model: JNCFCfg
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["jncf"]
    dataset: str
    seed: int