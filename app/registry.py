"""Model configs live in models/*.yaml; this module loads them and hands out one backend per config."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from app.backends import BACKENDS, DecisionBackend

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelConfig(BaseModel):
    id: str
    aliases: list[str] = []
    backend: str = "vllm"
    served_model_name: str
    base_url: str = "http://localhost:8000/v1"
    base_url_env: str | None = None  # env var that overrides base_url
    thinking_mode: str | None = None  # label used in metadata and calibration keys
    chat_template_kwargs: dict = {}
    # When set, the model reasons first: generate until this text's first token, append the
    # text, then read the label distribution at the next position.
    think_end: str | None = None
    thinking_budget: int = 4096
    supports_logprobs: bool = True
    confidence: str = "normalized_entropy"
    timeout_s: float = 600

    @property
    def url(self) -> str:
        return os.environ.get(self.base_url_env or "", self.base_url).rstrip("/")


MODELS = {
    cfg.id: cfg
    for cfg in (ModelConfig(**yaml.safe_load(p.read_text())) for p in sorted(MODELS_DIR.glob("*.yaml")))
}
_backends: dict[str, DecisionBackend] = {}


def get_model(name: str) -> ModelConfig:
    for cfg in MODELS.values():
        if name == cfg.id or name in cfg.aliases:
            return cfg
    raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")


def backend_for(cfg: ModelConfig) -> DecisionBackend:
    if cfg.id not in _backends:
        _backends[cfg.id] = BACKENDS[cfg.backend](cfg)
    return _backends[cfg.id]
