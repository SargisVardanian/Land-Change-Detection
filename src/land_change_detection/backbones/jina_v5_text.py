from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TextRole = Literal["query", "document"]


@dataclass(frozen=True)
class TextFeatures:
    global_embedding: Tensor
    token_embeddings: Tensor
    attention_mask: Tensor
    role: TextRole
    metadata: dict


@dataclass(frozen=True)
class JinaV5TextConfig:
    model_path: str | Path
    max_length: int = 64
    retrieval_dim: int = 512
    hidden_dim: int = 1024
    freeze: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = True
    metadata: dict = field(default_factory=dict)


def role_prefix(role: TextRole) -> str:
    if role == "query":
        return "Query: "
    if role == "document":
        return "Document: "
    raise ValueError(f"Unsupported text role: {role}")


def _last_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden states must have shape [B, L, C].")
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError("attention_mask must have shape [B, L].")
    last_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, last_indices]


class JinaV5TextEncoder(nn.Module):
    """Offline Jina retrieval encoder with learned local-token projection."""

    def __init__(self, config: JinaV5TextConfig):
        super().__init__()
        self.config = config
        self.model_path = Path(config.model_path)
        self.local_projection = nn.Linear(config.hidden_dim, config.retrieval_dim)
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for JinaV5TextEncoder.") from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing local Jina v5 model path: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        if self.config.freeze:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @property
    def model_parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def dtype(self) -> str:
        if self.model is None:
            return "unknown"
        try:
            return str(next(self.model.parameters()).dtype)
        except StopIteration:
            return "unknown"

    def _format(self, texts: list[str], role: TextRole) -> list[str]:
        prefix = role_prefix(role)
        return [text if text.startswith(prefix) else f"{prefix}{text}" for text in texts]

    def forward(self, texts: list[str], role: TextRole = "query") -> TextFeatures:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Jina v5 tokenizer/model were not loaded.")
        encoded = self.tokenizer(
            self._format(texts, role),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(not self.config.freeze):
            output = self.model(**encoded, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state.to(self.local_projection.weight.dtype)
        if hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"Expected Jina hidden dim {self.config.hidden_dim}, got {hidden.shape[-1]}.")
        pooled = F.normalize(_last_token_pool(hidden, encoded["attention_mask"]), dim=-1)
        global_embedding = F.normalize(pooled[:, : self.config.retrieval_dim], dim=-1)
        token_embeddings = F.normalize(self.local_projection(hidden), dim=-1)
        metadata = {
            "repo_id": "jinaai/jina-embeddings-v5-text-small-retrieval",
            "model_path": str(self.model_path),
            "max_length": self.config.max_length,
            "global_source_dim": hidden.shape[-1],
            "global_dim": global_embedding.shape[-1],
            "token_dim": token_embeddings.shape[-1],
            "frozen": self.config.freeze,
            **self.config.metadata,
        }
        return TextFeatures(
            global_embedding=global_embedding,
            token_embeddings=token_embeddings,
            attention_mask=encoded["attention_mask"],
            role=role,
            metadata=metadata,
        )
