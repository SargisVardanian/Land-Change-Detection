from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_TOKEN_GROUP_TERMS = {
    "object": {"building", "buildings", "house", "houses", "road", "roads", "tree", "trees", "field", "fields", "water", "vegetation", "plant", "plants"},
    "direction": {"appeared", "built", "constructed", "added", "disappeared", "removed", "demolished", "increased", "expanded", "decreased", "reduced"},
    "location": {"top", "bottom", "upper", "lower", "left", "right", "center", "middle", "north", "south", "east", "west"},
    "count": {"one", "two", "three", "four", "five", "single", "several", "many", "few"},
    "relation": {"replace", "replaced", "replacing", "converted", "conversion", "place"},
}


def _clean_token(token: str) -> str:
    return token.casefold().lstrip("▁ġ#").strip(".,:;!?()[]{}")


def _content_token_mask(tokenizer, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Keep lexical content tokens; exclude tokenizer specials and stopwords."""
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for row, ids in enumerate(input_ids.detach().cpu().tolist()):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for column, (token_id, token) in enumerate(zip(ids, tokens, strict=True)):
            cleaned = _clean_token(str(token))
            mask[row, column] = (
                bool(attention_mask[row, column])
                and token_id not in special
                and bool(cleaned)
            )
        # A malformed/stopword-only caption still has a deterministic non-special fallback.
        if not bool(mask[row].any()):
            for column, token_id in enumerate(ids):
                if bool(attention_mask[row, column]) and token_id not in special:
                    mask[row, column] = True
                    break
    return mask


def _token_group_metadata(tokenizer, input_ids: Tensor, attention_mask: Tensor, texts: list[str]) -> dict:
    group_masks = {
        name: torch.zeros_like(attention_mask, dtype=torch.bool)
        for name in _TOKEN_GROUP_TERMS
    }
    for row, ids in enumerate(input_ids.detach().cpu().tolist()):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for column, token in enumerate(tokens):
            cleaned = _clean_token(str(token))
            for name, terms in _TOKEN_GROUP_TERMS.items():
                if cleaned in terms or (name == "count" and cleaned.isdigit()):
                    group_masks[name][row, column] = True
    location_targets = torch.full((len(texts), 2), 0.5, device=attention_mask.device)
    count_targets = torch.zeros(len(texts), device=attention_mask.device)
    direction_targets = torch.zeros(len(texts), device=attention_mask.device, dtype=torch.long)
    count_words = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "several": 3, "few": 3, "many": 5}
    for row, text in enumerate(texts):
        words = text.casefold().replace("-", " ").split()
        if any(word.strip(".,:;!?()[]{}") in {"appeared", "built", "constructed", "added", "increased", "expanded"} for word in words):
            direction_targets[row] = 1
        elif any(word.strip(".,:;!?()[]{}") in {"disappeared", "removed", "demolished", "decreased", "reduced"} for word in words):
            direction_targets[row] = -1
        if any(word in words for word in ("top", "upper", "north")):
            location_targets[row, 1] = 0.0
        elif any(word in words for word in ("bottom", "lower", "south")):
            location_targets[row, 1] = 1.0
        if any(word in words for word in ("left", "west")):
            location_targets[row, 0] = 0.0
        elif any(word in words for word in ("right", "east")):
            location_targets[row, 0] = 1.0
        for word in words:
            cleaned = word.strip(".,:;!?()[]{}")
            if cleaned.isdigit():
                count_targets[row] = float(cleaned)
                break
            if cleaned in count_words:
                count_targets[row] = float(count_words[cleaned])
                break
    return {
        "token_group_masks": group_masks,
        "location_targets": location_targets,
        "count_targets": count_targets,
        "direction_targets": direction_targets,
    }


TextRole = Literal["query", "document"]
GlobalProjectionMode = Literal["learned_projection", "matryoshka_truncate"]


@dataclass(frozen=True)
class TextFeatures:
    global_embedding: Tensor
    token_embeddings: Tensor
    attention_mask: Tensor
    content_token_mask: Tensor
    role: TextRole
    metadata: dict


@dataclass(frozen=True)
class JinaV5TextConfig:
    model_path: str | Path
    max_length: int = 64
    retrieval_dim: int = 512
    hidden_dim: int = 1024
    global_projection_mode: GlobalProjectionMode = "learned_projection"
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
        self.global_projection = nn.Linear(config.hidden_dim, config.retrieval_dim)
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

    def _project_global(self, pooled: Tensor) -> Tensor:
        if self.config.global_projection_mode == "learned_projection":
            return F.normalize(self.global_projection(pooled), dim=-1)
        if self.config.global_projection_mode == "matryoshka_truncate":
            return F.normalize(pooled[:, : self.config.retrieval_dim], dim=-1)
        raise ValueError(f"Unsupported global projection mode: {self.config.global_projection_mode}")

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
        pooled = _last_token_pool(hidden, encoded["attention_mask"])
        global_embedding = self._project_global(pooled)
        token_embeddings = F.normalize(self.local_projection(hidden), dim=-1)
        metadata = {
            "repo_id": "jinaai/jina-embeddings-v5-text-small-retrieval",
            "model_path": str(self.model_path),
            "max_length": self.config.max_length,
            "global_source_dim": hidden.shape[-1],
            "global_dim": global_embedding.shape[-1],
            "global_projection_mode": self.config.global_projection_mode,
            "token_dim": token_embeddings.shape[-1],
            "content_token_count": _content_token_mask(self.tokenizer, encoded["input_ids"], encoded["attention_mask"]).sum(dim=1).detach().cpu().tolist(),
            "frozen": self.config.freeze,
            **self.config.metadata,
            **_token_group_metadata(self.tokenizer, encoded["input_ids"], encoded["attention_mask"], texts),
        }
        return TextFeatures(
            global_embedding=global_embedding,
            token_embeddings=token_embeddings,
            attention_mask=encoded["attention_mask"],
            content_token_mask=_content_token_mask(self.tokenizer, encoded["input_ids"], encoded["attention_mask"]),
            role=role,
            metadata=metadata,
        )
