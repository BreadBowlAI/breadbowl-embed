"""Shared Qwen backbone plus slot extractor and routing/value heads."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class QKVOutput:
    routing: torch.Tensor
    value: torch.Tensor
    # [batch, slots, tokens] for one head; [batch, heads, slots, tokens] otherwise.
    attention: torch.Tensor
    slot_states: torch.Tensor | None = None
    slot_queries: torch.Tensor | None = None


def resolve_hidden_size(config: Any) -> int:
    """Resolve backbone width across slightly different HF config layouts."""
    direct_candidates = ("hidden_size", "d_model", "dim", "model_dim")
    for name in direct_candidates:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value

    nested_candidates = ("text_config", "language_config", "llm_config")
    for nested_name in nested_candidates:
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        for name in direct_candidates:
            value = getattr(nested, name, None)
            if isinstance(value, int) and value > 0:
                return value

    raise AttributeError(
        "Unable to resolve hidden size from backbone config. "
        "Expected one of hidden_size/d_model/dim/model_dim on the config or nested text config."
    )


def resolve_torch_dtype(config: dict[str, Any]) -> torch.dtype | None:
    raw = config.get("torch_dtype")
    if raw in (None, "auto"):
        if torch.backends.mps.is_available():
            return torch.float32
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported torch_dtype config value: {raw}")
    return mapping[raw]


def resolve_pretrained_source(config: dict[str, Any]) -> str:
    """Resolve a local path or cached snapshot for the configured backbone."""
    pretrained = os.path.expandvars(str(config["pretrained"]))
    if "$" in pretrained:
        raise RuntimeError(
            f"Backbone pretrained path contains an unresolved environment variable: {pretrained!r}."
        )
    revision = config.get("revision")
    if not config.get("local_files_only", False):
        return pretrained

    pretrained_path = Path(pretrained).expanduser()
    if pretrained_path.exists():
        return str(pretrained_path)

    try:
        snapshot_kwargs: dict[str, Any] = {"repo_id": pretrained, "local_files_only": True}
        if revision:
            snapshot_kwargs["revision"] = revision
        return snapshot_download(**snapshot_kwargs)
    except Exception as exc:  # pragma: no cover - depends on local HF cache state
        raise RuntimeError(
            f"local_files_only is enabled, but no cached snapshot was found for {pretrained!r}."
        ) from exc


def resolve_jepa_sentinel_token(train_config: dict[str, Any] | None) -> str | None:
    """Return the configured JEPA sentinel token when masked-doc training uses one."""
    if not train_config:
        return None
    stage = str(train_config.get("stage", "weak"))
    if stage not in {"jepa", "candidate_distill"}:
        return None
    jepa_cfg = train_config.get("jepa", {})
    strategy = str(jepa_cfg.get("mask_token_strategy", "sentinel"))
    if strategy != "sentinel":
        return None
    token = str(jepa_cfg.get("mask_token_text", "<jepa_mask>")).strip()
    if not token:
        raise ValueError("JEPA sentinel masking requires a non-empty `jepa.mask_token_text`.")
    return token


def _iter_cache_configs(config: Any):
    seen: set[int] = set()
    stack = [config]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        for nested_name in ("text_config", "language_config", "llm_config"):
            stack.append(getattr(item, nested_name, None))


def disable_backbone_cache(backbone: nn.Module) -> None:
    config = getattr(backbone, "config", None)
    for item in _iter_cache_configs(config):
        if hasattr(item, "use_cache"):
            item.use_cache = False


def backbone_cache_state(backbone: nn.Module) -> bool | str:
    config = getattr(backbone, "config", None)
    states: list[bool] = []
    for item in _iter_cache_configs(config):
        if hasattr(item, "use_cache"):
            states.append(bool(item.use_cache))
    if not states:
        return "unknown"
    return any(states)


def _copy_token_embedding(
    embedding: nn.Embedding | None,
    *,
    source_token_id: int,
    target_token_id: int,
) -> None:
    if embedding is None:
        return
    weight = embedding.weight.data
    if source_token_id >= weight.size(0) or target_token_id >= weight.size(0):
        raise IndexError("Token embedding copy referenced an out-of-range token id.")
    weight[target_token_id].copy_(weight[source_token_id])


def _is_vocab_expansion_weight(key: str) -> bool:
    return key.endswith(".weight") and any(
        marker in key for marker in ("embed", "lm_head", "shared", "wte")
    )


def _adapt_vocab_weight(
    current_tensor: torch.Tensor,
    loaded_tensor: torch.Tensor,
    *,
    preserved_rows: set[int],
) -> torch.Tensor:
    expanded = current_tensor.detach().clone()
    row_overlap = min(current_tensor.size(0), loaded_tensor.size(0))
    loaded_tensor = loaded_tensor.to(dtype=expanded.dtype)
    if not preserved_rows:
        expanded[:row_overlap].copy_(loaded_tensor[:row_overlap])
        return expanded

    copied_start = 0
    for row_idx in sorted(index for index in preserved_rows if 0 <= index < row_overlap):
        if copied_start < row_idx:
            expanded[copied_start:row_idx].copy_(loaded_tensor[copied_start:row_idx])
        copied_start = row_idx + 1
    if copied_start < row_overlap:
        expanded[copied_start:row_overlap].copy_(loaded_tensor[copied_start:row_overlap])
    return expanded


def _adapt_state_dict_for_vocab_expansion(
    module: nn.Module,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    current_state = module.state_dict()
    preserved_rows: set[int] = set()
    jepa_mask_token_id = getattr(module, "jepa_mask_token_id", None)
    if isinstance(jepa_mask_token_id, int) and jepa_mask_token_id >= 0:
        preserved_rows.add(jepa_mask_token_id)
    adapted_state: dict[str, Any] = {}
    incompatible_shapes: list[str] = []
    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if current_value is None or not isinstance(value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
            adapted_state[key] = value
            continue
        if value.shape == current_value.shape:
            adapted_state[key] = value
            continue
        if (
            _is_vocab_expansion_weight(key)
            and value.ndim == current_value.ndim == 2
            and value.shape[1] == current_value.shape[1]
            and value.shape[0] != current_value.shape[0]
        ):
            adapted_state[key] = _adapt_vocab_weight(
                current_value,
                value,
                preserved_rows=preserved_rows,
            )
            continue
        incompatible_shapes.append(
            f"{key}: checkpoint {tuple(value.shape)} vs current {tuple(current_value.shape)}"
        )
    return adapted_state, incompatible_shapes


def load_qkv_state_dict(module: nn.Module, state_dict: dict[str, Any]) -> None:
    """Load model weights while allowing optional JEPA-specific additions."""
    # Old checkpoints have one extractor. Initialize the optional independent
    # value extractor from its restored weights, not from random construction.
    # A partially present new extractor is corruption, never an old checkpoint.
    if getattr(module, "value_slot_extractor", None) is not None and not any(
        key.startswith("value_slot_extractor.") for key in state_dict
    ):
        state_dict = dict(state_dict)
        for key in module.value_slot_extractor.state_dict():
            source = "slot_extractor." + key
            if source in state_dict:
                state_dict["value_slot_extractor." + key] = state_dict[source]
    adapted_state, incompatible_shapes = _adapt_state_dict_for_vocab_expansion(module, state_dict)
    if incompatible_shapes:
        raise RuntimeError(
            "Failed to adapt QKV state dict due to incompatible tensor shapes: "
            + "; ".join(incompatible_shapes)
        )
    incompatible = module.load_state_dict(adapted_state, strict=False)
    raw_missing_keys = list(getattr(incompatible, "missing_keys", []))
    checkpoint_has_value_adapter = any(key.startswith("value_adapter.") for key in state_dict)
    missing_keys = [
        key
        for key in raw_missing_keys
        if key != "jepa_mask_embedding_delta"
        and (checkpoint_has_value_adapter or not key.startswith("value_adapter."))
    ]
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing keys: {missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected keys: {unexpected_keys}")
        raise RuntimeError("Failed to load QKV state dict with compatible keys: " + "; ".join(details))


def configure_tokenizer_and_model(
    model: "QKVRetriever",
    tokenizer: Any,
    train_config: dict[str, Any] | None = None,
) -> Any:
    """Apply tokenizer special-token setup and resize model embeddings to match."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentinel_token = resolve_jepa_sentinel_token(train_config)
    if sentinel_token is None:
        return tokenizer

    reference_token_id = getattr(tokenizer, "mask_token_id", None)
    if reference_token_id is None:
        reference_token_id = getattr(tokenizer, "eos_token_id", None)
    if reference_token_id is None:
        reference_token_id = getattr(tokenizer, "pad_token_id", None)

    vocab_size_before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [sentinel_token]})
    if len(tokenizer) != vocab_size_before:
        model.backbone.resize_token_embeddings(len(tokenizer))

    sentinel_token_id = tokenizer.convert_tokens_to_ids(sentinel_token)
    if sentinel_token_id is None:
        raise ValueError(
            f"Failed to register JEPA sentinel token {sentinel_token!r} in the tokenizer."
        )
    if reference_token_id is not None and sentinel_token_id != reference_token_id:
        _copy_token_embedding(
            model.backbone.get_input_embeddings(),
            source_token_id=int(reference_token_id),
            target_token_id=int(sentinel_token_id),
        )
        output_embeddings = model.backbone.get_output_embeddings()
        if isinstance(output_embeddings, nn.Embedding):
            _copy_token_embedding(
                output_embeddings,
                source_token_id=int(reference_token_id),
                target_token_id=int(sentinel_token_id),
            )
    model.enable_trainable_jepa_mask_token(sentinel_token_id)
    return tokenizer


class SlotExtractor(nn.Module):
    """Learned slot queries with cross-attention; d_attn is the total head width."""

    def __init__(
        self,
        hidden_size: int,
        num_slots: int,
        d_attn: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        slot_query_init_std: float = 0.02,
        slot_query_init_mode: Literal["normal", "orthogonal"] = "normal",
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("slot_extractor.num_heads must be a positive integer.")
        if d_attn <= 0 or d_attn % num_heads != 0:
            raise ValueError("slot_extractor.d_attn must be positive and divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.d_attn = d_attn
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.slot_queries = nn.Parameter(
            self._init_slot_queries(
                num_slots=num_slots,
                hidden_size=hidden_size,
                init_std=slot_query_init_std,
                init_mode=slot_query_init_mode,
            )
        )
        self.q_proj = nn.Linear(hidden_size, d_attn)
        self.k_proj = nn.Linear(hidden_size, d_attn)
        self.v_proj = nn.Linear(hidden_size, d_attn)
        self.out_proj = nn.Linear(d_attn, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.head_dim)

    @staticmethod
    def _init_slot_queries(
        *,
        num_slots: int,
        hidden_size: int,
        init_std: float,
        init_mode: Literal["normal", "orthogonal"],
    ) -> torch.Tensor:
        if init_mode == "normal":
            return torch.randn(num_slots, hidden_size) * init_std
        slot_queries = torch.empty(num_slots, hidden_size)
        # Match the rough row-norm scale of N(0, init_std) while making rows
        # start out decorrelated to break slot symmetry early.
        nn.init.orthogonal_(slot_queries, gain=init_std * math.sqrt(hidden_size))
        return slot_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        slot_queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.q_proj(slot_queries)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.num_heads > 1:
            q = q.reshape(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(batch_size, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(batch_size, hidden_states.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).to(dtype=torch.bool)
            if self.num_heads > 1:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        slot_states = torch.matmul(attn, v)
        if self.num_heads > 1:
            slot_states = slot_states.transpose(1, 2).reshape(batch_size, self.num_slots, self.d_attn)
        slot_states = self.out_proj(slot_states)
        slot_states = self.norm(slot_states)
        return slot_states, attn


class ResidualValueAdapter(nn.Module):
    """Zero-initialized residual MLP that expands value-only capacity safely."""

    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int,
        adapter_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if adapter_size <= 0:
            raise ValueError("Value adapter size must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Value adapter dropout must be in [0, 1).")
        self.down = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_size, output_size)
        # Preserve the exact step-1000 value function at initialization. This
        # makes linear-versus-MLP comparisons attributable to learned capacity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, slot_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(slot_states))))


class QKVRetriever(nn.Module):
    """Shared backbone that emits routing and value slots for any input text."""

    def __init__(
        self,
        backbone: nn.Module,
        num_slots: int,
        d_attn: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        slot_query_init_std: float = 0.02,
        slot_query_init_mode: Literal["normal", "orthogonal"] = "normal",
        num_heads: int = 1,
        value_adapter_hidden_dim: int | None = None,
        value_adapter_dropout: float = 0.0,
        independent_value_extractor: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = resolve_hidden_size(backbone.config)
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.d_k = d_k or hidden_size
        self.d_v = d_v or hidden_size

        self.slot_extractor = SlotExtractor(
            hidden_size=hidden_size,
            num_slots=num_slots,
            d_attn=d_attn,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            slot_query_init_std=slot_query_init_std,
            slot_query_init_mode=slot_query_init_mode,
            num_heads=num_heads,
        )
        self.routing_head = nn.Linear(hidden_size, self.d_k)
        self.value_slot_extractor = deepcopy(self.slot_extractor) if independent_value_extractor else None
        self.value_head = nn.Linear(hidden_size, self.d_v)
        self.value_adapter: nn.Module | None = None
        if value_adapter_hidden_dim is not None and value_adapter_hidden_dim > 0:
            self.value_adapter = ResidualValueAdapter(
                hidden_size=hidden_size,
                output_size=self.d_v,
                adapter_size=value_adapter_hidden_dim,
                dropout=value_adapter_dropout,
            )
        self.jepa_mask_token_id: int | None = None
        self.register_parameter("jepa_mask_embedding_delta", None)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        disable_backbone_cache(self.backbone)
        if not hasattr(self.backbone, "gradient_checkpointing_enable"):
            raise RuntimeError("Configured gradient_checkpointing=true, but the backbone does not support it.")
        try:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.backbone.gradient_checkpointing_enable()

        is_checkpointing = getattr(self.backbone, "is_gradient_checkpointing", None)
        if callable(is_checkpointing):
            is_checkpointing = is_checkpointing()
        if is_checkpointing is False:
            raise RuntimeError("Backbone gradient checkpointing was requested but did not enable.")

    def enable_trainable_jepa_mask_token(self, token_id: int) -> None:
        """Register a learned offset for the JEPA sentinel token embedding."""
        self.jepa_mask_token_id = int(token_id)
        if self.jepa_mask_embedding_delta is not None:
            return
        input_embeddings = self.backbone.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("Backbone does not expose input embeddings for JEPA sentinel training.")
        self.jepa_mask_embedding_delta = nn.Parameter(
            torch.zeros(
                self.hidden_size,
                dtype=input_embeddings.weight.dtype,
                device=input_embeddings.weight.device,
            )
        )

    def _build_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.jepa_mask_embedding_delta is None or self.jepa_mask_token_id is None:
            return None
        jepa_mask_positions = input_ids.eq(self.jepa_mask_token_id)
        if not torch.any(jepa_mask_positions):
            return None
        input_embeddings = self.backbone.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("Backbone does not expose input embeddings for JEPA sentinel training.")
        inputs_embeds = input_embeddings(input_ids)
        sentinel_delta = self.jepa_mask_embedding_delta.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds + jepa_mask_positions.unsqueeze(-1).to(inputs_embeds.dtype) * sentinel_delta
        return inputs_embeds

    def compute_value(self, slot_states: torch.Tensor) -> torch.Tensor:
        """Project frozen or trainable slot states through the value path."""
        value = self.value_head(slot_states)
        if self.value_adapter is not None:
            value = value + self.value_adapter(slot_states)
        return value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> QKVOutput:
        inputs_embeds = self._build_inputs_embeds(input_ids)
        backbone_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "return_dict": True,
            "use_cache": False,
        }
        if inputs_embeds is None:
            backbone_kwargs["input_ids"] = input_ids
        else:
            backbone_kwargs["inputs_embeds"] = inputs_embeds
        outputs = self.backbone(**backbone_kwargs)
        hidden_states = outputs.last_hidden_state.to(self.routing_head.weight.dtype)
        slot_states, attn = self.slot_extractor(hidden_states, attention_mask)
        routing = self.routing_head(slot_states)
        value_states = slot_states
        if self.value_slot_extractor is not None:
            value_states, _ = self.value_slot_extractor(hidden_states, attention_mask)
        value = self.compute_value(value_states)
        return QKVOutput(
            routing=routing,
            value=value,
            attention=attn,
            slot_states=slot_states,
            slot_queries=self.slot_extractor.slot_queries,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        train_config: dict[str, Any] | None = None,
        backbone_init: Literal["pretrained", "config"] = "pretrained",
        backbone_source_override: str | None = None,
        tokenizer_source_override: str | None = None,
        local_files_only_override: bool | None = None,
        enable_gradient_checkpointing: bool | None = None,
    ) -> tuple["QKVRetriever", Any]:
        backbone_cfg = config["backbone"]
        pretrained_source = backbone_source_override or resolve_pretrained_source(backbone_cfg)
        tokenizer_source = tokenizer_source_override or pretrained_source
        model_kwargs = {}
        if backbone_cfg.get("attn_implementation"):
            model_kwargs["attn_implementation"] = backbone_cfg["attn_implementation"]
        torch_dtype = resolve_torch_dtype(backbone_cfg)
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        local_files_only = (
            backbone_cfg.get("local_files_only", False)
            if local_files_only_override is None
            else local_files_only_override
        )
        revision = backbone_cfg.get("revision") if backbone_source_override is None else None
        if backbone_init == "pretrained":
            if revision:
                model_kwargs["revision"] = revision
            backbone = AutoModel.from_pretrained(
                pretrained_source,
                trust_remote_code=backbone_cfg.get("trust_remote_code", False),
                local_files_only=local_files_only,
                **model_kwargs,
            )
        elif backbone_init == "config":
            config_kwargs: dict[str, Any] = {
                "trust_remote_code": backbone_cfg.get("trust_remote_code", False),
                "local_files_only": local_files_only,
            }
            if revision:
                config_kwargs["revision"] = revision
            backbone_config = AutoConfig.from_pretrained(pretrained_source, **config_kwargs)
            backbone = AutoModel.from_config(
                backbone_config,
                trust_remote_code=backbone_cfg.get("trust_remote_code", False),
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backbone initialization mode: {backbone_init}")
        tokenizer_revision = backbone_cfg.get("revision") if tokenizer_source_override is None else None
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=backbone_cfg.get("trust_remote_code", False),
            local_files_only=local_files_only,
            revision=tokenizer_revision,
        )
        model = cls(
            backbone=backbone,
            num_slots=config["slot_extractor"]["num_slots"],
            d_attn=config["slot_extractor"]["d_attn"],
            d_k=config["heads"].get("d_k"),
            d_v=config["heads"].get("d_v"),
            dropout=config["slot_extractor"].get("dropout", 0.0),
            use_layer_norm=config["slot_extractor"].get("use_layer_norm", True),
            slot_query_init_std=config["slot_extractor"].get("slot_query_init_std", 0.02),
            slot_query_init_mode=config["slot_extractor"].get("slot_query_init_mode", "normal"),
            num_heads=config["slot_extractor"].get("num_heads", 1),
            value_adapter_hidden_dim=config["heads"].get("value_adapter_hidden_dim"),
            value_adapter_dropout=float(config["heads"].get("value_adapter_dropout", 0.0)),
            independent_value_extractor=bool(config["heads"].get("independent_value_extractor", False)),
        )
        tokenizer = configure_tokenizer_and_model(model, tokenizer, train_config)
        disable_backbone_cache(backbone)
        should_enable_gradient_checkpointing = (
            backbone_cfg.get("gradient_checkpointing", False)
            if enable_gradient_checkpointing is None
            else enable_gradient_checkpointing
        )
        if should_enable_gradient_checkpointing:
            gradient_checkpointing_kwargs = backbone_cfg.get("gradient_checkpointing_kwargs")
            if gradient_checkpointing_kwargs is not None and not isinstance(
                gradient_checkpointing_kwargs, dict
            ):
                raise ValueError("backbone.gradient_checkpointing_kwargs must be a mapping when set.")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        return model, tokenizer
