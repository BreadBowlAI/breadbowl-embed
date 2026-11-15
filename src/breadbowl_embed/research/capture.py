"""Intermediate tensors of the encoder: a traced forward pass.

`trace(model, texts, kind)` runs exactly the forward pass of `encode_queries`/`encode_documents`
(same tokenizer call, same network call, same batching) with temporary forward hooks that copy
intermediate tensors out. The hooks are removed when the call returns and never change a
computation, so `trace(...).encoding()` equals the encoder's output bit for bit.

Every captured tensor is returned as a float32 CPU copy with padding removed. On a bfloat16 or
float16 model the copies are exact (every bfloat16/float16 value is a float32 value); the model
itself computed in its own dtype on its own device. Pre-softmax attention logits are recomputed
from the captured slot-query and key projections with the extractor's own operations, in the
model's dtype, so softmax(logits) reproduces the returned attention.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..embeddings import Embeddings
from ._base import ResearchObject
from .tokens import TokenView, build_views, resolve_instructions

# What `trace` can capture besides attention, routing and value (always present).
CAPTURE_OPTIONS = ("hidden", "logits", "projections", "slot_states", "value_parts")
DEFAULT_CAPTURE = ("hidden", "logits", "slot_states", "value_parts")


@dataclass(frozen=True, eq=False)
class Trace(ResearchObject):
    """Intermediate tensors of one encoded text (float32 CPU copies, padding removed).

    Shapes use T = tokens, S = slots, H = backbone width, A = slot-attention width (d_attn),
    h = slot-attention heads, a = A / h:

    attention             [h, S, T] slot-extraction attention (post-softmax); equals the encoders'
                          return_attention output.
    attention_logits      [h, S, T] pre-softmax scaled dot products (capture "logits").
    hidden_states         [T, H] backbone output tokens as the slot extractor reads them (capture
                          "hidden"); `layer_states` holds the outputs of requested decoder layers.
    slot_query_projection [h, S, a], key_projection / value_projection [h, T, a] (capture
                          "projections").
    slot_readout          [S, A] attention readout, heads concatenated, before the output
                          projection; slot_states_pre_norm [S, H] after it, before the layer norm;
                          slot_states [S, H] after the layer norm: the input of both heads
                          (capture "slot_states").
    routing_raw           [S, d_k] routing head output before L2 normalisation.
    routing               [S, d_k] L2-normalised routing, as the encoders return it.
    value_linear, value_adapter [S, d_v] the two summands of `value` (capture "value_parts").
    value                 [S, d_v] raw value vectors, as the encoders return them.
    slot_queries          [S, H] the learned slot queries (shared by all texts).
    compute_dtype is the model's compute dtype setting; network_dtype the dtype the slot extractor
    and heads ran in (a mixed release such as v1.2-preview runs its backbone in bfloat16).
    """
    json_type = "trace"

    tokens: TokenView
    representation_id: str
    compute_dtype: str
    network_dtype: str
    attention: torch.Tensor
    routing_raw: torch.Tensor
    routing: torch.Tensor
    value: torch.Tensor
    slot_queries: torch.Tensor
    attention_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    layer_states: dict = field(default_factory=dict)
    slot_query_projection: torch.Tensor | None = None
    key_projection: torch.Tensor | None = None
    value_projection: torch.Tensor | None = None
    slot_readout: torch.Tensor | None = None
    slot_states_pre_norm: torch.Tensor | None = None
    slot_states: torch.Tensor | None = None
    value_linear: torch.Tensor | None = None
    value_adapter: torch.Tensor | None = None

    @property
    def kind(self) -> str:
        return self.tokens.kind

    def attention_map(self):
        """The slot-extraction attention of this text as an AttentionMap."""
        from .attention import AttentionMap
        return AttentionMap(self.tokens, self.attention, self.attention_logits)

    def tensors(self) -> dict[str, torch.Tensor]:
        """Every captured tensor by name (layer outputs as "layer_<n>")."""
        names = ("attention", "attention_logits", "hidden_states", "slot_query_projection", "key_projection",
                 "value_projection", "slot_readout", "slot_states_pre_norm", "slot_states", "routing_raw",
                 "routing", "value_linear", "value_adapter", "value", "slot_queries")
        out = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        out.update({f"layer_{n}": t for n, t in self.layer_states.items()})
        return out

    def summary(self) -> dict:
        """Shapes and norms of the captured tensors (cheap to print or save)."""
        out = {}
        for name, tensor in self.tensors().items():
            entry = {"shape": list(tensor.shape)}
            if tensor.ndim >= 1 and name not in ("attention", "attention_logits"):
                norms = tensor.norm(dim=-1)
                entry.update(norm_mean=float(norms.mean()), norm_min=float(norms.min()), norm_max=float(norms.max()))
            out[name] = entry
        return out

    def save(self, directory) -> None:
        """Write the tensors as Safetensors plus `trace.json` (token view and summary) to a new directory."""
        from pathlib import Path
        from safetensors.torch import save_file
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        save_file({k: v.contiguous() for k, v in self.tensors().items()}, str(root / "tensors.safetensors"))
        (root / "trace.json").write_text(self.to_json(indent=1) + "\n", encoding="utf-8")

    def _payload(self) -> dict:
        return {"representation_id": self.representation_id, "compute_dtype": self.compute_dtype,
                "network_dtype": self.network_dtype, "tokens": self.tokens, "tensors": self.summary()}

    def _html(self) -> str:
        from ._html import trace_html
        return trace_html(self)


@dataclass(frozen=True, eq=False)
class TraceSet(ResearchObject):
    """Traces of several texts encoded together; `encoding()` rebuilds the encoders' output."""
    json_type = "trace_set"

    traces: tuple[Trace, ...]
    kind: str
    representation_id: str
    batch_size: int

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int) -> Trace:
        return self.traces[index]

    def __iter__(self):
        return iter(self.traces)

    def encoding(self) -> Embeddings:
        """The `Embeddings` the encoder returns for these texts with return_attention and return_tokens."""
        if not self.traces:
            raise ValueError("No traces.")
        return Embeddings(torch.stack([t.routing for t in self.traces]), torch.stack([t.value for t in self.traces]),
                          self.representation_id, self.kind,
                          attention=tuple(t.attention for t in self.traces),
                          input_ids=tuple(torch.tensor(t.tokens.input_ids, dtype=torch.long) for t in self.traces),
                          tokens=tuple(t.tokens.tokens for t in self.traces),
                          token_texts=tuple(t.tokens.token_texts for t in self.traces),
                          offsets=tuple(t.tokens.offsets for t in self.traces),
                          truncated=tuple(t.tokens.truncated for t in self.traces))

    def attention_maps(self) -> list:
        return [t.attention_map() for t in self.traces]

    def _payload(self) -> dict:
        return {"kind": self.kind, "representation_id": self.representation_id, "batch_size": self.batch_size,
                "traces": list(self.traces)}

    def _html(self) -> str:
        from ._html import trace_set_html
        return trace_set_html(self)


def decoder_layers(backbone: nn.Module) -> nn.ModuleList:
    """The backbone's text transformer blocks (Qwen text model: `layers`; the multimodal Qwen3.5 model of
    v1.2-preview: `language_model.layers`; BERT-like: `encoder.layer`)."""
    for name in ("layers", "language_model.layers", "model.layers", "text_model.layers", "encoder.layer", "h"):
        module = backbone
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, nn.ModuleList):
            return module
    raise ValueError("Could not find the backbone's decoder layers; layer capture is not supported for "
                     f"{type(backbone).__name__}.")


@contextmanager
def _hooks(network, capture: set[str], layers: tuple[int, ...]):
    """Temporary forward hooks that store tensors in a dict; removed on exit."""
    store: dict = {}
    handles = []
    extractor = network.slot_extractor

    def keep(name, index=None):
        def hook(module, inputs, output):
            store[name] = output if index is None else output[index]
        return hook

    def keep_input(name):
        def hook(module, inputs):
            store[name] = inputs[0]
        return hook

    try:
        handles.append(extractor.register_forward_pre_hook(keep_input("hidden")))
        handles.append(network.routing_head.register_forward_hook(keep("routing_raw")))
        if capture & {"logits", "projections"}:
            handles.append(extractor.q_proj.register_forward_hook(keep("q")))
            handles.append(extractor.k_proj.register_forward_hook(keep("k")))
        if "projections" in capture:
            handles.append(extractor.v_proj.register_forward_hook(keep("v")))
        if "slot_states" in capture:
            handles.append(extractor.out_proj.register_forward_pre_hook(keep_input("readout")))
            handles.append(extractor.out_proj.register_forward_hook(keep("pre_norm")))
        if "value_parts" in capture:
            handles.append(network.value_head.register_forward_hook(keep("value_linear")))
            if network.value_adapter is not None:
                handles.append(network.value_adapter.register_forward_hook(keep("value_adapter")))
        if layers:
            blocks = decoder_layers(network.backbone)
            for n in layers:
                handles.append(blocks[n].register_forward_hook(_layer_hook(store, n)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def _layer_hook(store, n):
    def hook(module, inputs, output):
        store[f"layer_{n}"] = output[0] if isinstance(output, (tuple, list)) else output
    return hook


def _heads_view(extractor, x: torch.Tensor) -> torch.Tensor:
    """[batch, n, A] -> [batch, heads, n, a], as SlotExtractor.forward reshapes q/k/v."""
    batch, n = x.shape[:2]
    if extractor.num_heads > 1:
        return x.reshape(batch, n, extractor.num_heads, extractor.head_dim).transpose(1, 2)
    return x.unsqueeze(1)


def attention_logits(extractor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre-softmax slot attention [batch, heads, slots, tokens] with the extractor's own operations."""
    batch = q.size(0)
    if extractor.num_heads > 1:
        q = q.reshape(batch, extractor.num_slots, extractor.num_heads, extractor.head_dim).transpose(1, 2)
        k = k.reshape(batch, k.size(1), extractor.num_heads, extractor.head_dim).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1)) * extractor.scale
    return (torch.matmul(q, k.transpose(-2, -1)) * extractor.scale).unsqueeze(1)


def _cpu(x: torch.Tensor | None) -> torch.Tensor | None:
    """A float32 CPU copy usable with autograd (never an inference-mode tensor)."""
    if x is None:
        return None
    y = x.detach().to(device="cpu", dtype=torch.float32)
    return y.clone() if y.is_inference() else y


def network_dtype(model) -> str:
    return str(model.network.routing_head.weight.dtype).removeprefix("torch.")


def trace(model, texts: list[str], kind: str, *, instruction: str | None = None,
          instructions: list[str] | None = None, batch_size: int = 1, capture=DEFAULT_CAPTURE,
          layers=None) -> TraceSet:
    """Encode `texts` and keep the encoder's intermediate tensors (see `Trace`).

    kind is "query" or "document"; instruction/instructions and batch_size mean what they mean
    for the encoders (batch_size 1, the default, makes every text independent of the others).
    capture: any of "hidden", "logits", "projections", "slot_states", "value_parts" ("all" for
    every one). layers: indices of backbone decoder layers whose outputs to keep (negative
    indices count from the end); each costs T x H floats per text.
    """
    if capture == "all":
        capture = CAPTURE_OPTIONS
    capture = set(capture)
    unknown = capture - set(CAPTURE_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown capture option(s) {sorted(unknown)}; expected {CAPTURE_OPTIONS}.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    instructions = resolve_instructions(model, texts, kind, instruction, instructions)
    network = model.network
    layer_ids: tuple[int, ...] = ()
    if layers:
        count = len(decoder_layers(network.backbone))
        if any(type(n) is not int or not -count <= n < count for n in layers):
            raise ValueError(f"Layer indices must be integers in [-{count}, {count}).")
        layer_ids = tuple(sorted({n % count for n in layers}))
    formatted = model.format_texts(list(texts), kind, instructions=instructions)
    extractor = network.slot_extractor
    slot_queries = _cpu(extractor.slot_queries)
    dtype_name = network_dtype(model)
    out = []
    for start in range(0, len(formatted), batch_size):
        chunk_texts = list(texts[start:start + batch_size])
        chunk_instructions = None if instructions is None else list(instructions[start:start + batch_size])
        batch, items = model._tokenize_batch(chunk_texts, formatted[start:start + batch_size], model._limit(kind), True)
        mask = batch["attention_mask"]
        if not bool(mask.sum(dim=1).all()):
            raise ValueError("A text produced no tokens (empty document?); it cannot be encoded.")
        views = build_views(model, chunk_texts, kind, chunk_instructions, batch, items)
        with torch.inference_mode(), _hooks(network, capture, layer_ids) as store:
            output = network(input_ids=batch["input_ids"].to(model.device), attention_mask=mask.to(model.device))
            logits = attention_logits(extractor, store["q"], store["k"]) if "logits" in capture else None
        attention = output.attention if output.attention.ndim == 4 else output.attention.unsqueeze(1)
        routing = _cpu(F.normalize(output.routing.float(), dim=-1))  # exactly as the encoders
        value = _cpu(output.value)
        for b, view in enumerate(views):
            keep = mask[b].bool()

            def tokens_of(x, keep=keep, b=b):
                return None if x is None else _cpu(x[b][..., keep.to(x.device), :])
            extra = {}
            if "logits" in capture:
                extra["attention_logits"] = _cpu(logits[b][..., keep.to(logits.device)])
            if "hidden" in capture:
                extra["hidden_states"] = tokens_of(store["hidden"])
            if "projections" in capture:
                extra["slot_query_projection"] = _cpu(_heads_view(extractor, store["q"])[b])
                extra["key_projection"] = _cpu(_heads_view(extractor, store["k"])[b][:, keep.to(store["k"].device)])
                extra["value_projection"] = _cpu(_heads_view(extractor, store["v"])[b][:, keep.to(store["v"].device)])
            if "slot_states" in capture:
                extra["slot_readout"] = _cpu(store["readout"][b])
                extra["slot_states_pre_norm"] = _cpu(store["pre_norm"][b])
                extra["slot_states"] = _cpu(output.slot_states[b])
            if "value_parts" in capture:
                extra["value_linear"] = _cpu(store["value_linear"][b])
                if "value_adapter" in store:
                    extra["value_adapter"] = _cpu(store["value_adapter"][b])
            layer_states = {n: tokens_of(store[f"layer_{n}"]) for n in layer_ids}
            out.append(Trace(tokens=view, representation_id=model.representation_id,
                             compute_dtype=model.compute_dtype, network_dtype=dtype_name,
                             attention=_cpu(attention[b][..., keep.to(attention.device)]),
                             routing_raw=_cpu(output.routing[b]), routing=routing[b], value=value[b],
                             slot_queries=slot_queries, layer_states=layer_states, **extra))
    return TraceSet(tuple(out), kind, model.representation_id, batch_size)


def run_heads(network, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """The part of QKVRetriever.forward after the backbone: slot extractor, routing and value heads.

    hidden_states [batch, T, H] in the network's dtype and device; attention_mask [batch, T]. With
    the backbone's own output and mask this reproduces the forward pass exactly; with a reduced
    mask the slots cannot read the masked tokens (whose backbone states are unchanged).
    Returns (routing_raw, value, attention).
    """
    slot_states, attention = network.slot_extractor(hidden_states, attention_mask)
    routing = network.routing_head(slot_states)
    value_states = slot_states
    if network.value_slot_extractor is not None:
        value_states, _ = network.value_slot_extractor(hidden_states, attention_mask)
    value = network.compute_value(value_states)
    return routing, value, attention
