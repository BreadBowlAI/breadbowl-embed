"""Slot-extraction attention as a first-class object.

Each of the 16 slots is pooled from the backbone's token states by a learned query attending
over the text's tokens, in 4 heads. An `AttentionMap` holds that attention for one text
([heads, slots, tokens], each row a probability distribution over tokens), together with the
`TokenView` that maps tokens back to the caller's words and characters and to the query
instruction. It describes how the encoder pooled tokens into slots; it is not the
query-to-document attention used in scoring (see `decomposition`), and attention weights are
not an explanation of the model's output on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._base import ResearchObject
from .tokens import SEGMENTS, TokenView

HEAD_MEAN = None


@dataclass(frozen=True, eq=False)
class AttentionMap(ResearchObject):
    """Slot x token attention of one text.

    tokens:  TokenView of the text (segments, offsets, words).
    weights: [heads, slots, tokens] post-softmax attention; every row sums to 1 over tokens.
    logits:  [heads, slots, tokens] pre-softmax scores, when traced (else None).

    Methods that take `head` use the mean over heads when head is None (the default). The heads
    are concatenated, then projected, to form a slot state, so the head mean is a summary
    convention, not a quantity the model computes.
    """
    json_type = "attention_map"

    tokens: TokenView
    weights: torch.Tensor
    logits: torch.Tensor | None = None

    def __post_init__(self):
        if self.weights.ndim == 2:
            object.__setattr__(self, "weights", self.weights.unsqueeze(0))
        if self.weights.ndim != 3 or self.weights.shape[-1] != len(self.tokens):
            raise ValueError("Attention must be [heads, slots, tokens] over the view's tokens.")
        if self.logits is not None and self.logits.shape != self.weights.shape:
            raise ValueError("Logits and weights shapes differ.")

    @classmethod
    def from_encoding(cls, model, encoding, index: int, text: str, *, instruction: str | None = None) -> "AttentionMap":
        """Wrap the attention of an encoding made with return_attention=True.

        `text` (and for queries the instruction it was encoded with) must be the ones encoded; the
        tokenizer is re-run to recover segments and offsets and the token ids are checked.
        """
        from .tokens import token_views
        if encoding.attention is None:
            raise ValueError("Encode with return_attention=True.")
        [view] = token_views(model, [text], encoding.kind, instruction=instruction)
        weights = encoding.attention[index]
        if weights.shape[-1] != len(view) or (encoding.input_ids is not None and
                                              list(encoding.input_ids[index].tolist()) != list(view.input_ids)):
            raise ValueError("The text/instruction does not match the encoded item.")
        return cls(view, weights.float())

    # ---- shape --------------------------------------------------------------------------------
    @property
    def num_heads(self) -> int:
        return self.weights.shape[0]

    @property
    def num_slots(self) -> int:
        return self.weights.shape[1]

    @property
    def num_tokens(self) -> int:
        return self.weights.shape[2]

    def matrix(self, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots, tokens] attention of one head, or the mean over heads (head=None)."""
        if head is None:
            return self.weights.mean(dim=0)
        if not -self.num_heads <= head < self.num_heads:
            raise IndexError(f"head must be in [0, {self.num_heads}).")
        return self.weights[head]

    def slot(self, slot: int, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[tokens] attention of one slot."""
        return self.matrix(head)[slot]

    # ---- per-slot statistics ------------------------------------------------------------------
    def entropy(self, head: int | None = HEAD_MEAN, *, normalized: bool = False) -> torch.Tensor:
        """[slots] entropy (nats) of each slot's distribution over tokens.

        normalized=True divides by log(tokens): 0 = one token, 1 = uniform.
        """
        p = self.matrix(head).double()
        h = -torch.xlogy(p, p).sum(-1)
        if normalized:
            h = h / math.log(self.num_tokens) if self.num_tokens > 1 else torch.zeros_like(h)
        return h.float()

    def entropy_per_head(self) -> torch.Tensor:
        """[heads, slots] entropy (nats) of every head's distribution."""
        p = self.weights.double()
        return (-torch.xlogy(p, p).sum(-1)).float()

    def effective_tokens(self, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots] exp(entropy): how many equally weighted tokens would give the same entropy."""
        return self.entropy(head).double().exp().float()

    def coverage(self, mass: float = 0.9, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots] fewest tokens that together hold `mass` of each slot's attention."""
        if not 0 < mass <= 1:
            raise ValueError("mass must be in (0, 1].")
        ordered = torch.sort(self.matrix(head).double(), dim=-1, descending=True).values.cumsum(-1)
        # A small tolerance so that mass=1 counts the tokens up to the last non-zero weight.
        return (ordered < mass - 1e-9).sum(-1).add(1).clamp(max=self.num_tokens)

    def coverage_fraction(self, mass: float = 0.9, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots] coverage(mass) as a fraction of the text's tokens."""
        return self.coverage(mass, head).float() / self.num_tokens

    def head_divergence(self) -> torch.Tensor:
        """[slots] mean Jensen-Shannon divergence (nats) between pairs of heads; 0 if heads agree."""
        p = self.weights.double()
        if self.num_heads < 2:
            return torch.zeros(self.num_slots)
        total, pairs = torch.zeros(self.num_slots, dtype=torch.float64), 0
        for a in range(self.num_heads):
            for b in range(a + 1, self.num_heads):
                m = 0.5 * (p[a] + p[b])
                kl_a = (torch.xlogy(p[a], p[a]) - torch.xlogy(p[a], m)).sum(-1)
                kl_b = (torch.xlogy(p[b], p[b]) - torch.xlogy(p[b], m)).sum(-1)
                total += 0.5 * (kl_a + kl_b)
                pairs += 1
        return (total / pairs).float()

    def segment_mass(self, head: int | None = HEAD_MEAN) -> dict[str, torch.Tensor]:
        """Attention mass of each slot per segment: {"instruction", "template", "text", "special"} -> [slots].

        For queries this says which slots read the instruction and which read the query text.
        The four masses of a slot sum to 1.
        """
        w = self.matrix(head)
        return {segment: w[:, self.tokens.segment_mask(segment)].sum(-1) for segment in SEGMENTS}

    def dominant_segment(self, head: int | None = HEAD_MEAN) -> list[str]:
        """For each slot, the segment holding most of its attention."""
        mass = self.segment_mass(head)
        stacked = torch.stack([mass[s] for s in SEGMENTS], dim=-1)
        return [SEGMENTS[i] for i in stacked.argmax(-1).tolist()]

    def slots_reading(self, segment: str, threshold: float = 0.5, head: int | None = HEAD_MEAN) -> list[int]:
        """Slots that put at least `threshold` of their attention on `segment` (e.g. "instruction")."""
        if segment not in SEGMENTS:
            raise ValueError(f"segment must be one of {SEGMENTS}.")
        return [s for s, m in enumerate(self.segment_mass(head)[segment].tolist()) if m >= threshold]

    def statistics(self, head: int | None = HEAD_MEAN, mass: float = 0.9) -> list[dict]:
        """Per-slot summary: entropy, normalised entropy, effective tokens, coverage, segment masses."""
        entropy = self.entropy(head)
        normalized = self.entropy(head, normalized=True)
        coverage = self.coverage(mass, head)
        segments = self.segment_mass(head)
        divergence = self.head_divergence()
        return [{"slot": s, "entropy": float(entropy[s]), "normalized_entropy": float(normalized[s]),
                 "effective_tokens": math.exp(float(entropy[s])), f"coverage_{int(mass * 100)}": int(coverage[s]),
                 "coverage_fraction": float(coverage[s]) / self.num_tokens,
                 "head_divergence": float(divergence[s]),
                 "segment_mass": {k: float(v[s]) for k, v in segments.items()}}
                for s in range(self.num_slots)]

    # ---- tokens, words and spans --------------------------------------------------------------
    def top_tokens(self, k: int = 5, head: int | None = HEAD_MEAN, *, segments=None) -> list[list[dict]]:
        """For each slot, its k most attended tokens (optionally only from the given segments)."""
        w = self.matrix(head)
        allowed = None if segments is None else self.tokens.segment_mask(*segments)
        out = []
        for slot in range(self.num_slots):
            row = w[slot] if allowed is None else w[slot].masked_fill(~allowed, -1.0)
            order = torch.argsort(row, descending=True, stable=True)[:k].tolist()
            out.append([{**self.tokens.describe_token(t), "weight": float(w[slot, t])}
                        for t in order if allowed is None or bool(allowed[t])])
        return out

    def word_matrix(self, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots, words] attention mass per word of the caller's text (sum over the word's tokens).

        A token counts towards the first word it overlaps; tokens of the instruction/template,
        special tokens and whitespace-only tokens count towards no word.
        """
        w = self.matrix(head)
        out = torch.zeros(self.num_slots, len(self.tokens.words), dtype=w.dtype)
        for t, word in enumerate(self.tokens.token_words):
            if word is not None:
                out[:, word] += w[:, t]
        return out

    def word_weights(self, slot: int, head: int | None = HEAD_MEAN) -> list[dict]:
        """[{word, start, end, weight}] for one slot, in text order."""
        row = self.word_matrix(head)[slot]
        return [{"index": w.index, "word": w.text, "start": w.start, "end": w.end, "weight": float(row[w.index])}
                for w in self.tokens.words]

    def span_mass(self, start: int, end: int, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots] attention mass of the tokens overlapping characters [start, end) of the caller's text."""
        indices = self.tokens.tokens_in_span(start, end)
        w = self.matrix(head)
        return w[:, indices].sum(-1) if indices else torch.zeros(self.num_slots, dtype=w.dtype)

    def phrase_mass(self, phrase: str, occurrence: int = 0, head: int | None = HEAD_MEAN) -> torch.Tensor:
        """[slots] span_mass of the n-th occurrence of `phrase` in the caller's text."""
        return self.span_mass(*self.tokens.find(phrase, occurrence), head=head)

    def character_spans(self, slot: int, head: int | None = HEAD_MEAN) -> list[tuple[int, int, float]]:
        """(start, end, weight) per caller-text span, for highlighting; tokens sharing a span add up."""
        row = self.slot(slot, head)
        merged: dict[tuple[int, int], float] = {}
        for t, offset in enumerate(self.tokens.offsets):
            if offset is not None:
                merged[offset] = merged.get(offset, 0.0) + float(row[t])
        return [(s, e, w) for (s, e), w in sorted(merged.items())]

    # ---- export -------------------------------------------------------------------------------
    def _payload(self) -> dict:
        return {"tokens": self.tokens, "heads": self.num_heads, "slots": self.num_slots,
                "weights": self.weights, "logits": self.logits, "statistics": self.statistics()}

    def _html(self) -> str:
        from ._html import attention_html
        return attention_html(self)

    def plot(self, **kwargs):
        """Heatmap of slots x tokens (needs the `viz` extra); see research.plot.attention."""
        from .plot import attention
        return attention(self, **kwargs)


def attention_maps(model, texts: list[str], kind: str, *, instruction: str | None = None,
                   instructions: list[str] | None = None, batch_size: int = 1,
                   logits: bool = True) -> list[AttentionMap]:
    """Encode `texts` and return their slot-extraction attention maps (with pre-softmax logits)."""
    from .capture import trace
    traces = trace(model, texts, kind, instruction=instruction, instructions=instructions, batch_size=batch_size,
                   capture=("logits",) if logits else ())
    return traces.attention_maps()
