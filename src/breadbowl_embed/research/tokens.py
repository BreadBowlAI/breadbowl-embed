"""Where every model input token comes from: caller text, query instruction, template or special.

`TokenView` extends the core `TokenizedText` with the offsets of each token in the formatted
model input, an "instruction" segment for queries, and a word segmentation of the caller's text,
so that token-level quantities (attention, attribution) can be summed over words and character
spans of the original text.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import re

import torch

from ..config import format_with_span
from ._base import ResearchObject

SEGMENTS = ("instruction", "template", "text", "special")
# A word is a run of word characters (Unicode \w, so CJK runs count as one word) joined by
# internal apostrophes or hyphens, or a single other non-space character (punctuation).
WORD_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]")


@dataclass(frozen=True)
class Word:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True, eq=False)
class TokenView(ResearchObject):
    """One text as the model reads it, token by token.

    kind:              "query" or "document".
    text:              the caller's text (for queries: the query without its instruction).
    formatted:         the exact model input; `span` is where `text` sits in it.
    instruction:       the query instruction used (None for documents).
    instruction_span:  where the instruction sits in `formatted` (None if not applicable).
    input_ids, tokens, token_texts, offsets, truncated, untruncated_token_count: as in the core
                       `TokenizedText` (offsets are character offsets into `text`, None outside).
    formatted_offsets: (start, end) of every token in `formatted`, None for zero-width tokens.
    segments:          per token "text" (overlaps the caller's text), "instruction" (overlaps the
                       query instruction), "template" (other template characters) or "special".
    """
    json_type = "token_view"

    kind: str
    text: str
    formatted: str
    span: tuple[int, int]
    instruction: str | None
    instruction_span: tuple[int, int] | None
    input_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_texts: tuple[str, ...]
    offsets: tuple
    formatted_offsets: tuple
    segments: tuple[str, ...]
    truncated: bool
    untruncated_token_count: int

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    def segment_mask(self, *segments: str) -> torch.Tensor:
        """Boolean [tokens] mask of tokens in any of the given segments."""
        unknown = set(segments) - set(SEGMENTS)
        if unknown:
            raise ValueError(f"Unknown segment(s) {sorted(unknown)}; expected {SEGMENTS}.")
        return torch.tensor([s in segments for s in self.segments], dtype=torch.bool)

    @cached_property
    def words(self) -> tuple[Word, ...]:
        """Words of the caller's text (see WORD_PATTERN), including any part cut by truncation."""
        return tuple(Word(i, m.start(), m.end(), m.group()) for i, m in enumerate(WORD_PATTERN.finditer(self.text)))

    @cached_property
    def token_words(self) -> tuple[int | None, ...]:
        """For every token, the index of the first word its caller-text span overlaps (or None)."""
        out = []
        starts = [w.start for w in self.words]
        for offset in self.offsets:
            if offset is None:
                out.append(None)
                continue
            start, end = offset
            match = None
            # Words are sorted and disjoint: scan from the last word starting before `end`.
            position = _bisect_right(starts, end - 1) - 1
            while position >= 0 and self.words[position].end > start:
                match = position
                position -= 1
            out.append(match)
        return tuple(out)

    def tokens_in_span(self, start: int, end: int) -> list[int]:
        """Indices of tokens whose caller-text offsets overlap the character span [start, end)."""
        if not 0 <= start <= end <= len(self.text):
            raise ValueError(f"Span ({start}, {end}) lies outside the text (length {len(self.text)}).")
        return [t for t, o in enumerate(self.offsets) if o is not None and o[0] < end and o[1] > start]

    def find(self, substring: str, occurrence: int = 0) -> tuple[int, int]:
        """Character span of the n-th occurrence of `substring` in the caller's text."""
        position = -1
        for _ in range(occurrence + 1):
            position = self.text.find(substring, position + 1)
            if position < 0:
                raise ValueError(f"{substring!r} does not occur {occurrence + 1} time(s) in the text.")
        return position, position + len(substring)

    def describe_token(self, index: int) -> dict:
        offset = self.offsets[index]
        return {"index": index, "token": self.tokens[index], "display": self.token_texts[index],
                "segment": self.segments[index], "offsets": offset,
                "text": None if offset is None else self.text[offset[0]:offset[1]]}

    def _payload(self) -> dict:
        return {"kind": self.kind, "text": self.text, "formatted": self.formatted, "span": self.span,
                "instruction": self.instruction, "instruction_span": self.instruction_span,
                "truncated": self.truncated, "token_count": self.token_count,
                "untruncated_token_count": self.untruncated_token_count,
                "tokens": [{"id": int(i), "token": t, "display": d, "segment": s, "offsets": o,
                            "formatted_offsets": f, "word": w}
                           for i, t, d, s, o, f, w in zip(self.input_ids, self.tokens, self.token_texts,
                                                          self.segments, self.offsets, self.formatted_offsets,
                                                          self.token_words)],
                "words": [{"index": w.index, "start": w.start, "end": w.end, "text": w.text} for w in self.words]}

    def _html(self) -> str:
        from ._html import token_view_html
        return token_view_html(self)


def _bisect_right(values: list[int], x: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < values[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def instruction_span(model, kind: str, text: str, instruction: str | None) -> tuple[int, int] | None:
    """Where the instruction lands in the formatted query (None for documents or templates without it)."""
    if kind != "query" or instruction is None:
        return None
    prep = model.config["preprocessing"]
    try:
        _, span = format_with_span(prep["query_format"], "instruction", query=text, instruction=instruction)
    except (ValueError, KeyError):
        return None
    return span


def build_views(model, texts: list[str], kind: str, instructions: list[str] | None, batch, items) -> list[TokenView]:
    """TokenViews from one `BreadBowl._tokenize_batch(..., details=True)` call (same ids as the encoders)."""
    mask = batch["attention_mask"].bool()
    prep = model.config["preprocessing"]
    views = []
    for b, (text, item) in enumerate(zip(texts, items)):
        instruction = None
        if kind == "query":
            instruction = prep["default_instruction"] if instructions is None else instructions[b]
        span = instruction_span(model, kind, text, instruction)
        formatted_offsets = tuple((int(s), int(e)) if s < e else None
                                  for s, e in batch["offset_mapping"][b][mask[b]].tolist())
        ids = tuple(int(i) for i in item.input_ids.tolist())
        segments = []
        for token_id, offset, formatted_offset in zip(ids, item.offsets, formatted_offsets):
            if offset is not None:
                segments.append("text")
            elif token_id in model.special_ids:
                segments.append("special")
            elif span is not None and formatted_offset is not None and formatted_offset[0] < span[1] \
                    and formatted_offset[1] > span[0]:
                segments.append("instruction")
            else:
                segments.append("template")
        views.append(TokenView(kind=kind, text=text, formatted=item.formatted, span=item.span, instruction=instruction,
                               instruction_span=span, input_ids=ids, tokens=item.tokens, token_texts=item.token_texts,
                               offsets=item.offsets, formatted_offsets=formatted_offsets, segments=tuple(segments),
                               truncated=item.truncated, untruncated_token_count=item.untruncated_token_count))
    return views


def resolve_instructions(model, texts, kind: str, instruction, instructions):
    """Validate texts/instructions as the encoders do; return the per-text instruction list or None."""
    if kind not in ("query", "document"):
        raise ValueError("kind must be 'query' or 'document'.")
    instructions = model._resolve_instructions(texts, instruction, instructions)
    model._check_texts(texts, instructions)
    if kind == "document" and instructions is not None:
        raise ValueError("Instructions apply to queries only.")
    return instructions


def token_views(model, texts: list[str], kind: str, *, instruction: str | None = None,
                instructions: list[str] | None = None) -> list[TokenView]:
    """TokenViews for `texts` without running the model (the encoders' tokenizer call)."""
    instructions = resolve_instructions(model, texts, kind, instruction, instructions)
    formatted = model.format_texts(list(texts), kind, instructions=instructions)
    views = []
    for start in range(0, len(formatted), 64):
        part_texts = list(texts[start:start + 64])
        part_instructions = None if instructions is None else list(instructions[start:start + 64])
        batch, items = model._tokenize_batch(part_texts, formatted[start:start + 64], model._limit(kind), True)
        views.extend(build_views(model, part_texts, kind, part_instructions, batch, items))
    return views
