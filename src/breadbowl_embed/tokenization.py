"""What the model reads: token ids, truncation, vocabulary strings, display strings and offsets."""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class TokenizedText:
    """One text as the encoder sees it, without running the model.

    text:           the caller's text.
    formatted:      the exact model input (query or document template applied).
    span:           (start, end) of `text` inside `formatted`.
    input_ids:      [tokens] ids after truncation at the model's limit (no padding).
    tokens:         vocabulary strings (byte-level for Qwen, e.g. "Ġhearts").
    token_texts:    display strings: the decoded text each token completes. Joined they equal the
                    decoded input; a token ending inside a multi-byte character shows "" and the
                    token that completes the character shows all of it, so nothing is "�".
    offsets:        (start, end) character offsets into `text`, or None for tokens that come only
                    from the template/instruction or are special tokens.
    segments:       "text" (overlaps the caller's text), "template" or "special" per token.
    truncated:      whether the token limit cut the input.
    untruncated_token_count: tokens the input would have without the limit.
    """
    text: str
    formatted: str
    span: tuple[int, int]
    input_ids: torch.Tensor
    tokens: tuple[str, ...]
    token_texts: tuple[str, ...]
    offsets: tuple
    segments: tuple[str, ...]
    truncated: bool
    untruncated_token_count: int

    @property
    def token_count(self) -> int:
        return len(self.input_ids)


def map_offsets(offsets, span) -> tuple:
    """Token offsets in the formatted text -> offsets in the caller's text (None outside it)."""
    first, last = span
    mapped = []
    for start, end in offsets:
        if start < end and start < last and end > first:
            mapped.append((max(start, first) - first, min(end, last) - first))
        else:
            mapped.append(None)
    return tuple(mapped)


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, characters)}


def _is_byte_level(tokenizer) -> bool:
    cached = getattr(tokenizer, "_breadbowl_byte_level", None)
    if cached is None:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        cached = type(decoder).__name__ == "ByteLevel"
        try:
            tokenizer._breadbowl_byte_level = cached
        except AttributeError:  # pragma: no cover - read-only tokenizer objects
            pass
    return cached


def display_strings(tokenizer, ids: list[int]) -> tuple[str, ...]:
    """Per-token display strings whose concatenation is tokenizer.decode(ids)."""
    if not ids:
        return ()
    if _is_byte_level(tokenizer):
        added = tokenizer.added_tokens_decoder
        table = _byte_decoder()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        out = []
        for token_id, token in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
            if token_id in added:
                data = added[token_id].content.encode("utf-8")
            else:
                data = bytes(table[c] for c in token)
            out.append(decoder.decode(data))
        out[-1] += decoder.decode(b"", final=True)
        return tuple(out)
    # Other tokenizers: attribute each newly stable prefix of the decoded text to the token
    # that produced it (O(n^2) decodes; fine at the model's token limits).
    full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    out, printed = [], 0
    for k in range(len(ids)):
        prefix = tokenizer.decode(ids[:k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        stable = 0
        for a, b in zip(prefix, full):
            if a != b:
                break
            stable += 1
        out.append(full[printed:stable] if stable > printed else "")
        printed = max(printed, stable)
    out[-1] += full[printed:]
    return tuple(out)


def segments_for(ids: list[int], offsets: tuple, special_ids: frozenset) -> tuple[str, ...]:
    return tuple("text" if offset is not None else ("special" if token_id in special_ids else "template")
                 for token_id, offset in zip(ids, offsets))
