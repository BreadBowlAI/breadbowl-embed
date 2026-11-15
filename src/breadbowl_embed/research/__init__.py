"""Research toolkit: look inside the encoder and the scorer, and test what you see.

**Experimental.** `breadbowl_embed.research` may change between minor versions (0.x -> 0.y);
the core API (`BreadBowl`, `Embeddings`, `score`, `explain`, `Index`, ...) is stable and is
never changed by this package: every tool here reuses the library's own tokenizer call, forward
pass and scorer, and its defaults reproduce the core numbers exactly.

What each tool answers:

  trace                 What did the encoder compute? Backbone token states, pre-softmax and
                        post-softmax slot attention, slot states before/after the layer norm,
                        raw routing, value parts, learned slot queries (TraceSet / Trace).
  attention_maps        Which tokens did each slot pool from? Per head and head-averaged, mapped
                        to tokens, words and character spans; entropy, coverage, instruction vs
                        query-text mass (AttentionMap).
  decompose             How does a score split over slot pairs? Exact parts of R, V and the final
                        score that recombine to `score` (ScoreDecomposition); `attribute` turns
                        them into attention-weighted token attribution (TokenAttribution).
  inspect_pair          All of the above for one query-document pair, with an HTML report.
  slot_ablation, ablate_slots, occlude, occlusion_sweep
                        Causal checks: remove slots and rescore; delete, replace or hide text
                        spans from the slots and re-encode.
  rescore, compare_rules, ScoringRule
                        Other scoring rules on the same vectors (routing-only, value-only, MaxSim,
                        other temperatures) without touching the model.
  slot_statistics       Do slots collapse? Norms, variance, between-slot cosine, effective rank,
                        which slots win in scoring.
  shortlist_diagnostics How close is a slot-shortlist profile to exact search; per-slot hit recall;
                        how many queries underfill the rerank list.
  compare_precision     Storage dtype x compute dtype x batch size on your own texts.

Every result object has `to_dict()`, `to_json()`, `save_json(path)` and a notebook display
(`_repr_html_`); several have `.plot()` (needs the `viz` extra). See docs/research.md.
"""
from .attention import AttentionMap, attention_maps
from .capture import CAPTURE_OPTIONS, Trace, TraceSet, run_heads, trace
from .decomposition import COMPONENTS, ScoreDecomposition, TokenAttribution, attribute, decompose
from .diagnostics import ShortlistDiagnostics, SlotStatistics, shortlist_diagnostics, slot_statistics
from .interventions import (Occlusion, OcclusionSweep, SlotAblation, ablate_slots, edit_text, occlude,
                            occlusion_sweep, slot_ablation)
from .pair import PairInspection, inspect_pair
from .precision import PrecisionComparison, compare_precision, with_compute_dtype
from .rules import PRESETS, RuleComparison, ScoringRule, compare_rules, rescore
from .tokens import SEGMENTS, TokenView, Word, token_views
from ._base import JSON_SCHEMA, ResearchObject

STABILITY = "experimental"

__all__ = [
    "AttentionMap", "CAPTURE_OPTIONS", "COMPONENTS", "JSON_SCHEMA", "Occlusion", "OcclusionSweep", "PRESETS",
    "PairInspection", "PrecisionComparison", "ResearchObject", "RuleComparison", "SEGMENTS", "STABILITY",
    "ScoreDecomposition", "ScoringRule", "ShortlistDiagnostics", "SlotAblation", "SlotStatistics", "TokenAttribution",
    "TokenView", "Trace", "TraceSet", "Word", "ablate_slots", "attention_maps", "attribute", "compare_precision",
    "compare_rules", "decompose", "edit_text", "inspect_pair", "occlude", "occlusion_sweep", "rescore", "run_heads",
    "shortlist_diagnostics", "slot_ablation", "slot_statistics", "token_views", "trace", "with_compute_dtype",
]
