"""Scoring-rule overrides: rescore the same representations under another rule.

A `ScoringRule` is a named `ScoringConfig`. Rescoring builds a new config for the call and
runs the library's own `score` with it; the model's `scoring` is never modified.

Presets (relative to the model's own settings, tau_k 0.02, tau_v 0.1, value_mix 0.25):
  "model"         the model's rule: 0.75 R + 0.25 V.
  "routing-only"  R alone (value_mix 0).
  "value-only"    V alone (value_mix 1).
  "maxsim"        plain MaxSim over slots: mean over query slots of the best routing cosine
                  (qk_mode "max", value_mix 0), a ColBERT-style baseline on the same vectors.
  "maxsim+value"  MaxSim in place of log-mean-exp, with the model's value term.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..config import ScoringConfig
from ..embeddings import Embeddings
from ..scoring import Scores, score
from ._base import ResearchObject
from ._metrics import kendall_tau, ranking, spearman, top_k_overlap

PRESETS = ("model", "routing-only", "value-only", "maxsim", "maxsim+value")


@dataclass(frozen=True)
class ScoringRule:
    """A named scoring configuration (see ScoringConfig for the parameters)."""
    name: str = "model"
    qk_mode: str = "logmeanexp"
    tau_k: float = 0.02
    tau_v: float = 0.1
    value_mix: float = 0.25

    def __post_init__(self):
        self.config()  # validate

    def config(self) -> ScoringConfig:
        return ScoringConfig(qk_mode=self.qk_mode, tau_k=self.tau_k, tau_v=self.tau_v, value_mix=self.value_mix)

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, name: str | None = None, **changes) -> "ScoringRule":
        """A copy with changed parameters (named "custom" unless `name` is given)."""
        values = {**self.to_dict(), **changes, "name": name or ("custom" if changes else self.name)}
        return ScoringRule(**values)

    @classmethod
    def from_config(cls, config: ScoringConfig, name: str = "model") -> "ScoringRule":
        return cls(name=name, **config.to_dict())

    @classmethod
    def preset(cls, name: str, base: ScoringConfig | None = None) -> "ScoringRule":
        """One of PRESETS, relative to `base` (default: the library's default settings)."""
        rule = cls.from_config(base or ScoringConfig(), "model")
        if name == "model":
            return rule
        if name == "routing-only":
            return rule.replace(name, value_mix=0.0)
        if name == "value-only":
            return rule.replace(name, value_mix=1.0)
        if name == "maxsim":
            return rule.replace(name, qk_mode="max", value_mix=0.0)
        if name == "maxsim+value":
            return rule.replace(name, qk_mode="max")
        raise ValueError(f"Unknown preset {name!r}; expected one of {PRESETS}.")


def resolve_rule(rule, base: ScoringConfig | None = None) -> ScoringRule:
    """A ScoringRule from None (the base/model rule), a preset name, a ScoringConfig or a ScoringRule."""
    if rule is None:
        return ScoringRule.from_config(base or ScoringConfig())
    if isinstance(rule, ScoringRule):
        return rule
    if isinstance(rule, ScoringConfig):
        return ScoringRule.from_config(rule, "custom")
    if isinstance(rule, str):
        return ScoringRule.preset(rule, base)
    raise TypeError("rule must be a ScoringRule, a ScoringConfig, a preset name or None.")


def base_config(model_or_config) -> ScoringConfig:
    if model_or_config is None:
        return ScoringConfig()
    if isinstance(model_or_config, ScoringConfig):
        return model_or_config
    return ScoringConfig(**model_or_config.scoring.to_dict())  # a BreadBowl model: copy, never share


def rescore(queries: Embeddings, documents: Embeddings, rule=None, *, base=None) -> Scores:
    """[queries, documents] Scores under `rule` (preset name, ScoringRule or ScoringConfig).

    `base` (a model or ScoringConfig) supplies the settings presets are relative to.
    """
    return score(queries, documents, config=resolve_rule(rule, base_config(base)).config())


@dataclass(frozen=True, eq=False)
class RuleComparison(ResearchObject):
    """Scores and rankings of the same pairs under several scoring rules.

    scores[name]: [queries, documents] final scores under that rule. Agreement statistics compare
    every rule's per-query ranking with the reference rule's (Kendall tau-b and Spearman rho over
    all documents, top-1 agreement, top-k overlap), averaged over queries.
    """
    json_type = "rule_comparison"

    rules: tuple[ScoringRule, ...]
    reference: str
    scores: dict
    agreement: dict
    top_k: int

    def table(self) -> list[dict]:
        return [{"rule": r.name, **{k: v for k, v in r.to_dict().items() if k != "name"}, **self.agreement[r.name]}
                for r in self.rules]

    def rankings(self, query: int = 0) -> dict[str, list[int]]:
        return {name: ranking(s[query]) for name, s in self.scores.items()}

    def _payload(self) -> dict:
        return {"reference": self.reference, "top_k": self.top_k, "rules": [r.to_dict() for r in self.rules],
                "agreement": self.agreement, "scores": self.scores}

    def _html(self) -> str:
        from ._html import table_html
        return table_html("Scoring rules", self.table(), note=f"Agreement with {self.reference!r}, mean over queries.")


def agreement(reference: torch.Tensor, other: torch.Tensor, top_k: int) -> dict:
    """Ranking agreement of two [queries, documents] score matrices, averaged over queries."""
    rows = []
    for a, b in zip(reference, other):
        ra, rb = ranking(a), ranking(b)
        rows.append({"kendall_tau": kendall_tau(a, b), "spearman": spearman(a, b), "top1_agreement": float(ra[0] == rb[0]),
                     "top_k_overlap": top_k_overlap(ra, rb, top_k), "same_ranking": float(ra == rb)})
    if not rows:
        return {}
    delta = (other.double() - reference.double()).abs()
    return {**{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
            "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean())}


def compare_rules(queries: Embeddings, documents: Embeddings, rules=PRESETS, *, base=None, reference: str = "model",
                  top_k: int = 10) -> RuleComparison:
    """Score every pair under each rule and compare rankings with the reference rule.

    rules: preset names, ScoringRules or ScoringConfigs (default: every preset). base: a model or
    ScoringConfig the presets are relative to. Absolute scores of different rules are on
    different scales; compare rankings, not values, across rules.
    """
    config = base_config(base)
    resolved = [resolve_rule(r, config) for r in rules]
    if len({r.name for r in resolved}) != len(resolved):
        raise ValueError("Give every rule a distinct name.")
    if reference not in {r.name for r in resolved}:
        resolved.insert(0, resolve_rule(reference, config))
    scores = {r.name: score(queries, documents, config=r.config()).combined for r in resolved}
    k = min(top_k, len(documents)) if len(documents) else 0
    return RuleComparison(tuple(resolved), reference, scores,
                          {name: agreement(scores[reference], s, k) for name, s in scores.items()}, k)
