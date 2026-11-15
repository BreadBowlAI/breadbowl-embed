"""One query-document pair, fully inspected: attention, score decomposition and attribution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._base import ResearchObject
from .attention import AttentionMap
from .capture import trace
from .decomposition import COMPONENTS, ScoreDecomposition, TokenAttribution, attribute, decompose
from .rules import ScoringRule


def model_summary(model) -> dict:
    """Identity and settings of a loaded model, as recorded in research results."""
    architecture = model.config["architecture"]
    return {"representation_id": model.representation_id, "release_id": model.release_id,
            "compute_dtype": model.compute_dtype, "device": str(model.device),
            "slots": architecture["slot_extractor"]["num_slots"],
            "slot_heads": architecture["slot_extractor"].get("num_heads", 1),
            "scoring": model.scoring.to_dict(), "query_format": model.config["preprocessing"]["query_format"],
            "max_query_length": model.config["preprocessing"]["max_query_length"],
            "max_document_length": model.config["preprocessing"]["max_document_length"]}


@dataclass(frozen=True, eq=False)
class PairInspection(ResearchObject):
    """Everything the toolkit can say about one (query, document) score without intervening.

    query / document: AttentionMaps (slot x token attention with pre-softmax logits);
    decomposition: the exact slot-pair decomposition of the score; `attribution(...)` builds
    attention-weighted token attribution from both. `report()` renders a self-contained HTML page.
    """
    json_type = "pair_inspection"

    query: AttentionMap
    document: AttentionMap
    decomposition: ScoreDecomposition
    model: dict

    @property
    def scores(self) -> dict:
        d = self.decomposition
        return {"final": d.final, "routing": d.routing, "value": d.value}

    def attribution(self, side: str = "document", component: str = "final", head: int | None = None) -> TokenAttribution:
        attention = self.document if side == "document" else self.query
        return attribute(self.decomposition, attention, side=side, component=component, head=head)

    def report(self, path: str | Path | None = None, *, title: str | None = None) -> str:
        """Self-contained HTML report (works offline). Writes it to `path` if given; returns the HTML."""
        from .report import render_pair_report
        html = render_pair_report(self, title=title)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def _payload(self) -> dict:
        return {"model": self.model, "scores": self.scores, "query": self.query, "document": self.document,
                "decomposition": self.decomposition,
                "attribution": {f"{side}_{component}": self.attribution(side, component)._payload_compact()
                                for side in ("query", "document") for component in COMPONENTS}}

    def _html(self) -> str:
        from ._html import pair_html
        return pair_html(self)


def inspect_pair(model, query: str, document: str, *, instruction: str | None = None, rule=None) -> PairInspection:
    """Encode one query and one document with tracing and decompose their score.

    rule: optional scoring-rule override (preset name, ScoringRule or ScoringConfig); default the
    model's own settings. Costs one query and one document forward pass.
    """
    if not isinstance(query, str) or not isinstance(document, str):
        raise TypeError("Pass one query string and one document string.")
    queries = trace(model, [query], "query", instruction=instruction, capture=("logits",))
    documents = trace(model, [document], "document", capture=("logits",))
    base = ScoringRule.from_config(model.scoring)
    [[decomposition]] = decompose(queries.encoding(), documents.encoding(),
                                  rule=base if rule is None else rule, base=model)
    return PairInspection(queries[0].attention_map(), documents[0].attention_map(), decomposition, model_summary(model))
