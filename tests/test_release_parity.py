"""Release gates that need real weights; skipped unless the files are supplied.

Step 5152 parity against the research loader (tools/verify_export.py), set:
  BREADBOWL_PARITY_CHECKPOINT   training checkpoint model.pt
  BREADBOWL_PARITY_BASE_MODEL   base-model directory (config, tokenizer, weights)
  BREADBOWL_PARITY_RELEASE      exported release directory
  BREADBOWL_PARITY_RESEARCH_SRC research checkout's src directory
  BREADBOWL_PARITY_RESEARCH_PYTHON (optional) Python of the research environment

v1.2-preview backward compatibility, set BREADBOWL_V12_RELEASE to a downloaded snapshot of
dotproductx/breadbowl-embed-v1.2-preview at the revision in docs/hub-verification.json.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARITY = ["BREADBOWL_PARITY_CHECKPOINT", "BREADBOWL_PARITY_BASE_MODEL", "BREADBOWL_PARITY_RELEASE",
          "BREADBOWL_PARITY_RESEARCH_SRC"]

pytestmark = pytest.mark.parity


def _paths(names):
    values = [os.environ.get(name) for name in names]
    if not all(values) or not all(Path(v).exists() for v in values):
        pytest.skip("set " + ", ".join(names) + " to existing paths to run this release gate")
    return values


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_release_matches_research_loader(tmp_path, dtype):
    checkpoint, base, release, source = _paths(PARITY)
    report = tmp_path / "parity.json"
    command = [sys.executable, str(ROOT / "tools/verify_export.py"), "all", "--research-src", source,
               "--checkpoint", checkpoint, "--base-model", base, "--release", release, "--dtype", dtype,
               "--output", str(report)]
    if os.environ.get("BREADBOWL_PARITY_RESEARCH_PYTHON"):
        command += ["--research-python", os.environ["BREADBOWL_PARITY_RESEARCH_PYTHON"]]
    subprocess.run(command, check=True)
    result = json.loads(report.read_text())
    assert result["passed"], result["gate"]
    assert result["queries"] >= 20 and result["documents"] >= 20
    assert result["token_counts"]["documents_truncated"] >= 1  # one document exceeds the token limit
    assert result["token_counts"]["queries_truncated"] >= 1  # and one query
    differences = result["max_absolute_differences"]
    for name in ("query_routing_normalised", "document_routing_normalised", "query_value", "document_value",
                 "query_attention", "document_attention"):
        assert differences[name] <= 1e-5, name
    for name in ("final_scores", "routing_scores_R", "value_scores_V"):
        assert differences[name] <= 1e-5, name
    # The default (one text per forward pass) against the research model encoding each text alone.
    assert result["gate"]["default_batch_size_within_tolerance"]
    assert result["gate"]["public_helpers_consistent"]  # score_routing, tokenize, display strings


def test_v12_preview_release_still_loads_and_reproduces_recorded_scores():
    [release] = _paths(["BREADBOWL_V12_RELEASE"])
    from breadbowl_embed import BreadBowl
    recorded = json.loads((ROOT / "docs/hub-verification.json").read_text())
    model = BreadBowl.from_pretrained(release, local_files_only=True)
    assert model.representation_id == recorded["representation_id"]
    passages = ["Octopuses have three hearts.", "The Pacific is Earth's largest ocean.",
                "Plants use sunlight during photosynthesis."]
    hits = model.rerank("How many hearts does an octopus have?", passages)
    assert [h.id for h in hits] == [r["id"] for r in recorded["results"]]
    # The v1.2 backbone is bfloat16, so CPU bf16 kernels shift scores by up to ~7e-3 between
    # machines and torch/transformers versions; v0.1.0a1 and this version give identical outputs
    # on the same stack (docs/validation.md). Identity and ranking are exact checks.
    for hit, expected in zip(hits, recorded["results"]):
        for field in ("score", "routing_score", "value_score"):
            assert getattr(hit, field) == pytest.approx(expected[field], abs=1e-2)
