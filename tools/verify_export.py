"""Compare a release with the original research code and checkpoint (the release gate).

Release-engineering only: the research source is needed here, never for inference.

  reference  Load the checkpoint through the research loader (qkv.inference_artifacts), encode the
             fixed texts below and save routing, value, slot attention, input ids and the
             research scorer's final/R/V scores. Run with the research environment.
  compare    Load the release through breadbowl_embed, encode the same texts with the same
             batching and compare everything with the reference. Writes a JSON report and exits
             non-zero when the gate fails.
  all        reference (in a subprocess) then compare.

  python tools/verify_export.py all --research-src /path/to/research/src \
      --checkpoint /path/to/model.pt --base-model /path/to/base-model \
      --release /path/to/release --output parity.json [--dtype float32|bfloat16]

The report holds no local paths. A text containing the literal string "<jepa_mask>" is encoded
on its own as a probe: the research tokenizer reads it as the training sentinel, a release
exported without the sentinel reads it as text, so the probe is reported but not gated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

QUERIES = [
    ("What is late interaction in retrieval?", None),
    ("Can cached document representations be reused?", None),
    ("how many hearts does an octopus have", None),
    ("symptoms of vitamin D deficiency in adults", None),
    ("Wer hat die Relativitätstheorie entwickelt?", None),
    ("東京タワーの高さは何メートルですか", None),
    ("python sort list of dicts by key", None),
    ("covid", None),
    ("¿Cuál es la capital de Australia y por qué no es Sídney?", None),
    ("effects of interest rate hikes on emerging market currencies and capital flows", None),
    ("Find the passage that answers: when did the Berlin Wall fall?",
     "Given a question, retrieve passages that answer it."),
    ("arguments against nuclear power", "Retrieve documents that argue the opposite position."),
    ("SELECT * FROM users WHERE id = 1; -- why is this slow", None),
    ("best 🍕 in naples?", None),
    ("What is the boiling point of water at the top of Mount Everest, and how does altitude change it?", None),
    ("q", None),
    ("transformer attention complexity quadratic sequence length alternatives linear attention "
     "state space models mamba gated delta networks hybrid architectures " * 9, None),  # over 128 tokens
    ("Does int8 quantisation of stored vectors hurt retrieval quality?", None),
    ("photosynthesis light reactions", None),
    ("who wrote the declaration of independence", None),
]

_LONG = ("The history of information retrieval begins with library catalogues and punched cards. "
         "Early systems matched exact terms; later systems weighted terms by frequency, introduced "
         "probabilistic relevance models, and eventually learned dense representations from data. "
         "Late-interaction retrievers keep several vectors per passage and compare them with the "
         "query's vectors at search time, trading storage for accuracy. ") * 12  # well over 384 tokens

DOCUMENTS = [
    "Late interaction compares query and document representations after separate encoding.",
    "BreadBowl uses routing vectors to retrieve documents and read their stored values.",
    "Apples and oranges are different fruits.",
    "Résumé: multilingual text can include café and 日本語.",
    "Octopuses have three hearts: two pump blood through the gills and one through the body.",
    "The Pacific is Earth's largest ocean.",
    "Vitamin D deficiency can cause fatigue, bone pain, muscle weakness and mood changes in adults.",
    "Albert Einstein entwickelte die spezielle und die allgemeine Relativitätstheorie.",
    "東京タワーの高さは333メートルです。1958年に完成しました。",
    "sorted(items, key=lambda d: d['name']) returns a new list ordered by the 'name' field.",
    "Canberra was chosen as Australia's capital in 1908 as a compromise between Sydney and Melbourne.",
    "The Berlin Wall fell on 9 November 1989, after East Germany opened its border crossings.",
    "Nuclear power produces long-lived radioactive waste and carries a risk of severe accidents.",
    "Neapolitan pizza is protected as a traditional speciality; L'Antica Pizzeria da Michele is famous. 🍕",
    "Water boils at about 70 °C on the summit of Everest because air pressure is roughly a third of sea level.",
    "x",
    _LONG,
    "Rising US interest rates tend to strengthen the dollar and pull capital out of emerging markets.",
    "In the light-dependent reactions, chlorophyll absorbs light and splits water, releasing oxygen.",
    "Thomas Jefferson drafted the Declaration of Independence in 1776.\n\nIt was adopted on July 4.",
]

PROBES = ["This passage contains the literal string <jepa_mask> in the middle of normal text."]
BATCH_SIZE = 8
# (slot hits, shortlist, rerank): the paper's and the default widths, and a narrow one that
# leaves documents out of the shortlist and underfills the rerank list.
PIPELINES = [(128, 1024, 100), (512, 1024, 100), (16, 6, 4)]


def research_pipeline(q, d, loss, slot_hits, shortlist, rerank):
    """The research evaluator's search (tools/evaluate_beir_pipeline.py, one shard, flat FAISS)."""
    import faiss
    from qkv.fresh_dev_metrics import coarse_candidates, merge_slot_hits
    from qkv.losses_impl.scoring import score_qk, score_qkv

    count, slots, dimension = d["routing"].shape
    flat = d["routing"].numpy().reshape(-1, dimension).copy()
    faiss.normalize_L2(flat)
    index = faiss.IndexFlatIP(dimension)
    index.add(flat)
    queries = q["routing"].numpy().reshape(-1, dimension).copy()
    faiss.normalize_L2(queries)
    n = len(q["routing"])
    scores, hits = index.search(queries, slot_hits)
    scores, hits = merge_slot_hits([scores.reshape(n, slots, slot_hits)], [hits.reshape(n, slots, slot_hits)], slot_hits)
    ranked = []
    with torch.inference_mode():
        for j in range(n):
            candidates = coarse_candidates(scores[j], hits[j], slots, shortlist)
            routing = score_qk(q["routing"][j:j + 1], d["routing"][candidates], tau_k=loss["tau_k"],
                               mode=loss["qk_mode"])[0]
            order = sorted(range(len(candidates)), key=lambda i: (-float(routing[i]), candidates[i]))[:rerank]
            chosen = [candidates[i] for i in order]
            value = score_qkv(q["routing"][j:j + 1], d["routing"][chosen], q["value"][j:j + 1], d["value"][chosen],
                              tau_v=loss["tau_v"])[0]
            r = routing[order]
            final = (1 - loss["value_mix"]) * r + loss["value_mix"] * value
            rows = sorted(zip(chosen, final.tolist(), r.tolist(), value.tolist()), key=lambda row: (-row[1], row[0]))
            ranked.append({"shortlist": candidates, "results": rows})
    return ranked


# ---- reference phase (research environment) ------------------------------------------------
def run_reference(args) -> None:
    sys.path.insert(0, str(Path(args.research_src).resolve()))
    os.environ["QKV_BASE_MODEL_PATH"] = str(Path(args.base_model).resolve())
    torch.set_num_threads(args.threads)
    from qkv.data import format_document, format_query
    from qkv.inference_artifacts import load_inference_artifacts
    from qkv.losses_impl.scoring import score_final
    from qkv.model import QKVOutput
    import transformers

    artifacts = load_inference_artifacts(model_path=args.checkpoint, model_config_path=args.model_config)
    model, tokenizer, data = artifacts.model, artifacts.tokenizer, artifacts.data_config
    loaded_dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters()})
    if args.dtype != "float32":
        model.to({"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype])
    loss = artifacts.train_config["loss"]
    checkpoint = artifacts.checkpoint
    info = {"step": checkpoint.get("step"),
            "embedded_model_config": checkpoint.get("model_config"),
            "loaded_model_config": artifacts.model_config,
            "loaded_parameter_dtypes": loaded_dtypes,
            "compute_dtype": args.dtype, "tokenizer_length": len(tokenizer),
            "padding_side": tokenizer.padding_side, "backbone_class": type(model.backbone).__name__,
            "torch": torch.__version__.split("+")[0], "transformers": transformers.__version__,
            "loss": {k: loss[k] for k in ("qk_mode", "tau_k", "tau_v", "value_mix")},
            "preprocessing": {k: data[k] for k in ("query_format", "document_format", "default_instruction",
                                                   "max_query_length", "max_document_length")}}
    del checkpoint
    artifacts = None

    def encode(texts, limit, batch_size=BATCH_SIZE):
        routing, value, attention, input_ids = [], [], [], []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=limit, return_tensors="pt")
            with torch.inference_mode():
                out = model(**batch)
            routing.append(out.routing.float())
            value.append(out.value.float())
            mask = batch["attention_mask"].bool()
            for b in range(len(mask)):
                attention.append(out.attention[b][..., mask[b]].float())
                input_ids.append(batch["input_ids"][b][mask[b]])
        return {"routing": torch.cat(routing), "value": torch.cat(value), "attention": attention,
                "input_ids": input_ids}

    queries = [format_query({"query": q} if i is None else {"query": q, "instruction": i}, data) for q, i in QUERIES]
    documents = [format_document(text, data) for text in DOCUMENTS]
    q = encode(queries, data["max_query_length"])
    d = encode(documents, data["max_document_length"])
    with torch.inference_mode():
        final, routing, value = score_final(
            QKVOutput(q["routing"], q["value"], torch.empty(0)), QKVOutput(d["routing"], d["value"], torch.empty(0)),
            alpha=1 - loss["value_mix"], beta=loss["value_mix"], tau_k=loss["tau_k"], qk_mode=loss["qk_mode"],
            tau_v=loss["tau_v"])
    probes = [encode([format_document(text, data)], data["max_document_length"]) for text in PROBES]
    pipelines = {f"{hits}/{shortlist}/{rerank}": research_pipeline(q, d, loss, hits, shortlist, rerank)
                 for hits, shortlist, rerank in PIPELINES}
    # Every text on its own (no padding): what the toolkit's default batch_size=1 computes.
    alone = {"queries": encode(queries, data["max_query_length"], 1),
             "documents": encode(documents, data["max_document_length"], 1)}
    with torch.inference_mode():
        alone["final"] = score_final(
            QKVOutput(alone["queries"]["routing"], alone["queries"]["value"], torch.empty(0)),
            QKVOutput(alone["documents"]["routing"], alone["documents"]["value"], torch.empty(0)),
            alpha=1 - loss["value_mix"], beta=loss["value_mix"], tau_k=loss["tau_k"], qk_mode=loss["qk_mode"],
            tau_v=loss["tau_v"])[0]
    torch.save({"info": info, "queries": q, "documents": d, "final": final, "routing": routing, "value": value,
                "probes": probes, "pipelines": pipelines, "alone": alone}, args.output)
    print(f"[reference] saved {len(queries)} queries, {len(documents)} documents ({args.dtype})", flush=True)


# ---- compare phase (toolkit environment) ---------------------------------------------------
def _max(x):
    return float(x.abs().max()) if x.numel() else 0.0


def run_compare(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    torch.set_num_threads(args.threads)
    import torch.nn.functional as F
    import transformers
    from breadbowl_embed import BreadBowl, Index, SearchProfile, __version__
    from breadbowl_embed.retrieval import PROFILES

    ref = torch.load(args.reference, map_location="cpu", weights_only=False)
    model = BreadBowl.from_pretrained(args.release, dtype=args.dtype)
    texts = [q for q, _ in QUERIES]
    instructions = [i if i is not None else model.config["preprocessing"]["default_instruction"] for _, i in QUERIES]
    q = model.encode_queries(texts, instructions=instructions, batch_size=BATCH_SIZE,
                             return_attention=True, return_tokens=True)
    d = model.encode_documents(DOCUMENTS, batch_size=BATCH_SIZE, return_attention=True, return_tokens=True)
    scores = model.score(q, d)
    report = {"scope": "CPU parity of the release against the research loader on fixed texts; "
                       "not a benchmark rerun",
              "compute_dtype": args.dtype, "queries": len(texts), "documents": len(DOCUMENTS),
              "batch_size": BATCH_SIZE, "reference": ref["info"],
              "release": {"representation_id": model.representation_id,
                          "provenance": model.config.get("provenance", {}),
                          "breadbowl_embed": __version__, "torch": torch.__version__.split("+")[0],
                          "transformers": transformers.__version__}}
    report["reference"].pop("embedded_model_config", None)
    loaded_config = report["reference"].pop("loaded_model_config", {})
    report["reference"]["loaded_backbone_torch_dtype"] = loaded_config.get("backbone", {}).get("torch_dtype")
    report["token_counts"] = {"queries": [len(i) for i in q.input_ids], "documents": [len(i) for i in d.input_ids],
                              "queries_truncated": sum(q.truncated), "documents_truncated": sum(d.truncated)}
    mismatched = [f"query {k}" for k, (a, b) in enumerate(zip(q.input_ids, ref["queries"]["input_ids"])) if not torch.equal(a, b)]
    mismatched += [f"document {k}" for k, (a, b) in enumerate(zip(d.input_ids, ref["documents"]["input_ids"])) if not torch.equal(a, b)]
    differences = {
        "query_routing_normalised": _max(q.routing - F.normalize(ref["queries"]["routing"], dim=-1)),
        "document_routing_normalised": _max(d.routing - F.normalize(ref["documents"]["routing"], dim=-1)),
        "query_value": _max(q.value - ref["queries"]["value"]),
        "document_value": _max(d.value - ref["documents"]["value"]),
        "query_attention": max(_max(a - b) for a, b in zip(q.attention, ref["queries"]["attention"])),
        "document_attention": max(_max(a - b) for a, b in zip(d.attention, ref["documents"]["attention"])),
        "final_scores": _max(scores.combined - ref["final"]),
        "routing_scores_R": _max(scores.routing - ref["routing"]),
        "value_scores_V": _max(scores.value - ref["value"]),
    }
    # Raw (unnormalised) routing straight from the network, same tokenisation and batching.
    raw = []
    with torch.inference_mode():
        for start in range(0, len(DOCUMENTS), BATCH_SIZE):
            formatted = [text for text, _ in model.format_texts(DOCUMENTS[start:start + BATCH_SIZE], "document")]
            batch = model.tokenizer(formatted, padding=True, truncation=True,
                                    max_length=model.config["preprocessing"]["max_document_length"], return_tensors="pt")
            raw.append(model.network(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).routing.float())
    differences["document_routing_raw"] = _max(torch.cat(raw) - ref["documents"]["routing"])
    explanations = model.explain(q, d)
    identity = {"final_minus_mix": 0.0, "mean_contribution_minus_R": 0.0, "mean_value_cosine_minus_V": 0.0,
                "explain_final_minus_score": 0.0}
    mix = model.scoring.value_mix
    for i, row in enumerate(explanations):
        for j, e in enumerate(row):
            identity["final_minus_mix"] = max(identity["final_minus_mix"],
                                              abs(e.score - ((1 - mix) * e.routing_score + mix * e.value_score)))
            identity["mean_contribution_minus_R"] = max(identity["mean_contribution_minus_R"],
                                                        abs(float(e.routing_contribution.mean()) - e.routing_score))
            identity["mean_value_cosine_minus_V"] = max(identity["mean_value_cosine_minus_V"],
                                                        abs(float(e.value_slot_cosine.mean()) - e.value_score))
            identity["explain_final_minus_score"] = max(identity["explain_final_minus_score"],
                                                        abs(e.score - float(scores.combined[i, j])))
    reference_order = ref["final"].argsort(dim=1, descending=True)
    rankings_equal = torch.equal(scores.combined.argsort(dim=1, descending=True), reference_order)
    ids = [f"d{j:02d}" for j in range(len(DOCUMENTS))]
    index = Index(d.without_tokens(), ids, scoring=model.scoring)
    expected_top = [[ids[j] for j in row[:10]] for row in reference_order.tolist()]
    exact = [[hit.id for hit in hits] for hits in index.search(q.without_tokens(), top_k=10, profile="exact")]
    search = {"exact_top10_equal_to_research_bruteforce": exact == expected_top}
    pipeline_differences = {}
    for key, reference_rows in ref.get("pipelines", {}).items():
        hits, shortlist, rerank = (int(x) for x in key.split("/"))
        profile = SearchProfile("custom", "slot-shortlist", hits, shortlist, rerank)
        named = [name for name, p in PROFILES.items() if (p.slot_hits, p.shortlist, p.rerank) == (hits, shortlist, rerank)]
        label = f"slot_shortlist_{key}" + (f"_{named[0]}" if named else "")
        got = index.search(q.without_tokens(), top_k=rerank, profile=profile)
        same_ids = all([hit.document_index for hit in mine] == [row[0] for row in theirs["results"]]
                       for mine, theirs in zip(got, reference_rows))
        worst = 0.0
        for mine, theirs in zip(got, reference_rows):
            for hit, row in zip(mine, theirs["results"]):
                worst = max(worst, abs(hit.score - row[1]), abs(hit.routing_score - row[2]), abs(hit.value_score - row[3]))
        search[label + "_equal_to_research_pipeline"] = same_ids
        pipeline_differences[label] = {
            "max_abs_score_difference": worst,
            "mean_results_per_query": sum(len(x) for x in got) / len(got),
            "mean_shortlist_per_query": sum(len(t["shortlist"]) for t in reference_rows) / len(reference_rows)}
    differences_search = pipeline_differences
    probes = []
    for text, probe in zip(PROBES, ref["probes"]):
        ours = model.encode_documents([text], return_tokens=True)
        probes.append({"research_tokens": len(probe["input_ids"][0]), "release_tokens": len(ours.input_ids[0]),
                       "same_input_ids": torch.equal(ours.input_ids[0], probe["input_ids"][0]),
                       "routing_normalised_max_diff": _max(ours.routing - F.normalize(probe["routing"], dim=-1)),
                       "value_max_diff": _max(ours.value - probe["value"])})
    # Public helpers on the same data: routing-only scoring and tokenisation without the model.
    routing_only = model.score_routing(q.without_tokens(), d.routing)
    tokenized = model.tokenize(texts, "query", instructions=instructions) + model.tokenize(DOCUMENTS, "document")
    helpers = {"score_routing_bit_identical": torch.equal(routing_only, scores.routing),
               "tokenize_matches_encode_and_research": all(
                   torch.equal(t.input_ids, e) and torch.equal(t.input_ids, r) and t.truncated == f
                   for t, e, r, f in zip(tokenized, q.input_ids + d.input_ids,
                                         ref["queries"]["input_ids"] + ref["documents"]["input_ids"],
                                         q.truncated + d.truncated)),
               "display_strings_join_to_decoded_input": all(
                   "".join(t.token_texts) == model.tokenizer.decode(t.input_ids.tolist()) for t in tokenized)}
    # The toolkit default (batch_size=1) against the research model encoding each text alone.
    default = {}
    if "alone" in ref:
        q1 = model.encode_queries(texts, instructions=instructions)
        d1 = model.encode_documents(DOCUMENTS)
        alone = ref["alone"]
        default = {"query_routing_normalised": _max(q1.routing - F.normalize(alone["queries"]["routing"], dim=-1)),
                   "query_value": _max(q1.value - alone["queries"]["value"]),
                   "document_routing_normalised": _max(d1.routing - F.normalize(alone["documents"]["routing"], dim=-1)),
                   "document_value": _max(d1.value - alone["documents"]["value"]),
                   "final_scores": _max(model.score(q1, d1).combined - alone["final"])}
        # How much batching changes the research model's own scores (batch 8 vs alone).
        report["research_batch8_vs_alone_max_final_score_difference"] = _max(ref["final"] - alone["final"])
    tolerance = args.tolerance
    gate = {"input_ids_identical": not mismatched,
            "tensors_within_tolerance": all(v <= tolerance for v in differences.values()),
            "default_batch_size_within_tolerance": bool(default) and all(v <= tolerance for v in default.values()),
            "explain_identities_within_tolerance": all(v <= tolerance for v in identity.values()),
            "rankings_identical": rankings_equal,
            "public_helpers_consistent": all(helpers.values()),
            "search_matches_research": all(search.values()) and all(
                v["max_abs_score_difference"] <= tolerance for v in differences_search.values())}
    report.update({"tolerance": tolerance, "max_absolute_differences": differences,
                   "default_batch_size_1_max_absolute_differences": default,
                   "explain_identities_max_abs": identity, "input_id_mismatches": mismatched,
                   "rankings_identical": rankings_equal, "public_helpers": helpers,
                   "search": search, "search_pipelines": differences_search,
                   "sentinel_probe_not_gated": probes, "gate": gate, "passed": all(gate.values())})
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "gate": gate, "max_absolute_differences": differences,
                      "default_batch_size_1_max_absolute_differences": default,
                      "research_batch8_vs_alone": report.get("research_batch8_vs_alone_max_final_score_difference"),
                      "explain_identities_max_abs": identity, "public_helpers": helpers, "search": search,
                      "search_pipelines": differences_search}, indent=2))
    return 0 if report["passed"] or args.no_gate else 1


def run_all(args) -> int:
    with tempfile.TemporaryDirectory() as temporary:
        reference = Path(temporary) / "reference.pt"
        command = [args.research_python or sys.executable, __file__, "reference", "--research-src", args.research_src,
                   "--checkpoint", args.checkpoint, "--base-model", args.base_model, "--output", str(reference),
                   "--dtype", args.dtype, "--threads", str(args.threads)]
        if args.model_config:
            command += ["--model-config", args.model_config]
        subprocess.run(command, check=True)
        args.reference = str(reference)
        return run_compare(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reference", "compare", "all"):
        p = commands.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
        p.add_argument("--threads", type=int, default=8)
        if name in ("reference", "all"):
            p.add_argument("--research-src", required=True, help="Research checkout's src directory")
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--base-model", required=True, help="Base model directory (QKV_BASE_MODEL_PATH)")
            p.add_argument("--model-config", help="Override the checkpoint's embedded model config (YAML)")
        if name in ("compare", "all"):
            p.add_argument("--release", required=True)
            p.add_argument("--tolerance", type=float, default=1e-5)
            p.add_argument("--no-gate", action="store_true", help="Report only; exit 0 on differences")
        if name == "compare":
            p.add_argument("--reference", required=True)
        if name == "all":
            p.add_argument("--research-python", help="Python of the research environment (default: this one)")
    args = parser.parse_args()
    if args.command == "reference":
        run_reference(args)
        return 0
    return run_compare(args) if args.command == "compare" else run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
