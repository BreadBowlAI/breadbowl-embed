"""Compare a release with the original research code and checkpoint.

Run from an installed toolkit environment. The original research source is only
needed for this release-engineering check, never for end-user inference.
"""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys

import torch
from breadbowl_embed import BreadBowl
from breadbowl_embed.config import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--reference-source", required=True, help="Research checkout's src directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(args.reference_source).resolve()))
    original_model = importlib.import_module("qkv.model")
    original_scoring = importlib.import_module("qkv.losses_impl.scoring")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    root = Path(args.release)
    reference, tokenizer = original_model.QKVRetriever.from_config(
        checkpoint["model_config"], train_config=checkpoint["train_config"], backbone_init="config",
        backbone_source_override=str(root / "backbone"), tokenizer_source_override=str(root / "tokenizer"),
        local_files_only_override=True, enable_gradient_checkpointing=False)
    reference.load_state_dict(checkpoint["model"], strict=True, assign=True)
    reference.eval()
    queries = ["What is late interaction in retrieval?", "Can cached document representations be reused?"]
    documents = ["Late interaction compares query and document representations after separate encoding.",
                 "BreadBowl uses routing vectors to retrieve documents and read their stored values.",
                 "Apples and oranges are different fruits.", "Résumé: multilingual text can include café and 日本語."]
    prep = checkpoint["data_config"]
    def encode(network, texts, role):
        formatted = [prep["query_format"].format(query=t, instruction=prep["default_instruction"]) if role == "query"
                     else prep["document_format"].format(text=t) for t in texts]
        batch = tokenizer(formatted, padding=True, truncation=True,
                          max_length=prep["max_query_length" if role == "query" else "max_document_length"],
                          return_tensors="pt")
        with torch.inference_mode():
            out = network(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return out
    ref_q, ref_d = encode(reference, queries, "query"), encode(reference, documents, "document")
    loss = checkpoint["train_config"]["loss"]
    with torch.inference_mode():
        ref_scores = original_scoring.score_final(ref_q, ref_d, alpha=1-loss["value_mix"], beta=loss["value_mix"],
                       tau_k=loss["tau_k"], tau_v=loss["tau_v"], qk_mode=loss["qk_mode"])
    del reference, checkpoint
    gc.collect()
    public = BreadBowl.from_pretrained(root, local_files_only=True)
    q, d = public.encode_queries(queries), public.encode_documents(documents)
    actual = public.score(q, d)
    differences = {}
    for name, left, right in [
        ("query_routing", q.routing, torch.nn.functional.normalize(ref_q.routing.float(), dim=-1)),
        ("document_routing", d.routing, torch.nn.functional.normalize(ref_d.routing.float(), dim=-1)),
        ("query_values", q.value, ref_q.value.float()), ("document_values", d.value, ref_d.value.float()),
        ("combined_scores", actual.combined, ref_scores[0]), ("routing_scores", actual.routing, ref_scores[1]),
        ("value_scores", actual.value, ref_scores[2])]:
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        differences[name] = float((left-right).abs().max())
    assert torch.equal(actual.combined.argsort(descending=True), ref_scores[0].argsort(descending=True))
    ids = [str(i) for i in range(len(documents))]
    results = public.index(d, ids=ids).search(q, top_k=2, candidate_k=len(documents))
    assert [[row.id for row in rows] for rows in results] == [[ids[j] for j in row[:2]] for row in ref_scores[0].argsort(descending=True).tolist()]
    report = {"verified": True, "scope": "CPU native-dtype smoke parity; not a BEIR rerun or GPU validation",
              "queries": len(queries), "documents": len(documents), "max_absolute_differences": differences,
              "checkpoint_sha256": sha256_file(Path(args.checkpoint)), "representation_id": public.representation_id,
              "rankings": [[r.to_dict() for r in rows] for rows in results]}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
