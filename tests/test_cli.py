import json

import pytest

from breadbowl_embed import Embeddings, Index
from breadbowl_embed.cli import main


def run(capsys, *args) -> list[dict]:
    assert main([str(a) for a in args]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


@pytest.fixture
def release(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    return tmp_path / "release"


def test_index_search_explain_info_encode(capsys, release, tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(row) for row in [
        {"id": "a", "text": "cats land", "metadata": {"kind": "pets"}},
        {"id": "b", "text": "fish water"},
        {"id": "c", "text": "dogs land"}]) + "\n")
    [built] = run(capsys, "index", "--model", release, "--input", docs, "--output", tmp_path / "index",
                  "--storage", "int8", "--chunk-size", 2, "--profile", "paper")
    assert built["indexed"] == 3 and built["shards"] == 2 and built["storage"] == "int8"
    assert built["default_profile"]["name"] == "paper"

    [answer] = run(capsys, "search", "--model", release, "--index", tmp_path / "index", "--query", "fish",
                   "--top-k", 2, "--explain")
    assert answer["profile"]["name"] == "paper" and len(answer["results"]) == 2
    assert {"slot_matches", "slot_hits"} <= set(answer["results"][0]["explanation"])
    expected = Index.load(tmp_path / "index").search(_encode_query(release, "fish"), top_k=2)[0]
    assert [r["id"] for r in answer["results"]] == [r.id for r in expected]

    [exact] = run(capsys, "search", "--model", release, "--index", tmp_path / "index", "--query", "fish",
                  "--profile", "exact", "--candidate-k", 3, "--top-k", 3)
    assert exact["profile"] == {"name": "custom", "method": "exact-routing", "slot_hits": None,
                                "shortlist": None, "rerank": 3}

    [explained] = run(capsys, "explain", "--model", release, "--query", "fish", "--document", "fish water cats")
    assert len(explained["slot_similarity"]) == 3 and len(explained["query_slots"]) == 3
    tokens = explained["document_slots"][0]["tokens"]
    assert all(t["text"] == "fish water cats"[slice(*t["offsets"])] for t in tokens)
    assert explained["score"] == pytest.approx(0.75 * explained["routing_score"] + 0.25 * explained["value_score"])

    [info] = run(capsys, "info", release)
    assert info["kind"] == "model" and info["preprocessing"]["max_document_length"] == 16
    [index_info] = run(capsys, "info", tmp_path / "index")
    assert index_info["kind"] == "index" and index_info["documents"] == 3

    queries = tmp_path / "queries.txt"
    queries.write_text("fish\ncats land\n")
    [encoded] = run(capsys, "encode", "--model", release, "--kind", "query", "--input", queries,
                    "--output", tmp_path / "encoded")
    assert encoded["encoded"] == 2
    assert len(Embeddings.load(tmp_path / "encoded")) == 2


def _encode_query(release, text):
    from breadbowl_embed import BreadBowl
    return BreadBowl.from_pretrained(release).encode_queries([text])


def test_tokenize_and_release_commands(capsys, tiny_qwen, tmp_path):
    from breadbowl_embed.export import from_checkpoint
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True)
    model.save_pretrained(tmp_path / "release")
    before = json.loads((tmp_path / "release/manifest.json").read_text())
    [result] = run(capsys, "release", tmp_path / "release", "--repository", "example/tiny", "--revision", "v1.3")
    assert result["release"]["huggingface"] == {"repository": "example/tiny", "revision": "v1.3"}
    after = json.loads((tmp_path / "release/manifest.json").read_text())
    assert after["representation_id"] == before["representation_id"]
    assert after["files"]["model.safetensors"] == before["files"]["model.safetensors"]
    [info] = run(capsys, "info", tmp_path / "release")
    assert info["schema_version"] == 2 and info["release"]["license"] == "Apache-2.0"
    assert info["release"]["weights_sha256"] == after["files"]["model.safetensors"]
    texts = tmp_path / "texts.txt"
    texts.write_text("fish water\n" + " ".join(["cats"] * 20) + "\n")
    first, second = run(capsys, "tokenize", "--model", tmp_path / "release", "--kind", "document", "--input", texts)
    assert [t["text"] for t in first["tokens"]] == ["fish", " water"] and not first["truncated"]
    assert second["truncated"] and second["token_count"] == 16 and second["untruncated_token_count"] == 20
