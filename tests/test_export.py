"""Export of a Qwen3.5 research checkpoint: text-only backbone, training sentinel removed."""
import json

import pytest
import torch
from safetensors import safe_open

from breadbowl_embed import BreadBowl
from breadbowl_embed.export import from_checkpoint

TEXTS = ["cats land", "fish water dogs land cats fish", "dogs"]


def research_outputs(tiny_qwen, texts, query=False):
    """Forward pass of the research network (full multimodal backbone, sentinel kept)."""
    tokenizer, data = tiny_qwen["tokenizer"], tiny_qwen["data"]
    formatted = [data["query_format"].format(instruction=data["default_instruction"], query=t) if query else t
                 for t in texts]
    batch = tokenizer(formatted, padding=True, truncation=True, return_tensors="pt",
                      max_length=data["max_query_length" if query else "max_document_length"])
    with torch.inference_mode():
        return tiny_qwen["network"](input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]), batch


def assert_matches_research(model, tiny_qwen):
    for query in (False, True):
        reference, batch = research_outputs(tiny_qwen, TEXTS, query)
        encode = model.encode_queries if query else model.encode_documents
        encoded = encode(TEXTS, batch_size=len(TEXTS), return_attention=True, return_tokens=True)
        torch.testing.assert_close(encoded.routing, torch.nn.functional.normalize(reference.routing, dim=-1), atol=0, rtol=0)
        torch.testing.assert_close(encoded.value, reference.value, atol=0, rtol=0)
        mask = batch["attention_mask"].bool()
        for b in range(len(TEXTS)):
            torch.testing.assert_close(encoded.attention[b], reference.attention[b][..., mask[b]], atol=0, rtol=0)
            assert torch.equal(encoded.input_ids[b], batch["input_ids"][b][mask[b]])


def test_text_only_export_reproduces_the_research_network(tiny_qwen, tmp_path):
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True)
    model.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release")
    assert type(loaded.network.backbone).__name__ == "Qwen3_5TextModel"
    assert_matches_research(loaded, tiny_qwen)
    config = json.loads((tmp_path / "release/config.json").read_text())
    removed = config["provenance"]["removed"]
    assert removed["non_text_backbone"]["modules"] == ["visual"] and removed["non_text_backbone"]["parameters"] > 0
    assert removed["jepa_sentinel"] == {"token": "<jepa_mask>", "token_id": 10, "embedding_rows": 1,
                                        "learned_offset": True}
    assert config["inference_setup"] == {"stage": "inference", "use_lora": False}
    assert config["provenance"]["parameter_dtypes"] == ["float32"]
    assert config["provenance"]["base_model"] == {"repo": "tiny/qwen3.5", "revision": "0123abc"}
    assert config["provenance"]["step"] == 5152
    text = (tmp_path / "release/config.json").read_text() + (tmp_path / "release/backbone/config.json").read_text()
    assert "/private" not in text and "QKV_BASE_MODEL_PATH" not in text and "optimizer" not in text
    with safe_open(str(tmp_path / "release/model.safetensors"), "pt") as handle:
        keys = set(handle.keys())
        assert handle.get_tensor("backbone.embed_tokens.weight").shape[0] == 10
    assert not any("visual" in key or "jepa" in key for key in keys)
    assert "<jepa_mask>" not in loaded.tokenizer.get_vocab()


def test_literal_sentinel_text_is_plain_text_after_export(tiny_qwen, tmp_path):
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True)
    encoded = model.encode_documents(["fish <jepa_mask> water"], return_tokens=True)
    assert encoded.tokens[0] == ("fish", "[UNK]", "[UNK]", "[UNK]", "water")
    _, research_batch = research_outputs(tiny_qwen, ["fish <jepa_mask> water"])
    assert research_batch["input_ids"][0].tolist() == [7, 10, 8]  # the research tokenizer reads the sentinel


def test_v12_layout_export_keeps_vision_and_sentinel(tiny_qwen, tmp_path):
    """text_only=False, keep_sentinel=True is the v1.2-preview layout; it still exports and loads."""
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True,
                            text_only=False, keep_sentinel=True)
    model.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release")
    assert type(loaded.network.backbone).__name__ == "Qwen3_5Model"
    assert loaded.config["inference_setup"]["stage"] == "candidate_distill"
    assert_matches_research(loaded, tiny_qwen)
    reference, _ = research_outputs(tiny_qwen, ["fish <jepa_mask> water"])
    encoded = loaded.encode_documents(["fish <jepa_mask> water"])
    torch.testing.assert_close(encoded.value, reference.value, atol=0, rtol=0)


def test_text_only_and_full_exports_agree_on_normal_text(tiny_qwen):
    text_only = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True)
    full = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True,
                           text_only=False, keep_sentinel=True)
    for encode in ("encode_queries", "encode_documents"):
        a, b = getattr(text_only, encode)(TEXTS), getattr(full, encode)(TEXTS)
        torch.testing.assert_close(a.routing, b.routing, atol=0, rtol=0)
        torch.testing.assert_close(a.value, b.value, atol=0, rtol=0)


def test_expected_checksum_is_enforced(tiny_qwen):
    with pytest.raises(ValueError, match="sha256"):
        from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True,
                        expected_sha256="0" * 64)


def test_schema2_release_metadata_keeps_identity(tiny_qwen, tmp_path):
    from breadbowl_embed import set_release_metadata
    from breadbowl_embed.model import release_identity
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True,
                            repository="example/tiny", revision="v0")
    model.save_pretrained(tmp_path / "release")
    config = json.loads((tmp_path / "release/config.json").read_text())
    manifest = json.loads((tmp_path / "release/manifest.json").read_text())
    assert config["schema_version"] == 2
    assert config["release"] == {"architecture": "qkv-slot-encoder", "license": "Apache-2.0",
                                 "weights_sha256": manifest["files"]["model.safetensors"],
                                 "huggingface": {"repository": "example/tiny", "revision": "v0"}}
    # The same release written as schema 1 (no release section) has the same identity.
    schema1 = {k: v for k, v in config.items() if k != "release"} | {"schema_version": 1}
    assert release_identity(schema1, manifest["files"]) == manifest["representation_id"]
    before = BreadBowl.from_pretrained(tmp_path / "release")
    set_release_metadata(tmp_path / "release", repository="example/other", revision="v1")
    after = BreadBowl.from_pretrained(tmp_path / "release")
    assert after.representation_id == before.representation_id
    assert after.info()["release"]["huggingface"] == {"repository": "example/other", "revision": "v1",
                                                      "loaded_revision": None}
    assert after.info()["release"]["license"] == "Apache-2.0"
    after.save_pretrained(tmp_path / "resaved")  # a re-save keeps schema 2 and the identity
    assert BreadBowl.from_pretrained(tmp_path / "resaved").representation_id == before.representation_id


def test_bfloat16_cast_while_loading_matches_module_to(tiny_qwen, tmp_path):
    model = from_checkpoint(tiny_qwen["checkpoint"], base_model_path=str(tiny_qwen["base"]), local_files_only=True)
    model.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release", dtype="bfloat16")
    reference = BreadBowl.from_pretrained(tmp_path / "release")
    reference.network.to(torch.bfloat16)
    buffers = dict(reference.network.named_buffers())
    assert any(b.is_floating_point() for b in buffers.values())  # rotary frequencies
    for name, buffer in loaded.network.named_buffers():
        assert buffer.dtype == buffers[name].dtype and torch.equal(buffer, buffers[name]), name
    a, b = loaded.encode_documents(TEXTS), reference.encode_documents(TEXTS)
    assert torch.equal(a.routing, b.routing) and torch.equal(a.value, b.value)
