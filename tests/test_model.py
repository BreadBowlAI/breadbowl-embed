import json

import pytest
import torch
from breadbowl_embed import BreadBowl
from breadbowl_embed.export import from_checkpoint


def test_encode_and_rerank_cached_documents(tiny_model):
    docs = ["cats land", "fish water", "dogs land"]
    encoded = tiny_model.encode_documents(docs, batch_size=2)
    torch.testing.assert_close(encoded.routing.norm(dim=-1), torch.ones(3, 3))
    assert tiny_model.rerank("fish", encoded) == tiny_model.rerank("fish", docs, batch_size=2)
    assert tiny_model.encode_queries([]).routing.shape == (0, 3, 8)
    with pytest.raises(TypeError):
        tiny_model.encode_queries("one string")


def test_model_export_roundtrip_offline_and_dtype_identity(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release", local_files_only=True)
    for role in ("queries", "documents"):
        original = getattr(tiny_model, f"encode_{role}")(["cats land", "fish water"])
        restored = getattr(loaded, f"encode_{role}")(["cats land", "fish water"])
        torch.testing.assert_close(original.routing, restored.routing, atol=0, rtol=0)
        torch.testing.assert_close(original.value, restored.value, atol=0, rtol=0)
    different = BreadBowl.from_pretrained(tmp_path / "release", dtype="float32")
    assert loaded.representation_id != different.representation_id
    loaded.save_pretrained(tmp_path / "resaved")
    reloaded = BreadBowl.from_pretrained(tmp_path / "resaved")
    assert loaded.representation_id == reloaded.representation_id


def test_dtype_spellings_share_identities(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    native = BreadBowl.from_pretrained(tmp_path / "release", dtype=None)
    assert native.representation_id == BreadBowl.from_pretrained(tmp_path / "release", dtype="native").representation_id
    as_torch = BreadBowl.from_pretrained(tmp_path / "release", dtype=torch.bfloat16)
    as_text = BreadBowl.from_pretrained(tmp_path / "release", dtype="bf16")
    assert as_torch.representation_id == as_text.representation_id
    assert {p.dtype for p in as_torch.network.parameters()} == {torch.bfloat16}
    encoded = as_torch.encode_documents(["cats land"])
    assert encoded.routing.dtype == torch.float32  # encoders always return float32
    with pytest.raises(ValueError):
        BreadBowl.from_pretrained(tmp_path / "release", dtype="int8")


def test_export_strips_training_state_and_keeps_scores(tiny_model, tmp_path):
    base = tmp_path / "base"
    tiny_model.network.backbone.config.save_pretrained(base)
    tiny_model.tokenizer.save_pretrained(base)
    state = {"model": tiny_model.network.state_dict(), "model_config": tiny_model.config["architecture"],
             "data_config": tiny_model.config["preprocessing"],
             "train_config": {"stage": "weak", "use_lora": False, "loss": tiny_model.config["scoring"]},
             "optimizer": {"do_not_publish": "private"}, "step": 4000}
    torch.save(state, tmp_path / "model.pt")
    converted = from_checkpoint(tmp_path / "model.pt", base_model_path=str(base), local_files_only=True)
    converted.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release")
    a = tiny_model.rerank("fish", ["cats", "fish"])
    b = loaded.rerank("fish", ["cats", "fish"])
    assert a == b
    assert loaded.config["provenance"]["step"] == 4000
    assert "optimizer" not in (tmp_path / "release/config.json").read_text()


def test_incomplete_release_and_modified_config_fail(tmp_path, tiny_model):
    with pytest.raises(ValueError, match="no BreadBowl weight release"):
        BreadBowl.from_pretrained(tmp_path)
    tiny_model.save_pretrained(tmp_path / "release")
    (tmp_path / "release/config.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        BreadBowl.from_pretrained(tmp_path / "release")


def test_hub_style_symlink_snapshot(tiny_model, tmp_path):
    release = tmp_path / "release"
    tiny_model.save_pretrained(release)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for path in release.rglob("*"):
        target = snapshot / path.relative_to(release)
        if path.is_dir():
            target.mkdir()
        else:
            target.symlink_to(path)
    loaded = BreadBowl.from_pretrained(snapshot, local_files_only=True)
    assert len(loaded.encode_queries(["fish"])) == 1


def test_attention_tokens_and_offsets(tiny_model):
    texts = ["cats land", "fish water dogs land cats"]
    encoded = tiny_model.encode_queries(texts, batch_size=2, return_attention=True, return_tokens=True)
    plain = tiny_model.encode_queries(texts, batch_size=2)
    torch.testing.assert_close(encoded.routing, plain.routing, atol=0, rtol=0)  # extras change nothing
    for i, text in enumerate(texts):
        tokens, offsets, attention = encoded.tokens[i], encoded.offsets[i], encoded.attention[i]
        assert attention.shape == (2, 3, len(tokens)) == (2, 3, len(encoded.input_ids[i]))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3))  # padding stripped, rows complete
        # "search Query :" come from the instruction/template; every word of the query maps back.
        assert tokens[:3] == ("search", "Query", ":") and offsets[:3] == (None, None, None)
        assert [text[slice(*span)] for span in offsets[3:]] == text.split()
        assert encoded.truncated[i] is False
    # A batch member's attention equals encoding it alone (padding does not leak in).
    alone = tiny_model.encode_queries(texts[:1], return_attention=True)
    torch.testing.assert_close(alone.attention[0], encoded.attention[0], atol=1e-6, rtol=1e-5)
    selected = encoded.select([1])
    assert selected.tokens == encoded.tokens[1:] and len(selected.attention) == 1


def test_truncation_flag_and_document_offsets(tiny_model):
    long_text = " ".join(["cats"] * 40)
    encoded = tiny_model.encode_documents([long_text, "fish"], return_tokens=True, return_attention=True)
    assert encoded.truncated == (True, False)
    assert len(encoded.input_ids[0]) == 16 == encoded.attention[0].shape[-1]
    assert encoded.offsets[1] == ((0, 4),)


def test_instructions(tiny_model):
    default = tiny_model.encode_queries(["fish"])
    same = tiny_model.encode_queries(["fish"], instruction="search")
    torch.testing.assert_close(default.routing, same.routing, atol=0, rtol=0)
    other = tiny_model.encode_queries(["fish"], instruction="cats dogs")
    assert not torch.equal(default.routing, other.routing)
    per_query = tiny_model.encode_queries(["fish", "fish"], instructions=["search", "cats dogs"])
    torch.testing.assert_close(per_query.routing[1:], other.routing, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        tiny_model.encode_queries(["fish"], instruction="a", instructions=["b"])


def test_empty_text_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="no tokens"):
        tiny_model.encode_documents(["fish", ""])


def test_info(tiny_model):
    info = tiny_model.info()
    assert info["architecture"]["slots"] == 3 and info["parameter_dtypes"] == ["float32"]
    assert info["release"]["architecture"] == "qkv-slot-encoder" and info["default_batch_size"] == 1


def test_loaded_info_for_a_schema1_release(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")  # the fixture is a schema-1 config
    manifest = json.loads((tmp_path / "release/manifest.json").read_text())
    loaded = BreadBowl.from_pretrained(tmp_path / "release").info()
    assert loaded["schema_version"] == 1 and loaded["release"]["license"] is None
    assert loaded["release_id"] == manifest["representation_id"]
    assert loaded["release"]["weights_sha256"] == manifest["files"]["model.safetensors"]
    assert loaded["compute_dtype"] == "native" and loaded["release"]["huggingface"]["loaded_revision"] is None


def test_default_batch_size_makes_each_text_independent(tiny_model):
    texts = ["cats", "fish water dogs land cats fish water", "land"]
    together = tiny_model.encode_documents(texts)
    for i, text in enumerate(texts):
        alone = tiny_model.encode_documents([text])
        assert torch.equal(alone.routing[0], together.routing[i]) and torch.equal(alone.value[0], together.value[i])
    queries = tiny_model.encode_queries(texts)
    assert torch.equal(queries.value[1], tiny_model.encode_queries(texts[1:2]).value[0])


def test_reader_matches_safetensors_and_casts_like_module_to(tmp_path):
    from safetensors.torch import load_file, save_file
    from breadbowl_embed._weights import read_safetensors
    torch.manual_seed(2)
    tensors = {"a": torch.randn(3, 5), "b": torch.randn(7).to(torch.bfloat16), "c": torch.arange(6).reshape(2, 3),
               "d": torch.tensor(1.5), "e": torch.zeros(0, 4), "f": torch.tensor([True, False])}
    save_file(tensors, str(tmp_path / "w.safetensors"))
    native = read_safetensors(tmp_path / "w.safetensors")
    reference = load_file(str(tmp_path / "w.safetensors"))
    assert set(native) == set(reference)
    for name in reference:
        assert native[name].dtype == reference[name].dtype and torch.equal(native[name], reference[name])
    cast = read_safetensors(tmp_path / "w.safetensors", torch.float16)
    for name, tensor in reference.items():
        expected = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
        assert cast[name].dtype == expected.dtype and torch.equal(cast[name], expected)


def test_cast_while_loading_equals_cast_after_loading(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release", dtype="bfloat16")
    reference = BreadBowl.from_pretrained(tmp_path / "release")
    reference.network.to(torch.bfloat16)  # the previous loader's method
    for (name, a), (_, b) in zip(loaded.network.state_dict().items(), reference.network.state_dict().items()):
        assert a.dtype == b.dtype and torch.equal(a, b), name
    for (name, a), (_, b) in zip(loaded.network.named_buffers(), reference.network.named_buffers()):
        assert a.dtype == b.dtype and torch.equal(a, b), name
