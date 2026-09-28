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
