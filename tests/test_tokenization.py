import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from breadbowl_embed.tokenization import _byte_decoder, display_strings


def byte_level_tokenizer() -> PreTrainedTokenizerFast:
    """One token per byte, as byte-level BPE with no merges: multi-byte characters span tokens."""
    alphabet = sorted(_byte_decoder(), key=_byte_decoder().get)
    core = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def test_byte_level_display_strings_never_split_characters():
    tokenizer = byte_level_tokenizer()
    text = "Résumé 🍕 日本"
    ids = tokenizer(text)["input_ids"]
    shown = display_strings(tokenizer, ids)
    assert len(ids) == len(text.encode("utf-8")) and len(shown) == len(ids)
    assert "".join(shown) == text == tokenizer.decode(ids)
    assert "�" not in "".join(shown)
    pizza = [s for s in shown if "🍕" in s]
    assert pizza == ["🍕"] and shown[shown.index("🍕") - 3:shown.index("🍕")] == ("", "", "")
    pad = tokenizer.convert_tokens_to_ids("<pad>")
    assert display_strings(tokenizer, ids[:2] + [pad]) == ("R", "", "�<pad>")  # cut inside "é", then a special


def test_generic_display_strings_join_to_decode(tiny_model):
    ids = tiny_model.tokenizer("search Query : cats land")["input_ids"]
    shown = display_strings(tiny_model.tokenizer, ids)
    assert "".join(shown) == tiny_model.tokenizer.decode(ids)
    assert len(shown) == len(ids) and all(shown)


def test_tokenize_matches_encoders(tiny_model):
    texts = ["cats land", "fish water dogs land cats", " ".join(["dogs"] * 30)]
    for kind, encode in (("query", tiny_model.encode_queries), ("document", tiny_model.encode_documents)):
        tokenized = tiny_model.tokenize(texts, kind)
        encoded = encode(texts, return_tokens=True, batch_size=3)
        for i, item in enumerate(tokenized):
            assert torch.equal(item.input_ids, encoded.input_ids[i])
            assert item.tokens == encoded.tokens[i] and item.offsets == encoded.offsets[i]
            assert item.token_texts == encoded.token_texts[i] and item.truncated == encoded.truncated[i]
            assert item.token_count == len(item.input_ids)
            assert item.formatted[slice(*item.span)] == texts[i]
        assert [t.truncated for t in tokenized] == [False, False, True]
    long = tiny_model.tokenize(texts, "document")[2]
    assert long.token_count == 16 and long.untruncated_token_count == 30
    query = tiny_model.tokenize(["fish"], "query", instruction="cats dogs")[0]
    assert query.formatted == "cats dogs Query: fish"
    assert query.segments == ("template", "template", "template", "template", "text")
    with pytest.raises(ValueError):
        tiny_model.tokenize(["fish"], "document", instruction="no")
    with pytest.raises(ValueError):
        tiny_model.tokenize(["fish"], "passage")
