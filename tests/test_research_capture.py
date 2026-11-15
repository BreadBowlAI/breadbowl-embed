"""breadbowl_embed.research: traced forward pass, token views and attention maps (tiny random model)."""
import math

import pytest
import torch

from breadbowl_embed import Embeddings
from breadbowl_embed.research import AttentionMap, attention_maps, run_heads, token_views, trace


def test_trace_reproduces_the_encoders_bit_for_bit(tiny_model):
    texts = ["cats land", "fish water dogs land cats", " ".join(["dogs"] * 30)]
    for kind, encode in (("query", tiny_model.encode_queries), ("document", tiny_model.encode_documents)):
        for batch_size in (1, 2):
            traced = trace(tiny_model, texts, kind, batch_size=batch_size, capture="all", layers=[0, -1])
            encoded = encode(texts, batch_size=batch_size, return_attention=True, return_tokens=True)
            rebuilt = traced.encoding()
            assert torch.equal(rebuilt.routing, encoded.routing) and torch.equal(rebuilt.value, encoded.value)
            assert rebuilt.representation_id == encoded.representation_id and rebuilt.kind == kind
            for i in range(len(texts)):
                assert torch.equal(rebuilt.attention[i], encoded.attention[i])
                assert torch.equal(rebuilt.input_ids[i], encoded.input_ids[i])
                assert rebuilt.offsets[i] == encoded.offsets[i] and rebuilt.tokens[i] == encoded.tokens[i]
                assert rebuilt.token_texts[i] == encoded.token_texts[i]
                assert rebuilt.truncated[i] == encoded.truncated[i]


def test_trace_tensors_are_consistent(tiny_model):
    [t] = trace(tiny_model, ["fish water dogs land"], "document", capture="all", layers=[0])
    tokens, slots, hidden = len(t.tokens), 3, 16
    assert t.attention.shape == t.attention_logits.shape == (2, slots, tokens)
    # softmax of the recomputed logits reproduces the model's attention
    torch.testing.assert_close(torch.softmax(t.attention_logits, dim=-1), t.attention, atol=0, rtol=0)
    torch.testing.assert_close(t.attention.sum(-1), torch.ones(2, slots))
    assert t.hidden_states.shape == (tokens, hidden) and t.layer_states[0].shape == (tokens, hidden)
    assert t.slot_query_projection.shape == (2, slots, 8) and t.key_projection.shape == (2, tokens, 8)
    assert t.slot_readout.shape == (slots, 16) and t.slot_states_pre_norm.shape == t.slot_states.shape == (slots, hidden)
    # routing is the L2-normalised raw routing head output; value = linear head + residual adapter
    torch.testing.assert_close(t.routing, torch.nn.functional.normalize(t.routing_raw, dim=-1), atol=0, rtol=0)
    torch.testing.assert_close(t.value, t.value_linear + t.value_adapter, atol=0, rtol=0)
    # the slot readout is attention @ value projection, heads concatenated
    readout = (t.attention @ t.value_projection).transpose(0, 1).reshape(slots, -1)
    torch.testing.assert_close(readout, t.slot_readout)
    # layer norm of the pre-norm states gives the slot states
    with torch.no_grad():
        torch.testing.assert_close(tiny_model.network.slot_extractor.norm(t.slot_states_pre_norm), t.slot_states)
    assert not t.hidden_states.is_inference()  # usable with autograd, e.g. to train a probe
    assert torch.equal(t.slot_queries, tiny_model.network.slot_extractor.slot_queries.detach())
    assert set(t.summary()) >= {"attention", "hidden_states", "routing_raw", "layer_0"}
    with pytest.raises(ValueError):
        trace(tiny_model, ["fish"], "document", capture=("nonsense",))
    with pytest.raises(ValueError):
        trace(tiny_model, ["fish"], "document", layers=[5])


def test_trace_does_not_leave_hooks_behind(tiny_model):
    before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.network.modules())
    trace(tiny_model, ["fish"], "query", capture="all", layers=[0])
    with pytest.raises(ValueError):
        trace(tiny_model, [""], "document")
    after = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_model.network.modules())
    assert before == after == 0


def test_run_heads_reproduces_the_forward_pass(tiny_model):
    [t] = trace(tiny_model, ["fish water dogs"], "document", capture=("hidden",))
    mask = torch.ones(1, len(t.tokens), dtype=torch.long)
    with torch.inference_mode():
        routing, value, attention = run_heads(tiny_model.network, t.hidden_states[None], mask)
    assert torch.equal(routing[0], t.routing_raw) and torch.equal(value[0], t.value)
    assert torch.equal(attention[0], t.attention)


def test_token_views_segments_words_and_spans(tiny_model):
    [view] = token_views(tiny_model, ["fish water"], "query", instruction="cats dogs")
    assert view.formatted == "cats dogs Query: fish water"
    assert view.segments == ("instruction", "instruction", "template", "template", "text", "text")
    assert view.instruction == "cats dogs" and view.formatted[slice(*view.instruction_span)] == "cats dogs"
    assert view.formatted_offsets[0] == (0, 4) and view.offsets[4] == (0, 4)
    assert [w.text for w in view.words] == ["fish", "water"] and view.token_words == (None, None, None, None, 0, 1)
    assert view.tokens_in_span(0, 3) == [4] and view.find("water") == (5, 10)
    assert view.segment_mask("instruction").tolist() == [True, True, False, False, False, False]
    [doc] = token_views(tiny_model, ["fish, water"], "document")
    assert doc.instruction is None and set(doc.segments) == {"text"}
    assert [w.text for w in doc.words] == ["fish", ",", "water"]
    with pytest.raises(ValueError):
        token_views(tiny_model, ["fish"], "document", instruction="x")


def test_attention_map_statistics(tiny_model):
    [m] = attention_maps(tiny_model, ["fish water dogs"], "query")
    assert isinstance(m, AttentionMap) and m.logits is not None
    assert (m.num_heads, m.num_slots, m.num_tokens) == (2, 3, 6)
    torch.testing.assert_close(m.matrix().sum(-1), torch.ones(3))
    torch.testing.assert_close(m.matrix(), m.weights.mean(0))
    mass = m.segment_mass()
    torch.testing.assert_close(sum(mass.values()), torch.ones(3))
    assert set(m.dominant_segment()) <= {"instruction", "template", "text"}
    assert all(0 <= s <= 2 for s in m.slots_reading("text", 0.0))
    entropy = m.entropy()
    assert bool((entropy >= 0).all()) and bool((entropy <= math.log(6) + 1e-6).all())
    normalized = m.entropy(normalized=True)
    assert bool((normalized <= 1 + 1e-6).all())
    torch.testing.assert_close(m.effective_tokens(), entropy.exp())
    assert m.coverage(1.0).tolist() == [6, 6, 6]  # softmax weights are all positive
    assert bool((m.coverage(0.5) <= m.coverage(0.9)).all())
    assert m.entropy_per_head().shape == (2, 3) and m.head_divergence().shape == (3,)
    top = m.top_tokens(2)
    assert len(top) == 3 and all(len(row) == 2 for row in top)
    assert top[0][0]["weight"] >= top[0][1]["weight"]
    only_text = m.top_tokens(5, segments=("text",))
    assert all(t["segment"] == "text" for row in only_text for t in row) and all(len(r) == 3 for r in only_text)
    words = m.word_matrix()
    assert words.shape == (3, 3)
    torch.testing.assert_close(words.sum(-1), mass["text"])
    torch.testing.assert_close(m.span_mass(0, len("fish water dogs")), mass["text"])
    torch.testing.assert_close(m.phrase_mass("water"), m.matrix()[:, 4])  # search Query : fish water dogs
    assert [w["word"] for w in m.word_weights(0)] == ["fish", "water", "dogs"]
    spans = m.character_spans(1)
    assert [s[:2] for s in spans] == [(0, 4), (5, 10), (11, 15)]
    stats = m.statistics()
    assert stats[0]["coverage_90"] >= 1 and set(stats[0]["segment_mass"]) == {"instruction", "template", "text",
                                                                             "special"}


def test_attention_map_from_an_encoding(tiny_model):
    encoded = tiny_model.encode_documents(["fish water", "cats"], return_attention=True, return_tokens=True)
    m = AttentionMap.from_encoding(tiny_model, encoded, 0, "fish water")
    assert torch.equal(m.weights, encoded.attention[0])
    with pytest.raises(ValueError):
        AttentionMap.from_encoding(tiny_model, encoded, 1, "fish water")
    with pytest.raises(ValueError):
        AttentionMap.from_encoding(tiny_model, tiny_model.encode_documents(["cats"]), 0, "cats")
    assert isinstance(encoded, Embeddings)
