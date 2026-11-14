from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from breadbowl_embed import BreadBowl
from breadbowl_embed.architecture import QKVRetriever
from breadbowl_embed.config import FORMAT

WORDS = {"[PAD]": 0, "[UNK]": 1, "search": 2, "Query": 3, ":": 4, "cats": 5, "dogs": 6,
         "fish": 7, "water": 8, "land": 9}


def word_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer_core = Tokenizer(WordLevel(dict(WORDS), unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, pad_token="[PAD]", unk_token="[UNK]")


@pytest.fixture
def tiny_model():
    torch.manual_seed(13)
    tokenizer = word_tokenizer()
    backbone = BertModel(BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=24,
                                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, attn_implementation="eager"))
    network = QKVRetriever(backbone=backbone, num_slots=3, d_attn=16, num_heads=2, d_k=8, d_v=8,
                           dropout=0, use_layer_norm=True, value_adapter_hidden_dim=16)
    config = {"format": FORMAT, "schema_version": 1,
              "architecture": {"backbone": {"pretrained": "backbone", "trust_remote_code": False,
                                              "attn_implementation": "eager"},
                               "slot_extractor": {"num_slots": 3, "d_attn": 16, "num_heads": 2,
                                                  "dropout": 0, "use_layer_norm": True},
                               "heads": {"d_k": 8, "d_v": 8, "value_adapter_hidden_dim": 16}},
              "preprocessing": {"query_format": "{instruction} Query: {query}", "document_format": "{text}",
                                "default_instruction": "search", "max_query_length": 12, "max_document_length": 16},
              "scoring": {"qk_mode": "logmeanexp", "tau_k": 0.02, "tau_v": 0.1, "value_mix": 0.25},
              "inference_setup": {"stage": "weak", "use_lora": False},
              "routing_normalized": True, "value_normalized": False}
    return BreadBowl(network, tokenizer, config, representation_id="tiny-test")


def build_tiny_qwen_checkpoint(root: Path) -> dict:
    """A research-style training checkpoint around a tiny multimodal Qwen3.5 backbone.

    Mirrors the v1.3 checkpoint's structure: Qwen3_5Model with a vision tower, fp32 weights, a
    `<jepa_mask>` sentinel added by candidate-distill training (one extra embedding row after the
    tokenizer's vocabulary, shrinking the config's padded vocabulary) and a learned sentinel offset.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    torch.manual_seed(7)
    base = root / "base"
    text = dict(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                rms_norm_eps=1e-6, tie_word_embeddings=True, attn_output_gate=True,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True},
                pad_token_id=0, eos_token_id=1, bos_token_id=None)
    vision = dict(depth=1, hidden_size=16, num_heads=2, intermediate_size=32, out_hidden_size=32, patch_size=2,
                  temporal_patch_size=1, spatial_merge_size=1, num_position_embeddings=16, in_channels=3,
                  deepstack_visual_indexes=[])
    Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True).save_pretrained(base)
    word_tokenizer().save_pretrained(base)
    architecture = {"backbone": {"pretrained": "${QKV_BASE_MODEL_PATH}", "source_repo": "tiny/qwen3.5",
                                 "revision": "0123abc", "attn_implementation": "eager", "torch_dtype": "float32",
                                 "trust_remote_code": False, "local_files_only": True},
                    "slot_extractor": {"num_slots": 3, "d_attn": 16, "num_heads": 2, "dropout": 0.1,
                                       "use_layer_norm": True},
                    "heads": {"d_k": 8, "d_v": 8, "value_adapter_hidden_dim": 16, "value_adapter_dropout": 0.1}}
    train = {"stage": "candidate_distill", "use_lora": False, "mixed_precision": "bf16",
             "loss": {"qk_mode": "logmeanexp", "tau_k": 0.02, "tau_v": 0.1, "value_mix": 0.25}}
    network, tokenizer = QKVRetriever.from_config(
        architecture, train_config=train, backbone_init="config", backbone_source_override=str(base),
        tokenizer_source_override=str(base), local_files_only_override=True, enable_gradient_checkpointing=False)
    with torch.no_grad():  # make the zero-initialised parts non-trivial, as after training
        for parameter in network.value_adapter.parameters():
            parameter.normal_(0, 0.2)
        network.jepa_mask_embedding_delta.normal_(0, 1.0)
        network.backbone.get_input_embeddings().weight[-1].normal_(0, 1.0)
    network.eval()
    data = {"query_format": "{instruction} Query: {query}", "document_format": "{text}",
            "default_instruction": "search", "max_query_length": 12, "max_document_length": 16,
            "train_path": "/private/training/data.jsonl"}
    checkpoint = root / "model.pt"
    torch.save({"model": network.state_dict(), "model_config": architecture, "data_config": data,
                "train_config": train, "optimizer": {"state": "private"}, "step": 5152}, checkpoint)
    return {"base": base, "checkpoint": checkpoint, "network": network, "tokenizer": tokenizer, "data": data}


@pytest.fixture
def tiny_qwen(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5")
    return build_tiny_qwen_checkpoint(tmp_path)
