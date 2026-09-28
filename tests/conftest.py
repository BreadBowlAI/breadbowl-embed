import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from breadbowl_embed import BreadBowl
from breadbowl_embed.architecture import QKVRetriever
from breadbowl_embed.config import FORMAT


@pytest.fixture
def tiny_model():
    torch.manual_seed(13)
    tokenizer_core = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "search": 2,
                                        "Query": 3, ":": 4, "cats": 5, "dogs": 6,
                                        "fish": 7, "water": 8, "land": 9}, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, pad_token="[PAD]", unk_token="[UNK]")
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
