from __future__ import annotations

import base64
import math
import struct
import sys

import pytest

from breadbowl import ScoreResponse, Tensor, TensorDecodeError
from tests.conftest import example, expanded

try:
    import numpy as np
except ImportError:  # pragma: no cover - the pure-Python paths are tested either way
    np = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy not installed")


def b64(fmt: str, values: list) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}{fmt}", *values)).decode()


def f16_bits(bits: list) -> str:
    return base64.b64encode(struct.pack(f"<{len(bits)}H", *bits)).decode()


class TestFloat16:
    def test_known_bit_patterns(self) -> None:
        # 1.0, -2.0, 0.5, max finite, smallest subnormal, zero, negative zero
        bits = [0x3C00, 0xC000, 0x3800, 0x7BFF, 0x0001, 0x0000, 0x8000]
        t = Tensor(
            encoding="base64",
            dtype="float16",
            shape=[7],
            axes=["dim"],
            data=f16_bits(bits),
            byte_order="little",
        )
        values = t.to_list()
        assert values[:4] == [1.0, -2.0, 0.5, 65504.0]
        assert values[4] == 2.0**-24
        assert values[5] == 0.0 and math.copysign(1, values[6]) == -1

    def test_infinity_and_nan_bits_decode(self) -> None:
        t = Tensor(
            encoding="base64", dtype="float16", shape=[2], axes=["dim"], data=f16_bits([0x7C00, 0x7E00])
        )
        inf, nan = t.flat()
        assert inf == math.inf and math.isnan(nan)

    def test_row_major_nesting(self) -> None:
        t = Tensor(
            encoding="base64",
            dtype="float16",
            shape=[2, 3],
            axes=["slot", "dim"],
            data=b64("e", [1, 2, 3, 4, 5, 6]),
        )
        assert t.to_list() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert t[1, 2] == 6.0 and t[0] == [1.0, 2.0, 3.0] and t[-1, 0] == 4.0
        assert t.axis("dim") == 1 and len(t) == 2 and t.size == 6 and t.ndim == 2

    def test_float_encoding_is_rounded_to_dtype(self) -> None:
        # 0.71 is not a float16 value; both encodings must decode to the same float16 number.
        as_float = Tensor(encoding="float", dtype="float16", shape=[1], axes=["query_slot"], data=[0.71])
        as_b64 = Tensor(
            encoding="base64", dtype="float16", shape=[1], axes=["query_slot"], data=b64("e", [0.71])
        )
        assert as_float.flat() == as_b64.flat() == [0.7099609375]
        assert as_float.to_bytes() == as_b64.to_bytes()


class TestFloat32:
    def test_base64_round_trip(self) -> None:
        values = [0.1, -1.5, 3.25, 1e-30]
        t = Tensor(
            encoding="base64", dtype="float32", shape=[2, 2], axes=["slot", "dim"], data=b64("f", values)
        )
        expected = [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]
        assert t.flat() == expected

    def test_float_encoding_three_dims(self) -> None:
        data = [[[0.5, 0.25], [0.125, 0.125]]]
        t = Tensor(
            encoding="float", dtype="float32", shape=[1, 2, 2], axes=["head", "slot", "token"], data=data
        )
        assert t.to_list() == data

    def test_from_list_both_encodings(self) -> None:
        values = [[1.0, 2.5], [-3.0, 0.0]]
        for encoding in ("base64", "float"):
            t = Tensor.from_list(values, axes=["slot", "dim"], dtype="float32", encoding=encoding)
            assert t.to_list() == values and t.shape == (2, 2)
            assert Tensor.from_dict(t.to_dict()) == t


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (
                dict(encoding="base64", dtype="float16", shape=[3], axes=["dim"], data=b64("e", [1, 2])),
                "needs 6",
            ),
            (
                dict(encoding="base64", dtype="float32", shape=[2], axes=["dim"], data=b64("e", [1, 2])),
                "needs 8",
            ),
            (
                dict(encoding="base64", dtype="float16", shape=[1], axes=["dim"], data="!!!"),
                "not valid base64",
            ),
            (dict(encoding="base64", dtype="float64", shape=[1], axes=["dim"], data=""), "unsupported dtype"),
            (dict(encoding="hex", dtype="float16", shape=[1], axes=["dim"], data=""), "unsupported encoding"),
            (
                dict(encoding="base64", dtype="float16", shape=[1, 1], axes=["dim"], data=""),
                "do not match shape",
            ),
            (dict(encoding="base64", dtype="float16", shape=[-1], axes=["dim"], data=""), "invalid shape"),
            (
                dict(
                    encoding="base64",
                    dtype="float16",
                    shape=[1],
                    axes=["dim"],
                    data=b64("e", [1]),
                    byte_order="big",
                ),
                "byte_order",
            ),
            (
                dict(
                    encoding="float", dtype="float16", shape=[2, 2], axes=["slot", "dim"], data=[[1, 2], [3]]
                ),
                "length 1",
            ),
            (
                dict(encoding="float", dtype="float16", shape=[2], axes=["dim"], data=[[1], [2]]),
                "must be a number",
            ),
            (
                dict(encoding="float", dtype="float16", shape=[2], axes=["dim"], data=[1, True]),
                "must be a number",
            ),
            (
                dict(encoding="float", dtype="float16", shape=[1], axes=["dim"], data=[1e6]),
                "not representable",
            ),
            (dict(encoding="float", dtype="float32", shape=[1], axes=["dim"], data=3.0), "must be a list"),
        ],
    )
    def test_rejects(self, kwargs: dict, message: str) -> None:
        with pytest.raises(TensorDecodeError, match=message):
            Tensor(**kwargs).to_list()

    def test_tensor_decode_error_is_value_error(self) -> None:
        assert issubclass(TensorDecodeError, ValueError)

    def test_from_dict_requires_fields(self) -> None:
        with pytest.raises(TensorDecodeError, match="missing"):
            Tensor.from_dict({"encoding": "base64"})

    def test_zero_sized_axis(self) -> None:
        t = Tensor(encoding="base64", dtype="float16", shape=[16, 0], axes=["slot", "token"], data="")
        assert t.to_list() == [[] for _ in range(16)]


class TestNumpy:
    @needs_numpy
    def test_to_numpy_matches_to_list(self) -> None:
        t = Tensor(
            encoding="base64",
            dtype="float16",
            shape=[2, 3],
            axes=["slot", "dim"],
            data=b64("e", [1, 2, 3, 4, 5, 0.1]),
        )
        arr = t.to_numpy()
        assert arr.dtype == np.float16 and arr.shape == (2, 3)
        assert arr.astype(float).tolist() == t.to_list()
        assert t.to_numpy("float32").dtype == np.float32
        assert np.asarray(t).shape == (2, 3)

    @needs_numpy
    def test_float_encoding_to_numpy(self) -> None:
        t = Tensor(encoding="float", dtype="float32", shape=[2], axes=["dim"], data=[0.5, 1.5])
        assert t.to_numpy().tolist() == [0.5, 1.5]

    @needs_numpy
    def test_numpy_checks_length(self) -> None:
        t = Tensor(encoding="base64", dtype="float16", shape=[4], axes=["dim"], data=b64("e", [1]))
        with pytest.raises(TensorDecodeError):
            t.to_numpy()

    def test_missing_numpy_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "numpy", None)
        t = Tensor(encoding="float", dtype="float32", shape=[1], axes=["dim"], data=[1.0])
        with pytest.raises(ImportError, match=r"breadbowl\[numpy\]"):
            t.to_numpy()
        assert t.to_list() == [1.0]  # the pure-Python path keeps working


class TestContractTensors:
    def test_expanded_encode_example_decodes_to_declared_shapes(self) -> None:
        body = expanded("encode", 200, "queryWithAttention")
        item = body["data"][0]
        routing = Tensor.from_dict(item["routing"])
        attention = Tensor.from_dict(item["attention"])
        assert len(routing.to_list()) == 16 and len(routing.to_list()[0]) == 256
        assert (
            attention.shape == (1, 16, item["token_count"])
            and len(attention.flat()) == 16 * item["token_count"]
        )
        assert routing.flat()[0] == 1.0  # the example bytes are float16 1.0 (0x3C00)

    def test_verbatim_contract_examples_are_abbreviated(self) -> None:
        # The contract shortens tensor data in its examples; the SDK refuses to guess.
        item = example("encode", 200)["data"][0]
        with pytest.raises(TensorDecodeError, match="needs 8192"):
            Tensor.from_dict(item["routing"]).to_list()

    def test_explained_score_example_mixed_encodings(self) -> None:
        res = ScoreResponse.from_dict(expanded("score", 200, "explained"))
        exp = res.results[0].explanation
        assert exp is not None
        assert exp.slot_similarity.shape == (16, 16) and exp.routing_contribution.encoding == "float"
        assert len(exp.value_attention.flat()) == 256
        assert exp.routing_contribution.flat()[0] == pytest.approx(0.71, abs=1e-3)
