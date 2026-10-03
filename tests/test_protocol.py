import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vrambiter import protocol as p
from vrambiter.errors import ProtocolError


def test_encode_is_one_compact_line():
    line = p.encode(p.Acquire(id=3, model="voxcpm2", timeout_s=2.5))
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {
        "type": "acquire",
        "id": 3,
        "model": "voxcpm2",
        "wait": True,
        "timeout_s": 2.5,
    }


def test_none_fields_are_omitted():
    wire = p.Register(id=1, model="m", vram_peak=10).to_wire()
    assert "vram_resident" not in wire and "vram_bytes" not in wire


@pytest.mark.parametrize("cls", list(p.REQUESTS.values()))
def test_every_request_roundtrips(cls):
    msg = _example(cls)
    assert p.decode_request(p.encode(msg)) == msg


@pytest.mark.parametrize("cls", list(p.REPLIES.values()))
def test_every_reply_roundtrips(cls):
    msg = _example(cls)
    assert p.decode_reply(p.encode(msg)) == msg


def _example(cls):
    samples = {
        p.Hello: dict(name="tts", pid=42, version="0.1", role="satellite"),
        p.Register: dict(
            model="m",
            vram_peak=9,
            vram_resident=8,
            host_peak=6,
            priority=2,
            pinned=True,
            state="resident",
            vram_bytes=7,
            leases=1,
        ),
        p.Acquire: dict(model="llama/gemma", wait=False, timeout_s=1.0),
        p.Cancel: dict(request=7),
        p.Loaded: dict(model="m", vram_bytes=5),
        p.LoadFailed: dict(model="m", error="boom"),
        p.Release: dict(lease="L1"),
        p.Unloaded: dict(model="m"),
        p.Evicted: dict(model="m", evict_id="E1"),
        p.EvictRefused: dict(model="m", evict_id="E1", reason="busy"),
        p.Pin: dict(model="a/b"),
        p.Unpin: dict(model="a/b"),
        p.EvictModel: dict(model="a/b"),
        p.Profile: dict(name="party"),
        p.StatusRequest: {},
        p.Welcome: dict(satellite="tts", arbiter_version="0.1"),
        p.Ok: dict(model="tts/m", leases=["L1"]),
        p.Error: dict(code="unknown_model", message="no", detail={"model": "x"}),
        p.Granted: dict(lease="L1", action="load", model="tts/m"),
        p.StatusReply: dict(devices=[{"index": 0}], models=[{"id": "a/b"}]),
        p.Evict: dict(evict_id="E1", model="m", reason="lru"),
        p.Notice: dict(kind="lease_lost", message="x", data={"lease": "L1"}),
    }
    return cls(id=5, **samples[cls])


def test_unknown_fields_are_ignored():
    msg = p.decode_request(b'{"type":"release","id":1,"lease":"L9","future_field":{"a":1}}\n')
    assert msg == p.Release(id=1, lease="L9")


def test_unknown_type_carries_request_id():
    with pytest.raises(p.UnknownMessageType) as info:
        p.decode_request(b'{"type":"teleport","id":12}')
    assert info.value.request_id == 12
    assert info.value.type_name == "teleport"


def test_status_type_name_is_direction_dependent():
    assert isinstance(p.decode_request(b'{"type":"status","id":1}'), p.StatusRequest)
    assert isinstance(p.decode_reply(b'{"type":"status","id":1}'), p.StatusReply)


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b"[1,2]",
        b'{"id":1}',
        b'{"type":5}',
        b'{"type":"acquire","id":1}',  # missing model
        b'{"type":"acquire","id":"x","model":"m"}',  # non-int id
        b'{"type":"acquire","id":true,"model":"m"}',
        b'{"type":"register","id":1,"model":"m","vram_peak":"9GiB"}',  # sizes are ints on the wire
        b'{"type":"register","id":1,"model":"m","vram_peak":true}',
        b'{"type":"acquire","id":1,"model":"m","wait":"yes"}',
        b"\xff\xfe",
    ],
)
def test_malformed_lines_raise_protocol_error(line):
    with pytest.raises(ProtocolError):
        p.decode_request(line)


def test_overlong_line_rejected():
    with pytest.raises(ProtocolError):
        p.decode_request(b"x" * (p.MAX_LINE_BYTES + 1))


def test_float_field_accepts_int():
    msg = p.decode_request(b'{"type":"acquire","id":1,"model":"m","timeout_s":3}')
    assert msg.timeout_s == 3.0 and isinstance(msg.timeout_s, float)


def test_split_model_id():
    assert p.split_model_id("llama/gemma") == ("llama", "gemma")
    assert p.split_model_id("llama/ggml-org/gemma:Q4") == ("llama", "ggml-org/gemma:Q4")
    for bad in ("gemma", "/x", "x/", ""):
        with pytest.raises(ValueError):
            p.split_model_id(bad)


@given(st.text(max_size=50), st.integers(min_value=0, max_value=2**62))
def test_arbitrary_strings_roundtrip(name, size):
    msg = p.Register(id=1, model=name, vram_peak=size)
    assert p.decode_request(p.encode(msg)) == msg
