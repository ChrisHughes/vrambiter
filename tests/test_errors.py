from vrambiter.errors import (
    ArbiterError,
    Holder,
    NameInUse,
    UnknownModel,
    VramUnavailable,
    error_from_wire,
)
from vrambiter.units import GiB


def test_vram_unavailable_message_names_holders():
    err = VramUnavailable(
        model="app/big",
        need=79 * GiB,
        free=75 * GiB,
        reason="nothing evictable or busy could make room",
        holders=[
            Holder(kind="foreign", name="pid 4321", bytes=20 * GiB, pid=4321),
            Holder(kind="model", name="tts/voxcpm2", bytes=8 * GiB, state="resident", pinned=True),
        ],
    )
    text = str(err)
    assert "app/big needs 79.0 GiB" in text
    assert "75.0 GiB is available" in text
    assert "pid 4321 (foreign) 20.0 GiB" in text
    assert "tts/voxcpm2 (resident, pinned) 8.0 GiB" in text


def test_vram_unavailable_wire_roundtrip():
    err = VramUnavailable(
        model="a/b",
        device=1,
        need=10,
        free=3,
        reason="timed out",
        holders=[Holder(kind="model", name="x/y", bytes=7, device=1, state="busy", priority=2)],
    )
    wire = err.to_wire()
    back = error_from_wire(wire["code"], wire["message"], wire["detail"])
    assert isinstance(back, VramUnavailable)
    assert (back.model, back.device, back.need, back.free, back.reason) == (
        "a/b",
        1,
        10,
        3,
        "timed out",
    )
    assert back.holders == err.holders
    assert str(back) == str(err)


def test_error_from_wire_known_and_unknown_codes():
    assert isinstance(error_from_wire("unknown_model", "nope"), UnknownModel)
    assert isinstance(error_from_wire("name_in_use", "taken"), NameInUse)
    unknown = error_from_wire("frobnicated", "what", {"x": 1})
    assert isinstance(unknown, ArbiterError)
    assert unknown.detail == {"x": 1, "code": "frobnicated"}


def test_holder_tolerates_extra_fields_on_decode():
    detail = {
        "model": "a/b",
        "holders": [{"kind": "foreign", "name": "pid 1", "bytes": 5, "new": 1}],
    }
    err = VramUnavailable.from_detail("m", detail)
    assert err.holders == [Holder(kind="foreign", name="pid 1", bytes=5)]
