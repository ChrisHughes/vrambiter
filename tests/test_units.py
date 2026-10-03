import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vrambiter.units import GiB, KiB, MiB, TiB, format_bytes, parse_bytes, parse_optional_bytes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("4096", 4096),
        ("17GiB", 17 * GiB),
        ("17 GiB", 17 * GiB),
        ("17gib", 17 * GiB),
        ("17Gi", 17 * GiB),
        ("9GB", 9 * 10**9),
        ("9G", 9 * 10**9),
        ("512MiB", 512 * MiB),
        ("512MB", 512 * 10**6),
        ("1KiB", KiB),
        ("1k", 1000),
        ("2TiB", 2 * TiB),
        ("1.5GiB", int(1.5 * GiB)),
        (".5GiB", GiB // 2),
        ("10B", 10),
        ("  3 MiB  ", 3 * MiB),
    ],
)
def test_parse_strings(text, expected):
    assert parse_bytes(text) == expected


def test_parse_ints_and_floats():
    assert parse_bytes(123) == 123
    assert parse_bytes(1.2) == 2  # fractions round up: a size is a claim on memory
    assert parse_bytes(0.0) == 0


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "GiB",
        "-1GiB",
        "1.2.3GB",
        "12 parsecs",
        "1e9",
        -1,
        -0.5,
        float("nan"),
        float("inf"),
        True,
        None,
        [1],
    ],
)
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_bytes(bad)


def test_fraction_rounds_up():
    assert parse_bytes("0.1KiB") == math.ceil(0.1 * 1024)


def test_parse_optional():
    assert parse_optional_bytes(None) is None
    assert parse_optional_bytes("1KiB") == 1024


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (KiB, "1.0 KiB"),
        (9 * GiB, "9.0 GiB"),
        (int(29.7 * GiB), "29.7 GiB"),
        (-2 * GiB, "-2.0 GiB"),
        (3 * TiB, "3.0 TiB"),
        (None, "?"),
    ],
)
def test_format(n, text):
    assert format_bytes(n) == text


@given(st.integers(min_value=0, max_value=2**50), st.sampled_from(["", "B", "KiB", "MiB", "GiB"]))
def test_roundtrip_integral_units(n, unit):
    mult = {"": 1, "B": 1, "KiB": KiB, "MiB": MiB, "GiB": GiB}[unit]
    assert parse_bytes(f"{n}{unit}") == n * mult
