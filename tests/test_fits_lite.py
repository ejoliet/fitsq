from __future__ import annotations

import numpy as np
import pytest

import synth
from fitsq import fits_lite as fl


def fetch_for(data: bytes) -> fl.Fetch:
    class _F:
        def __call__(self, offset: int, length: int) -> bytes:
            return data[offset : offset + length]

    return _F()


def test_parse_card_value_and_comment() -> None:
    c = fl.parse_card(f"{'NAXIS1  = ':<10}{'91':>20} / row length" + " " * 40)
    assert c is not None
    assert (c.key, c.value, c.comment) == ("NAXIS1", "91", "row length")


def test_parse_card_quoted_value_with_slash_and_escape() -> None:
    raw = f"{'TTYPE1  =':<10}'a/b''c'   / real comment".ljust(80)
    c = fl.parse_card(raw)
    assert c is not None
    assert c.value == "a/b'c"
    assert c.comment == "real comment"


def test_parse_card_blank_and_valueless() -> None:
    assert fl.parse_card(" " * 80) is None
    c = fl.parse_card("COMMENT hello there".ljust(80))
    assert c is not None and c.key == "COMMENT" and c.value == ""


def test_read_header_spans_multiple_blocks() -> None:
    extra = [synth.card("COMMENT", None, f"filler {i}") for i in range(120)]
    data = synth.primary_header(extra)
    assert len(data) > fl.BLOCK
    header = fl.read_header(fetch_for(data), 0)
    assert header.nbytes == len(data)
    assert header.get("SIMPLE") == "T"
    assert sum(1 for c in header.cards if c.key == "COMMENT") == 120


def test_read_header_without_end_raises() -> None:
    broken = synth.header_block([synth.card("SIMPLE", True)]).replace(b"END", b"NOT", 1)
    with pytest.raises(fl.FitsError, match="END"):
        fl.read_header(fetch_for(broken), 0)


def test_read_header_truncated_raises() -> None:
    with pytest.raises(fl.FitsError, match="truncated"):
        fl.read_header(fetch_for(b""), 0)


def test_tform_to_dtype() -> None:
    assert fl.tform_to_dtype("1K") == (">i8", 8)
    assert fl.tform_to_dtype("D") == (">f8", 8)
    assert fl.tform_to_dtype("3A") == ("S3", 3)
    assert fl.tform_to_dtype("8D") == ("8>f8", 64)
    assert fl.tform_to_dtype("16X") == ("S2", 2)
    with pytest.raises(fl.FitsError):
        fl.tform_to_dtype("1PJ")
    with pytest.raises(fl.FitsError):
        fl.tform_to_dtype("banana")


def test_data_nbytes_and_padding() -> None:
    header = fl.read_header(fetch_for(synth.primary_header()), 0)
    assert fl.data_nbytes(header) == 0
    assert fl.padded(1) == fl.BLOCK
    assert fl.padded(fl.BLOCK) == fl.BLOCK


def test_read_bintable_info_matches_ground_truth() -> None:
    ra, dec = synth.patch(268.77, -29.25, n=10)
    data = synth.catalog(ra, dec)
    info = fl.read_bintable_info(fetch_for(data))
    assert info.nrows == 10
    assert info.row_bytes == 91
    assert info.data_offset == len(synth.primary_header()) + len(synth.bintable_header(10))
    assert (info.ra_col, info.dec_col) == ("ra", "dec")
    rows = fl.decode_rows(data[info.data_offset : info.data_offset + 10 * 91], info.dtype)
    np.testing.assert_allclose(rows["ra"], ra)
    np.testing.assert_allclose(rows["dec"], dec)
    assert [hdu for hdu, _ in info.cards].count(1) > 20


def test_read_bintable_info_with_nonempty_primary_data() -> None:
    """data_offset must account for primary array bytes and their padding."""
    image_header = synth.header_block(
        [
            synth.card("SIMPLE", True),
            synth.card("BITPIX", -32),
            synth.card("NAXIS", 2),
            synth.card("NAXIS1", 10),
            synth.card("NAXIS2", 10),
            synth.card("EXTEND", True),
        ]
    )
    image_data = b"\0" * fl.padded(10 * 10 * 4)
    ra, dec = synth.patch(10.0, 10.0, n=5)
    data = image_header + image_data + synth.bintable_header(5) + synth.rows_bytes(ra, dec)
    info = fl.read_bintable_info(fetch_for(data))
    assert info.data_offset == len(image_header) + len(image_data) + len(synth.bintable_header(5))
    rows = fl.decode_rows(data[info.data_offset : info.data_offset + 5 * 91], info.dtype)
    np.testing.assert_allclose(rows["dec"], dec)


def test_non_fits_and_non_bintable_rejected() -> None:
    with pytest.raises(fl.FitsError, match="SIMPLE"):
        fl.read_bintable_info(fetch_for(synth.header_block([synth.card("FOO", 1)])))
    image = synth.primary_header() + synth.header_block(
        [synth.card("XTENSION", "IMAGE"), synth.card("NAXIS", 0), synth.card("BITPIX", 8)]
    )
    with pytest.raises(fl.FitsError, match="BINTABLE"):
        fl.read_bintable_info(fetch_for(image))


def test_row_dtype_rejects_width_mismatch() -> None:
    bad = synth.bintable_header(1).replace(
        b"NAXIS1  =                   91", b"NAXIS1  =                   90"
    )
    data = synth.primary_header() + bad
    with pytest.raises(fl.FitsError, match="NAXIS1"):
        fl.read_bintable_info(fetch_for(data))


def test_missing_coord_columns() -> None:
    header = synth.bintable_header(1).replace(b"'ra'", b"'xx'")
    with pytest.raises(fl.FitsError, match="no ra/dec"):
        fl.read_bintable_info(fetch_for(synth.primary_header() + header))


def test_sample_windows() -> None:
    assert fl.sample_windows(0, 3, 100) == []
    assert fl.sample_windows(250, 3, 100) == [(0, 250)]  # whole table when small
    assert fl.sample_windows(1000, 3, 100) == [(0, 100), (450, 100), (900, 100)]
    windows = fl.sample_windows(1000, 5, 100)
    assert windows[0] == (0, 100) and windows[-1] == (900, 100) and len(windows) == 5
    assert fl.sample_windows(1000, 1, 100) == [(0, 100)]


def test_decode_rows_ignores_partial_row() -> None:
    ra, dec = synth.patch(1.0, 1.0, n=3)
    raw = synth.rows_bytes(ra, dec)[: 2 * 91 + 40]
    assert len(fl.decode_rows(raw, synth.ROW_DTYPE)) == 2
