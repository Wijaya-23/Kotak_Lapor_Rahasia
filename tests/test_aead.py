import base64
import os

import pytest

from klr import aead
from klr.errors import DecryptionError

KEY = bytes(range(32))
AAD = aead.build_aad("tabel", "kolom", "baris-1")


def _flip(blob: str, index: int) -> str:
    raw = bytearray(base64.b64decode(blob))
    raw[index] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


@pytest.mark.parametrize("data", [b"", b"x", os.urandom(1000), "teks \U0001f512".encode()])
def test_round_trip(data):
    assert aead.open_sealed(KEY, aead.seal(KEY, data, AAD), AAD) == data


def test_format_dan_nonce_acak():
    a, b = aead.seal(KEY, b"sama", AAD), aead.seal(KEY, b"sama", AAD)
    assert a != b
    raw = base64.b64decode(a)
    assert raw[0] == aead.VERSION
    assert len(raw) == 1 + 12 + 4 + 16


def test_perubahan_satu_bit_terdeteksi_di_mana_pun():
    blob = aead.seal(KEY, b"data penting", AAD)
    for i in range(len(base64.b64decode(blob))):
        with pytest.raises(DecryptionError):
            aead.open_sealed(KEY, _flip(blob, i), AAD)


def test_kunci_salah_ditolak():
    blob = aead.seal(KEY, b"data", AAD)
    with pytest.raises(DecryptionError):
        aead.open_sealed(bytes(32), blob, AAD)


def test_aad_salah_ditolak():
    blob = aead.seal(KEY, b"data", AAD)
    with pytest.raises(DecryptionError):
        aead.open_sealed(KEY, blob, aead.build_aad("tabel", "kolom", "baris-2"))


def test_build_aad_batas_tidak_ambigu():
    assert aead.build_aad("ab", "c") != aead.build_aad("a", "bc")
    assert aead.build_aad("a", "") != aead.build_aad("a")


@pytest.mark.parametrize("blob", ["", "!!!", base64.b64encode(b"\x01" + bytes(20)).decode()])
def test_paket_rusak_ditolak(blob):
    with pytest.raises(DecryptionError):
        aead.open_sealed(KEY, blob, AAD)


def test_panjang_kunci_divalidasi():
    with pytest.raises(ValueError):
        aead.seal(bytes(16), b"x", AAD)
    with pytest.raises(ValueError):
        aead.open_sealed(bytes(16), aead.seal(KEY, b"x", AAD), AAD)
