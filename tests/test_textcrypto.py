import base64
import os

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from klr import textcrypto as tc
from klr.errors import DecryptionError

REPORT_ID = "rpt-0001"
REPORTER_ID = "user-pelapor-01"


@pytest.fixture(scope="module")
def admin_key():
    # Membuat kunci RSA-2048 agak lambat, jadi cukup sekali per berkas tes.
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _enc(text, key, rid=REPORT_ID, uid=REPORTER_ID):
    return tc.encrypt_text(text, key.public_key(), rid, uid)


def _dec(packet, key, rid=REPORT_ID, uid=REPORTER_ID):
    return tc.decrypt_text(packet, key, rid, uid)


def _flip_bit(packet: str, index: int) -> str:
    raw = bytearray(base64.b64decode(packet))
    raw[index] ^= 0x01
    return base64.b64encode(bytes(raw)).decode("ascii")


# ---------------------------------------------------------------- Vigenere
def test_vigenere_vektor_uji_dengan_wraparound():
    # 0+1=1, 1+2=3, 255+1=256 -> 0 (mod 256), 10+2=12
    assert tc.vigenere_encrypt(b"\x00\x01\xff\x0a", b"\x01\x02") == b"\x01\x03\x00\x0c"
    assert tc.vigenere_decrypt(b"\x01\x03\x00\x0c", b"\x01\x02") == b"\x00\x01\xff\x0a"


def test_vigenere_round_trip_semua_nilai_byte():
    key = os.urandom(16)
    data = bytes(range(256)) * 3
    assert tc.vigenere_decrypt(tc.vigenere_encrypt(data, key), key) == data


def test_vigenere_data_kosong_dan_kunci_lebih_panjang_dari_data():
    assert tc.vigenere_encrypt(b"", b"kunci") == b""
    assert tc.vigenere_decrypt(tc.vigenere_encrypt(b"ab", bytes(range(16))), bytes(range(16))) == b"ab"


def test_vigenere_kunci_kosong_ditolak():
    with pytest.raises(ValueError):
        tc.vigenere_encrypt(b"abc", b"")
    with pytest.raises(ValueError):
        tc.vigenere_decrypt(b"abc", b"")


# ------------------------------------------------------------ round trip
@pytest.mark.parametrize(
    "text",
    [
        "Laporan biasa",
        "",
        "Ada penyimpangan anggaran Rp 1.500.000.000 di divisi X, 12/11/2026.",
        "Emoji \U0001f6a8 dan aksara lain: 日本語, مرحبا, ไทย",
        "baris 1\nbaris 2\r\n\ttab dan nol\x00di tengah",
        "A" * 100_000,
    ],
)
def test_round_trip(text, admin_key):
    assert _dec(_enc(text, admin_key), admin_key) == text


def test_paket_berbeda_setiap_kali_karena_kunci_dan_nonce_acak(admin_key):
    assert _enc("sama", admin_key) != _enc("sama", admin_key)


def test_plaintext_tidak_terlihat_di_paket(admin_key):
    teks = "RAHASIA-SANGAT-PENTING"
    raw = base64.b64decode(_enc(teks, admin_key))
    assert teks.encode() not in raw


def test_format_paket_sesuai_laporan(admin_key):
    teks = "x" * 50
    raw = base64.b64decode(_enc(teks, admin_key))
    # versi 1 + blob RSA 256 + nonce 12 + ciphertext (50) + tag 16
    assert len(raw) == 1 + 256 + 12 + 50 + 16
    assert raw[0] == tc.VERSION


def test_dua_lapis_benar_benar_ada(admin_key):
    """Setelah lapis AES dibuka, isinya BUKAN plaintext (masih terbungkus Vigenere)."""
    teks = "pesan rahasia untuk admin"
    packet = _enc(teks, admin_key)
    k_v, inner = tc._unwrap(packet, admin_key, REPORT_ID, REPORTER_ID)
    assert len(k_v) == 16
    assert inner != teks.encode()
    assert tc.vigenere_decrypt(inner, k_v) == teks.encode()


# ------------------------------------------------------------ integritas
def _posisi_uji(panjang_teks):
    """Satu posisi di tiap bagian paket: versi, blob (awal/tengah/akhir), nonce, ciphertext, tag."""
    blob_awal, blob_akhir = 1, 1 + 256
    nonce = blob_akhir
    ct = nonce + 12
    total = 1 + 256 + 12 + panjang_teks + 16
    return [0, blob_awal, blob_awal + 128, blob_akhir - 1, nonce, nonce + 11, ct, total - 17, total - 16, total - 1]


def test_perubahan_satu_bit_di_bagian_mana_pun_terdeteksi(admin_key):
    teks = "x" * 40
    packet = _enc(teks, admin_key)
    for idx in _posisi_uji(len(teks)):
        with pytest.raises(DecryptionError):
            _dec(_flip_bit(packet, idx), admin_key)


def test_paket_terpotong_atau_sampah_ditolak(admin_key):
    packet = _enc("halo", admin_key)
    raw = base64.b64decode(packet)
    for rusak in [
        "",
        "bukan base64 !!!",
        base64.b64encode(raw[:100]).decode(),
        base64.b64encode(raw[:-1]).decode(),
        base64.b64encode(raw + b"\x00").decode(),
        base64.b64encode(b"\x01" + os.urandom(400)).decode(),
    ]:
        with pytest.raises(DecryptionError):
            _dec(rusak, admin_key)


def test_kunci_admin_salah_ditolak(admin_key, other_key):
    with pytest.raises(DecryptionError):
        _dec(_enc("rahasia", admin_key), other_key)


# ------------------------------------------------------------------- AAD
def test_report_id_berbeda_ditolak(admin_key):
    packet = _enc("rahasia", admin_key)
    with pytest.raises(DecryptionError):
        _dec(packet, admin_key, rid="rpt-0002")


def test_reporter_id_berbeda_ditolak(admin_key):
    packet = _enc("rahasia", admin_key)
    with pytest.raises(DecryptionError):
        _dec(packet, admin_key, uid="user-lain")


def test_batas_antar_id_tidak_ambigu(admin_key):
    """("ab","c") dan ("a","bc") harus dianggap berbeda."""
    packet = _enc("rahasia", admin_key, rid="ab", uid="c")
    assert _dec(packet, admin_key, rid="ab", uid="c") == "rahasia"
    with pytest.raises(DecryptionError):
        _dec(packet, admin_key, rid="a", uid="bc")


def test_aad_unicode_dan_kosong(admin_key):
    packet = _enc("ok", admin_key, rid="", uid="pelapor-é")
    assert _dec(packet, admin_key, rid="", uid="pelapor-é") == "ok"


def test_pesan_error_sama_untuk_semua_penyebab(admin_key, other_key):
    """Penyerang tidak boleh bisa membedakan tahap mana yang gagal."""
    packet = _enc("rahasia", admin_key)
    pesan = set()
    for fn in [
        lambda: _dec(packet, other_key),
        lambda: _dec(_flip_bit(packet, 270), admin_key),  # byte pertama ciphertext
        lambda: _dec("???", admin_key),
        lambda: _dec(packet, admin_key, rid="lain"),
    ]:
        with pytest.raises(DecryptionError) as ei:
            fn()
        pesan.add(str(ei.value))
    assert len(pesan) == 1
