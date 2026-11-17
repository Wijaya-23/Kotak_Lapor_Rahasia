"""AES-256-GCM serbaguna untuk membungkus kunci dan mengenkripsi kolom basis data.

Format hasil (di-base64 agar muat di kolom TEXT SQLite):

    versi (1 byte) || nonce (12 byte) || ciphertext || tag (16 byte)

AAD (data tambahan terautentikasi) mengikat ciphertext pada konteksnya, misalnya
"slot DEK milik user 7". Ciphertext yang dipindah ke baris/pengguna lain akan
ditolak saat dibuka, walaupun kuncinya benar.
"""
from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import DecryptionError

VERSION = 1
KEY_LEN = 32
NONCE_LEN = 12
TAG_LEN = 16


def build_aad(*parts: str) -> bytes:
    """Gabungkan beberapa teks menjadi AAD. Tiap bagian diawali panjangnya (2 byte)
    supaya batas antarbagian tidak ambigu."""
    out = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise ValueError("bagian AAD terlalu panjang")
        out += len(raw).to_bytes(2, "big") + raw
    return bytes(out)


def seal(key: bytes, plaintext: bytes, aad: bytes) -> str:
    """Enkripsi dengan AES-256-GCM. Nonce acak baru setiap pemanggilan."""
    if len(key) != KEY_LEN:
        raise ValueError(f"kunci harus {KEY_LEN} byte")
    nonce = os.urandom(NONCE_LEN)
    header = bytes([VERSION])
    ct = AESGCM(key).encrypt(nonce, plaintext, header + aad)  # versi ikut diautentikasi
    return base64.b64encode(header + nonce + ct).decode("ascii")


def open_sealed(key: bytes, blob: str | bytes, aad: bytes) -> bytes:
    """Buka hasil seal(). Kunci salah, AAD salah, atau data diubah -> DecryptionError."""
    if len(key) != KEY_LEN:
        raise ValueError(f"kunci harus {KEY_LEN} byte")
    try:
        raw = base64.b64decode(blob, validate=True)
    except (ValueError, TypeError):
        raise DecryptionError("dekripsi gagal") from None
    if len(raw) < 1 + NONCE_LEN + TAG_LEN or raw[0] != VERSION:
        raise DecryptionError("dekripsi gagal")
    header, nonce, ct = raw[:1], raw[1 : 1 + NONCE_LEN], raw[1 + NONCE_LEN :]
    try:
        return AESGCM(key).decrypt(nonce, ct, header + aad)
    except InvalidTag:
        raise DecryptionError("dekripsi gagal") from None
