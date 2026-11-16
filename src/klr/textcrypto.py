"""Super enkripsi pesan teks (laporan subbab 3.3).

Dua lapis berurutan:

    plaintext --UTF-8--> byte --Vigenere byte-level (K_v, 16 byte)--> --AES-256-GCM (K_a, nonce)--> ciphertext

K_v dan K_a dibuat acak untuk SETIAP laporan, lalu dibungkus (48 byte) dengan
RSA-OAEP (SHA-256) memakai kunci publik enkripsi Admin. Hanya Admin yang punya
kunci privatnya, jadi hanya Admin yang bisa membuka.

Format paket (di-base64 sebelum disimpan):

    +---------+------------------+-----------+---------------------------+
    | versi   | blob RSA-OAEP    | nonce     | ciphertext AES-GCM + tag  |
    | 1 byte  | 256 byte (2048b) | 12 byte   | panjang pesan + 16 byte   |
    +---------+------------------+-----------+---------------------------+
"""
from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import DecryptionError

VERSION = 1
KV_LEN = 16  # kunci Vigenere
KA_LEN = 32  # kunci AES-256
NONCE_LEN = 12  # nonce standar GCM
TAG_LEN = 16  # tag autentikasi GCM

_OAEP = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


# --------------------------------------------------------------------------
# Lapis 1: Vigenere byte-level
# --------------------------------------------------------------------------
def vigenere_encrypt(data: bytes, key: bytes) -> bytes:
    """C_i = (P_i + K_(i mod m)) mod 256, untuk setiap byte."""
    if not key:
        raise ValueError("kunci Vigenere tidak boleh kosong")
    m = len(key)
    return bytes((b + key[i % m]) & 0xFF for i, b in enumerate(data))


def vigenere_decrypt(data: bytes, key: bytes) -> bytes:
    """P_i = (C_i - K_(i mod m)) mod 256."""
    if not key:
        raise ValueError("kunci Vigenere tidak boleh kosong")
    m = len(key)
    return bytes((b - key[i % m]) & 0xFF for i, b in enumerate(data))


# --------------------------------------------------------------------------
# Lapis 2: AES-256-GCM, dengan AAD = report_id + reporter_id
# --------------------------------------------------------------------------
def _length_prefixed(text: str) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ValueError("id terlalu panjang")
    return len(raw).to_bytes(2, "big") + raw


def build_aad(report_id: str, reporter_id: str) -> bytes:
    """Data tambahan terautentikasi (AAD).

    Tiap id diawali panjangnya, supaya ("ab", "c") dan ("a", "bc") menghasilkan
    AAD yang berbeda. Kalau hanya disambung, keduanya sama dan ciphertext
    bisa dipindahkan antar laporan tanpa terdeteksi.
    """
    return _length_prefixed(report_id) + _length_prefixed(reporter_id)


# --------------------------------------------------------------------------
# Enkripsi dan dekripsi paket
# --------------------------------------------------------------------------
def encrypt_text(
    plaintext: str,
    admin_public_key: RSAPublicKey,
    report_id: str,
    reporter_id: str,
) -> str:
    """Enkripsi teks laporan menjadi paket base64.

    Dipanggil oleh Pelapor. Hanya butuh kunci PUBLIK Admin.
    """
    k_v = os.urandom(KV_LEN)
    k_a = os.urandom(KA_LEN)
    nonce = os.urandom(NONCE_LEN)

    inner = vigenere_encrypt(plaintext.encode("utf-8"), k_v)  # lapis 1
    sealed = AESGCM(k_a).encrypt(nonce, inner, build_aad(report_id, reporter_id))  # lapis 2
    blob = admin_public_key.encrypt(k_v + k_a, _OAEP)  # bungkus kunci sesi

    return base64.b64encode(bytes([VERSION]) + blob + nonce + sealed).decode("ascii")


def _unwrap(
    packet: str | bytes,
    admin_private_key: RSAPrivateKey,
    report_id: str,
    reporter_id: str,
) -> tuple[bytes, bytes]:
    """Buka paket sampai lapisan Vigenere. Mengembalikan (K_v, hasil_AES).

    Semua kegagalan dijadikan DecryptionError dengan pesan yang sama.
    """
    try:
        raw = base64.b64decode(packet, validate=True)
    except (ValueError, TypeError):
        raise DecryptionError("dekripsi gagal") from None

    blob_len = admin_private_key.key_size // 8
    header_len = 1 + blob_len + NONCE_LEN
    if len(raw) < header_len + TAG_LEN or raw[0] != VERSION:
        raise DecryptionError("dekripsi gagal")

    blob = raw[1 : 1 + blob_len]
    nonce = raw[1 + blob_len : header_len]
    sealed = raw[header_len:]

    try:
        keys = admin_private_key.decrypt(blob, _OAEP)
        if len(keys) != KV_LEN + KA_LEN:
            raise ValueError
        k_v, k_a = keys[:KV_LEN], keys[KV_LEN:]
        inner = AESGCM(k_a).decrypt(nonce, sealed, build_aad(report_id, reporter_id))
    except (ValueError, InvalidTag):
        raise DecryptionError("dekripsi gagal") from None
    return k_v, inner


def decrypt_text(
    packet: str | bytes,
    admin_private_key: RSAPrivateKey,
    report_id: str,
    reporter_id: str,
) -> str:
    """Dekripsi paket menjadi teks asli. Dipanggil oleh Admin.

    Urutan: buka blob RSA -> verifikasi tag AES-GCM (gagal = tolak, tidak ada
    plaintext sama sekali) -> Vigenere invers.
    """
    k_v, inner = _unwrap(packet, admin_private_key, report_id, reporter_id)
    try:
        return vigenere_decrypt(inner, k_v).decode("utf-8")
    except UnicodeDecodeError:
        raise DecryptionError("dekripsi gagal") from None
