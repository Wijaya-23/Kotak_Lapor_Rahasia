"""Manajemen kunci yang dibuka oleh KEK (laporan subbab 3.2 dan 3.5.1).

Yang dibungkus di sini, masing-masing dengan SUB-KUNCI KEK yang berbeda:

    kunci privat enkripsi Admin  -> users.enc_privkey_wrapped   (sub-kunci klr-privkey-enc-v1)
    DEK (kunci basis data)        -> key_slots.wrapped_dek       (sub-kunci klr-dek-slot-v1)
    template wajah                -> users.face_template_enc     (sub-kunci klr-face-template-v1)

Setiap hasil bungkusan juga diikat ke user_id lewat AAD, jadi slot milik A yang
disalin ke baris B tidak bisa dibuka walaupun B punya KEK yang benar untuk dirinya.
"""
from __future__ import annotations

import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .aead import build_aad, open_sealed, seal
from .errors import DecryptionError
from .kdf import INFO_DEK_SLOT, INFO_FACE_TEMPLATE, INFO_PRIVKEY_ENC, derive_subkey

DEK_LEN = 32
RSA_BITS = 2048

_PURPOSE_PRIVKEY = "klr-enc-privkey"
_PURPOSE_DEK = "klr-dek-slot"
_PURPOSE_FACE = "klr-face-template"


# --------------------------------------------------------------------------
# Pasangan kunci enkripsi Admin (terpisah dari kunci penandatangan di dongle)
# --------------------------------------------------------------------------
def generate_encryption_keypair() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=RSA_BITS)


def encryption_public_key_pem(private_key: rsa.RSAPrivateKey) -> str:
    """Kunci publik untuk kolom users.enc_pubkey. Dipakai Pelapor untuk mengenkripsi laporan."""
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


def load_encryption_public_key(pem: str) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(pem.encode("ascii"))
    if not isinstance(key, rsa.RSAPublicKey):
        raise ValueError("kunci publik enkripsi bukan RSA")
    return key


def wrap_private_key(private_key: rsa.RSAPrivateKey, kek: bytes, user_id: str) -> str:
    """Bungkus kunci privat enkripsi dengan sub-kunci KEK. Hasilnya aman disimpan di basis data."""
    der = private_key.private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    return seal(derive_subkey(kek, INFO_PRIVKEY_ENC), der, build_aad(_PURPOSE_PRIVKEY, user_id))


def unwrap_private_key(blob: str, kek: bytes, user_id: str) -> rsa.RSAPrivateKey:
    der = open_sealed(derive_subkey(kek, INFO_PRIVKEY_ENC), blob, build_aad(_PURPOSE_PRIVKEY, user_id))
    try:
        key = serialization.load_der_private_key(der, password=None)
    except ValueError:
        raise DecryptionError("dekripsi gagal") from None
    if not isinstance(key, rsa.RSAPrivateKey):
        raise DecryptionError("dekripsi gagal")
    return key


# --------------------------------------------------------------------------
# DEK dan key slot
# --------------------------------------------------------------------------
def new_dek() -> bytes:
    """DEK 256 bit acak. Dibuat SEKALI saat inisialisasi sistem."""
    return os.urandom(DEK_LEN)


def wrap_dek(dek: bytes, kek: bytes, user_id: str) -> str:
    """Satu slot DEK untuk satu pengguna (baris di tabel key_slots)."""
    if len(dek) != DEK_LEN:
        raise ValueError(f"DEK harus {DEK_LEN} byte")
    return seal(derive_subkey(kek, INFO_DEK_SLOT), dek, build_aad(_PURPOSE_DEK, user_id))


def unwrap_dek(blob: str, kek: bytes, user_id: str) -> bytes:
    dek = open_sealed(derive_subkey(kek, INFO_DEK_SLOT), blob, build_aad(_PURPOSE_DEK, user_id))
    if len(dek) != DEK_LEN:
        raise DecryptionError("dekripsi gagal")
    return dek


# --------------------------------------------------------------------------
# Template wajah (isinya byte; serialisasi embedding diurus modul wajah)
# --------------------------------------------------------------------------
def wrap_face_template(template: bytes, kek: bytes, user_id: str) -> str:
    return seal(derive_subkey(kek, INFO_FACE_TEMPLATE), template, build_aad(_PURPOSE_FACE, user_id))


def unwrap_face_template(blob: str, kek: bytes, user_id: str) -> bytes:
    return open_sealed(derive_subkey(kek, INFO_FACE_TEMPLATE), blob, build_aad(_PURPOSE_FACE, user_id))
