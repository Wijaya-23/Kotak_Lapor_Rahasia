"""Penurunan kunci KLR (laporan subbab 3.2).

Alur:

    password --Argon2id(salt_kek)--> 32 byte ---+
                                                 +--> HKDF-SHA256(info="klr-kek-v1") --> KEK
    dongle_secret (32 byte, di flashdisk) -------+

KEK hanya bisa dihitung kalau password DAN dongle_secret sama-sama benar.
Dari KEK, sub-kunci untuk tiap fungsi diturunkan dengan HKDF memakai label
(info) yang berbeda, sehingga bocornya satu sub-kunci tidak membuka yang lain.

Password juga diverifikasi lewat hash Argon2id terpisah (`hash_password`),
dengan salt berbeda dari salt_kek. Dengan begitu hash yang tersimpan di basis
data tidak bisa dipakai untuk menghitung KEK.
"""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SALT_LEN = 16  # byte
KEY_LEN = 32  # byte (256 bit)
DONGLE_SECRET_LEN = 32  # byte


@dataclass(frozen=True)
class Argon2Params:
    """Parameter Argon2id. Nilai bawaan sesuai laporan: 64 MiB, 3 iterasi, 4 jalur."""

    memory_kib: int = 64 * 1024
    time_cost: int = 3
    parallelism: int = 4


DEFAULT_PARAMS = Argon2Params()

# Label HKDF. Tiap fungsi memakai label sendiri; akhiran -v1 = versi skema.
INFO_KEK = b"klr-kek-v1"
INFO_PRIVKEY_ENC = b"klr-privkey-enc-v1"  # membuka kunci privat enkripsi Admin
INFO_DEK_SLOT = b"klr-dek-slot-v1"  # membuka slot DEK di tabel key_slots
INFO_FACE_TEMPLATE = b"klr-face-template-v1"  # membuka template wajah


def new_salt() -> bytes:
    """Salt acak 16 byte dari generator acak sistem operasi."""
    return os.urandom(SALT_LEN)


def _password_bytes(password: str) -> bytes:
    """Normalisasi Unicode (NFKC) lalu ke byte UTF-8.

    Tanpa normalisasi, password yang sama bisa menghasilkan byte berbeda
    tergantung cara pengetikan (misalnya huruf beraksen), dan login gagal.
    """
    if not password:
        raise ValueError("password tidak boleh kosong")
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def hash_password(password: str, params: Argon2Params = DEFAULT_PARAMS) -> str:
    """Hash Argon2id untuk verifikasi password.

    Hasilnya string `$argon2id$...` yang sudah memuat salt (acak, 16 byte)
    dan parameter, jadi cukup disimpan satu kolom di tabel users.
    """
    hasher = PasswordHasher(
        time_cost=params.time_cost,
        memory_cost=params.memory_kib,
        parallelism=params.parallelism,
        hash_len=KEY_LEN,
        salt_len=SALT_LEN,
        type=Type.ID,
    )
    return hasher.hash(_password_bytes(password).decode("utf-8"))


def verify_password(encoded_hash: str, password: str) -> bool:
    """True kalau password cocok dengan hash. Tidak pernah melempar error karena password salah."""
    try:
        return PasswordHasher().verify(encoded_hash, _password_bytes(password).decode("utf-8"))
    except (VerificationError, InvalidHashError, ValueError):
        return False


def derive_kek(
    password: str,
    salt_kek: bytes,
    dongle_secret: bytes,
    params: Argon2Params = DEFAULT_PARAMS,
) -> bytes:
    """Hitung KEK (Key Encryption Key) 32 byte.

    KEK = HKDF-SHA256( Argon2id(password, salt_kek) || dongle_secret, info="klr-kek-v1" )

    `salt_kek` disimpan di basis data (tidak rahasia). `dongle_secret` dibaca
    dari flashdisk. KEK hanya hidup di memori selama sesi; jangan ditulis ke disk.
    """
    if len(salt_kek) != SALT_LEN:
        raise ValueError(f"salt_kek harus {SALT_LEN} byte")
    if len(dongle_secret) != DONGLE_SECRET_LEN:
        raise ValueError(f"dongle_secret harus {DONGLE_SECRET_LEN} byte")

    stretched = hash_secret_raw(
        secret=_password_bytes(password),
        salt=salt_kek,
        time_cost=params.time_cost,
        memory_cost=params.memory_kib,
        parallelism=params.parallelism,
        hash_len=KEY_LEN,
        type=Type.ID,
    )
    # Dua potongan sama-sama 32 byte, jadi sambungan (||) tidak ambigu.
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=None, info=INFO_KEK).derive(
        stretched + dongle_secret
    )


def derive_subkey(kek: bytes, info: bytes) -> bytes:
    """Turunkan sub-kunci 32 byte dari KEK untuk satu fungsi (label `info`)."""
    if len(kek) != KEY_LEN:
        raise ValueError(f"KEK harus {KEY_LEN} byte")
    if not info:
        raise ValueError("label info tidak boleh kosong")
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=None, info=info).derive(kek)
