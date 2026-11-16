"""Dongle simulasi: flashdisk berisi berkas dongle.key (laporan subbab 3.1 dan 3.1.1).

Isi dongle.key (JSON):
    format         "klr-dongle-v1"
    dongle_id      UUID, dicocokkan dengan basis data
    dongle_secret  32 byte acak (base64), bahan KEK. TIDAK PERNAH disimpan di basis data
    signing_key    kunci privat RSA-2048 (PEM), untuk challenge login dan tanda tangan blok

Faktor dongle saat login:
    1. Cari flashdisk (drive removable) yang berisi dongle.key.
    2. Cocokkan dongle_id dengan yang terdaftar.
    3. Cocokkan nomor seri volume flashdisk dengan yang tercatat saat pendaftaran.
       (Menangkal penyalinan dongle.key ke flashdisk lain.)
    4. Challenge-response: aplikasi membuat nonce acak 32 byte, dongle menandatanganinya
       dengan RSA-PSS (SHA-256), aplikasi memverifikasi dengan kunci publik di basis data.
       Nonce hanya berlaku sekali dan kedaluwarsa, sehingga jawaban lama tidak bisa diputar ulang.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

import psutil
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .errors import KlrError
from .kdf import DONGLE_SECRET_LEN

DONGLE_FILENAME = "dongle.key"
FORMAT_TAG = "klr-dongle-v1"
RSA_BITS = 2048
CHALLENGE_LEN = 32
CHALLENGE_TTL_SECONDS = 60.0

# Konteks tanda tangan. Kunci yang sama juga dipakai menandatangani blok blockchain,
# jadi yang ditandatangani bukan nonce mentah, melainkan KONTEKS || nonce. Dengan begitu
# tanda tangan login tidak bisa dipakai ulang sebagai tanda tangan blok, dan sebaliknya.
LOGIN_CONTEXT = b"klr-login-challenge-v1\x00"

_PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)


class DongleError(KlrError):
    """Faktor dongle gagal (tidak ada, rusak, tidak terdaftar, atau tidak cocok)."""


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Dongle:
    """Isi dongle.key yang sudah dimuat. Rahasia tidak ikut tampil di repr/print."""

    dongle_id: str
    dongle_secret: bytes = field(repr=False)
    signing_key: rsa.RSAPrivateKey = field(repr=False)
    path: str = ""

    def sign_challenge(self, challenge: bytes) -> bytes:
        """Tandatangani nonce login (RSA-PSS, SHA-256)."""
        return self.signing_key.sign(LOGIN_CONTEXT + challenge, _PSS, hashes.SHA256())

    def public_key_pem(self) -> str:
        return public_key_to_pem(self.signing_key.public_key())


@dataclass(frozen=True)
class DongleRecord:
    """Data dongle yang disimpan di basis data (tabel users). Tidak memuat rahasia."""

    dongle_id: str
    public_key_pem: str
    volume_serial: str | None


# --------------------------------------------------------------------------
# Kunci publik dan verifikasi tanda tangan
# --------------------------------------------------------------------------
def public_key_to_pem(key: rsa.RSAPublicKey) -> str:
    return key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


def public_key_from_pem(pem: str) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(pem.encode("ascii"))
    if not isinstance(key, rsa.RSAPublicKey):
        raise DongleError("kunci publik bukan RSA")
    return key


def verify_challenge_signature(public_key: rsa.RSAPublicKey, challenge: bytes, signature: bytes) -> bool:
    try:
        public_key.verify(signature, LOGIN_CONTEXT + challenge, _PSS, hashes.SHA256())
        return True
    except InvalidSignature:
        return False


# --------------------------------------------------------------------------
# Nonce sekali pakai
# --------------------------------------------------------------------------
class ChallengeIssuer:
    """Membuat nonce login dan memastikan tiap nonce hanya bisa dipakai sekali.

    `clock` bisa diganti saat pengujian untuk mensimulasikan waktu berlalu.
    """

    def __init__(self, ttl_seconds: float = CHALLENGE_TTL_SECONDS, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._clock = clock
        self._pending: dict[bytes, float] = {}

    def issue(self) -> bytes:
        challenge = os.urandom(CHALLENGE_LEN)
        self._pending[challenge] = self._clock()
        return challenge

    def verify_and_consume(self, challenge: bytes, signature: bytes, public_key: rsa.RSAPublicKey) -> bool:
        issued_at = self._pending.pop(challenge, None)  # dihapus apa pun hasilnya: sekali pakai
        if issued_at is None:
            return False
        if self._clock() - issued_at > self._ttl:
            return False
        return verify_challenge_signature(public_key, challenge, signature)


# --------------------------------------------------------------------------
# Flashdisk dan nomor seri
# --------------------------------------------------------------------------
def list_removable_drives() -> list[str]:
    """Daftar akar drive removable, misalnya ['E:\\\\']. (Windows; di OS lain biasanya kosong.)"""
    return [p.mountpoint for p in psutil.disk_partitions(all=False) if "removable" in p.opts.split(",")]


def read_volume_serial(path: str) -> str | None:
    """Nomor seri volume tempat `path` berada, format 'XXXX-XXXX' seperti perintah `vol E:`.

    Ini nomor seri sistem berkas (berubah jika flashdisk diformat ulang), bukan nomor seri
    perangkat keras USB. Di luar Windows mengembalikan None.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if not drive:
        return None
    serial = wintypes.DWORD()
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive + "\\"), None, 0, ctypes.byref(serial), None, None, None, 0
    )
    if not ok:
        return None
    return f"{serial.value >> 16:04X}-{serial.value & 0xFFFF:04X}"


def find_dongles(roots: Iterable[str] | None = None) -> list[str]:
    """Cari dongle.key di akar tiap drive. Bawaan: semua drive removable."""
    if roots is None:
        roots = list_removable_drives()
    found = []
    for root in roots:
        candidate = os.path.join(root, DONGLE_FILENAME)
        try:
            if os.path.isfile(candidate):
                found.append(candidate)
        except OSError:  # misalnya pembaca kartu tanpa kartu
            continue
    return found


def require_single_dongle(roots: Iterable[str] | None = None) -> str:
    found = find_dongles(roots)
    if not found:
        raise DongleError("dongle tidak ditemukan. Tancapkan flashdisk dongle.")
    if len(found) > 1:
        raise DongleError("lebih dari satu dongle tertancap. Cabut yang tidak dipakai.")
    return found[0]


# --------------------------------------------------------------------------
# Membuat dan memuat dongle.key
# --------------------------------------------------------------------------
def create_dongle(root: str, overwrite: bool = False) -> DongleRecord:
    """Buat dongle.key baru di `root` (akar flashdisk). Dipakai saat pendaftaran pengguna.

    Mengembalikan data PUBLIK untuk disimpan di basis data. dongle_secret tidak ikut.
    """
    target = os.path.join(root, DONGLE_FILENAME)
    if os.path.exists(target) and not overwrite:
        raise DongleError("flashdisk ini sudah berisi dongle.key. Pakai flashdisk lain.")

    serial = read_volume_serial(root)
    if os.name == "nt" and serial is None:
        # Tanpa nomor seri, pemeriksaan salinan akan terlewati selamanya. Lebih baik gagal sekarang.
        raise DongleError("nomor seri flashdisk tidak terbaca")

    dongle_id = str(uuid.uuid4())
    secret = os.urandom(DONGLE_SECRET_LEN)
    key = rsa.generate_private_key(public_exponent=65537, key_size=RSA_BITS)
    payload = {
        "format": FORMAT_TAG,
        "dongle_id": dongle_id,
        "dongle_secret": base64.b64encode(secret).decode("ascii"),
        "signing_key": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode("ascii"),
    }

    # Tulis ke berkas sementara lalu ganti nama, supaya dongle.key tidak pernah setengah jadi.
    # fsync memastikan data benar-benar tertulis sebelum flashdisk dicabut.
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)

    return DongleRecord(
        dongle_id=dongle_id,
        public_key_pem=public_key_to_pem(key.public_key()),
        volume_serial=serial,
    )


def load_dongle(path: str) -> Dongle:
    """Muat dan validasi dongle.key. Berkas rusak atau tidak lengkap menjadi DongleError."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("format") != FORMAT_TAG:
            raise ValueError("format tidak dikenal")
        dongle_id = str(uuid.UUID(data["dongle_id"]))
        secret = base64.b64decode(data["dongle_secret"], validate=True)
        if len(secret) != DONGLE_SECRET_LEN:
            raise ValueError("panjang dongle_secret salah")
        key = serialization.load_pem_private_key(data["signing_key"].encode("ascii"), password=None)
        if not isinstance(key, rsa.RSAPrivateKey) or key.key_size != RSA_BITS:
            raise ValueError("kunci bukan RSA-2048")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, binascii.Error) as exc:
        raise DongleError(f"berkas dongle rusak atau tidak valid ({exc.__class__.__name__})") from None
    return Dongle(dongle_id=dongle_id, dongle_secret=secret, signing_key=key, path=path)


# --------------------------------------------------------------------------
# Faktor dongle lengkap
# --------------------------------------------------------------------------
def authenticate_dongle(
    dongle: Dongle,
    record: DongleRecord,
    issuer: ChallengeIssuer,
    actual_serial: str | None,
) -> None:
    """Jalankan seluruh pemeriksaan faktor dongle. Lolos = tidak melempar error.

    `record` diambil dari basis data berdasarkan dongle.dongle_id.
    `actual_serial` adalah read_volume_serial() dari flashdisk yang sedang tertancap.
    """
    if dongle.dongle_id != record.dongle_id:
        raise DongleError("dongle tidak terdaftar")

    # Nomor seri dicek jika tercatat saat pendaftaran (selalu, di Windows).
    if record.volume_serial is not None and actual_serial != record.volume_serial:
        raise DongleError("nomor seri flashdisk tidak cocok. Dongle mungkin hasil salinan.")

    challenge = issuer.issue()
    signature = dongle.sign_challenge(challenge)
    if not issuer.verify_and_consume(challenge, signature, public_key_from_pem(record.public_key_pem)):
        raise DongleError("challenge-response gagal: kunci di dongle tidak cocok dengan yang terdaftar")
