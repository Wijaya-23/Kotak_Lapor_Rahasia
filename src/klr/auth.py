"""Pendaftaran dan login tiga faktor (laporan subbab 3.1, 3.2, 3.2.1).

Urutan login yang dijalankan:

    1. DONGLE   cari flashdisk, muat dongle.key, cari pengguna dari dongle_id,
                cek penguncian akun, cek nomor seri + challenge-response RSA-PSS
    2. PASSWORD verifikasi hash Argon2id
    3. KEK      KEK = HKDF(Argon2id(password, kek_salt) || dongle_secret)
    4. WAJAH    buka template dengan sub-kunci KEK, bandingkan dengan webcam
    5. SESI     buka slot DEK, dan kunci privat enkripsi jika Admin

KEK dihitung SEBELUM cek wajah karena template wajah tersimpan terenkripsi dengan
sub-kunci KEK. Jika wajah gagal, KEK dibuang dan DEK tidak pernah dibuka. (Laporan
3.1 menulis KEK dihitung setelah ketiga faktor lolos; urutan di atas yang benar.)

Penguncian: 5 kali gagal berturut-turut pada faktor apa pun -> akun dikunci 15 menit.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa

from . import dongle as dg
from . import face, kdf, keys
from .db import ROLE_ADMIN, ROLE_PELAPOR, Database, UserRecord, new_id
from .errors import DecryptionError, KlrError

MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 15 * 60
MIN_PASSWORD_LEN = 8

STEP_DONGLE = "dongle"
STEP_LOCKED = "terkunci"
STEP_PASSWORD = "password"
STEP_FACE = "wajah"


class AuthError(KlrError):
    """Login atau pendaftaran gagal. `step` menyebut langkah yang gagal."""

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step


# --------------------------------------------------------------------------
# Sesi
# --------------------------------------------------------------------------
@dataclass
class Session:
    """Hasil login. Kunci hanya hidup di sini, di memori, dan dihapus saat logout.

    Catatan jujur: Python tidak menjamin salinan kunci di memori benar-benar hilang.
    logout() menimpa isi bytearray dan membuang referensi; itu upaya terbaik, bukan
    jaminan (risiko "pengambilan memori" di tabel ancaman 4.1).
    """

    user: UserRecord
    _kek: bytearray = field(repr=False)
    _dek: bytearray = field(repr=False)
    admin_private_key: rsa.RSAPrivateKey | None = field(default=None, repr=False)
    active: bool = True

    @property
    def is_admin(self) -> bool:
        return self.user.role == ROLE_ADMIN

    @property
    def kek(self) -> bytes:
        self._require_active()
        return bytes(self._kek)

    @property
    def dek(self) -> bytes:
        self._require_active()
        return bytes(self._dek)

    def _require_active(self) -> None:
        if not self.active:
            raise AuthError("sesi", "sesi sudah berakhir. Login lagi.")

    def logout(self) -> None:
        for buf in (self._kek, self._dek):
            for i in range(len(buf)):
                buf[i] = 0
        self.admin_private_key = None
        self.active = False


# --------------------------------------------------------------------------
# Pendaftaran
# --------------------------------------------------------------------------
def _check_new_user(db: Database, username: str, password: str) -> None:
    if not username.strip():
        raise ValueError("username tidak boleh kosong")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password minimal {MIN_PASSWORD_LEN} karakter")
    if db.get_user_by_username(username) is not None:
        raise ValueError("username sudah dipakai")


def _create_user(
    db: Database,
    *,
    role: str,
    username: str,
    password: str,
    dongle_root: str,
    face_samples: Sequence[np.ndarray],
    dek: bytes,
    params: kdf.Argon2Params,
    enc_private_key: rsa.RSAPrivateKey | None = None,
) -> UserRecord:
    _check_new_user(db, username, password)
    template = face.build_template(face_samples)  # gagal di sini = belum ada yang ditulis

    # dongle.key ditulis dulu (menolak flashdisk yang sudah berisi dongle).
    record = dg.create_dongle(dongle_root)
    dongle_path = os.path.join(dongle_root, dg.DONGLE_FILENAME)
    try:
        secret = dg.load_dongle(dongle_path).dongle_secret
        user_id = new_id()
        kek_salt = kdf.new_salt()
        kek = kdf.derive_kek(password, kek_salt, secret, params)
        user = UserRecord(
            id=user_id,
            username=username,
            role=role,
            pw_hash=kdf.hash_password(password, params),
            kek_salt=kek_salt,
            dongle_id=record.dongle_id,
            dongle_serial=record.volume_serial,
            sign_pubkey=record.public_key_pem,
            face_template_enc=keys.wrap_face_template(face.template_to_bytes(template), kek, user_id),
            enc_pubkey=None if enc_private_key is None else keys.encryption_public_key_pem(enc_private_key),
            enc_privkey_wrapped=None if enc_private_key is None else keys.wrap_private_key(enc_private_key, kek, user_id),
        )
        db.add_user(user, wrapped_dek=keys.wrap_dek(dek, kek, user_id))
    except BaseException:
        # Batalkan: jangan tinggalkan dongle.key yang tidak terdaftar di basis data.
        try:
            os.remove(dongle_path)
        except OSError:
            pass
        raise
    return user


def initialize_admin(
    db: Database,
    *,
    username: str,
    password: str,
    dongle_root: str,
    face_samples: Sequence[np.ndarray],
    params: kdf.Argon2Params = kdf.DEFAULT_PARAMS,
) -> UserRecord:
    """Inisialisasi sistem: Admin pertama, DEK, dan pasangan kunci enkripsi Admin dibuat."""
    if db.is_initialized():
        raise AuthError("inisialisasi", "sistem sudah diinisialisasi")
    return _create_user(
        db,
        role=ROLE_ADMIN,
        username=username,
        password=password,
        dongle_root=dongle_root,
        face_samples=face_samples,
        dek=keys.new_dek(),  # DEK dibuat sekali, di sini
        params=params,
        enc_private_key=keys.generate_encryption_keypair(),
    )


def register_pelapor(
    db: Database,
    admin: Session,
    *,
    username: str,
    password: str,
    dongle_root: str,
    face_samples: Sequence[np.ndarray],
    params: kdf.Argon2Params = kdf.DEFAULT_PARAMS,
) -> UserRecord:
    """Admin yang sedang login mendaftarkan Pelapor. Admin memegang DEK di memori,
    lalu membungkusnya dengan KEK Pelapor baru menjadi slot baru (laporan 3.2.1)."""
    if not admin.active or not admin.is_admin:
        raise AuthError("izin", "hanya Admin yang sedang login yang bisa mendaftarkan pengguna")
    return _create_user(
        db,
        role=ROLE_PELAPOR,
        username=username,
        password=password,
        dongle_root=dongle_root,
        face_samples=face_samples,
        dek=admin.dek,
        params=params,
    )


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------
def _record_failure(db: Database, user: UserRecord, now: float) -> str:
    count = user.failed_logins + 1
    if count >= MAX_FAILED_LOGINS:
        db.set_login_state(user.id, 0, now + LOCK_SECONDS)
        return f" Akun dikunci {LOCK_SECONDS // 60} menit."
    db.set_login_state(user.id, count, None)
    return f" Sisa percobaan: {MAX_FAILED_LOGINS - count}."


def login(
    db: Database,
    *,
    password: str,
    capture_face: Callable[[], np.ndarray],
    dongle_roots: Iterable[str] | None = None,
    issuer: dg.ChallengeIssuer | None = None,
    clock: Callable[[], float] = time.time,
    params: kdf.Argon2Params = kdf.DEFAULT_PARAMS,
    threshold: float = face.DEFAULT_THRESHOLD,
) -> Session:
    """Login tiga faktor. Berhasil -> Session. Gagal -> AuthError dengan .step.

    `capture_face` dipanggil HANYA jika dongle dan password sudah lolos, dan harus
    mengembalikan embedding wajah (atau melempar FaceError).
    """
    # 1. Dongle: menemukan dan mengenali pengguna
    try:
        path = dg.require_single_dongle(dongle_roots)
        dongle = dg.load_dongle(path)
    except dg.DongleError as exc:
        raise AuthError(STEP_DONGLE, str(exc)) from None
    user = db.get_user_by_dongle(dongle.dongle_id)
    if user is None:
        raise AuthError(STEP_DONGLE, "dongle tidak terdaftar")

    now = clock()
    if user.locked_until is not None and now < user.locked_until:
        menit = max(1, int((user.locked_until - now + 59) // 60))
        raise AuthError(STEP_LOCKED, f"akun terkunci. Coba lagi dalam {menit} menit.")

    record = dg.DongleRecord(user.dongle_id, user.sign_pubkey, user.dongle_serial)
    try:
        dg.authenticate_dongle(dongle, record, issuer or dg.ChallengeIssuer(), dg.read_volume_serial(path))
    except dg.DongleError as exc:
        raise AuthError(STEP_DONGLE, str(exc) + _record_failure(db, user, now)) from None

    # 2. Password
    if not kdf.verify_password(user.pw_hash, password):
        raise AuthError(STEP_PASSWORD, "password salah." + _record_failure(db, user, now))

    # 3. KEK (password dan dongle_secret sama-sama dibutuhkan)
    kek = bytearray(kdf.derive_kek(password, user.kek_salt, dongle.dongle_secret, params))

    # 4. Wajah
    try:
        template = face.template_from_bytes(keys.unwrap_face_template(user.face_template_enc, bytes(kek), user.id))
        embedding = capture_face()
        ok, distance = face.verify(template, embedding, threshold)
    except (face.FaceError, DecryptionError) as exc:
        kek[:] = bytes(len(kek))
        raise AuthError(STEP_FACE, f"verifikasi wajah gagal: {exc}." + _record_failure(db, user, now)) from None
    if not ok:
        kek[:] = bytes(len(kek))
        raise AuthError(STEP_FACE, f"wajah tidak cocok (jarak {distance:.3f})." + _record_failure(db, user, now))

    # 5. Sesi: semua faktor lolos
    db.set_login_state(user.id, 0, None)
    slot = db.get_key_slot(user.id)
    dek = keys.unwrap_dek(slot, bytes(kek), user.id)
    admin_key = None
    if user.role == ROLE_ADMIN:
        admin_key = keys.unwrap_private_key(user.enc_privkey_wrapped, bytes(kek), user.id)
    return Session(user=db.get_user(user.id), _kek=kek, _dek=bytearray(dek), admin_private_key=admin_key)
