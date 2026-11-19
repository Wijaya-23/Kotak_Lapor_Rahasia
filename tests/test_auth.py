"""Tes login tiga faktor dan pendaftaran. Flashdisk = folder sementara, webcam = embedding tiruan."""
import os
import sqlite3

import numpy as np
import pytest

from klr import auth, keys, textcrypto
from klr import dongle as dg
from klr.auth import AuthError
from klr.db import Database, new_id
from klr.face import FaceError
from klr.kdf import Argon2Params

FAST = Argon2Params(memory_kib=1024, time_cost=1, parallelism=1)
rng = np.random.default_rng(7)


def _unit(v):
    return v / np.linalg.norm(v)


WAJAH_ADMIN = _unit(rng.normal(size=128))
WAJAH_PELAPOR = _unit(rng.normal(size=128))
WAJAH_ORANG_LAIN = _unit(rng.normal(size=128))
PW_ADMIN = "Admin#Kuat2026"
PW_PELAPOR = "Pelapor!2026"


def _sample(base):
    return base + rng.normal(scale=0.02, size=128)


def _samples(base):
    return [_sample(base) for _ in range(5)]


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


class Camera:
    """Webcam tiruan: mencatat berapa kali dipanggil dan wajah siapa yang 'terlihat'."""

    def __init__(self, base):
        self.base = base
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return _sample(self.base)


@pytest.fixture
def env(tmp_path):
    roots = {name: tmp_path / name for name in ("fd_admin", "fd_pelapor", "fd_kosong", "fd_lain")}
    for p in roots.values():
        p.mkdir()
    with Database(str(tmp_path / "klr.db")) as db:
        yield db, {k: str(v) for k, v in roots.items()}, Clock()


@pytest.fixture
def admin_ready(env):
    db, roots, clock = env
    auth.initialize_admin(db, username="admin", password=PW_ADMIN, dongle_root=roots["fd_admin"],
                          face_samples=_samples(WAJAH_ADMIN), params=FAST)
    return env


def _login(db, root, password, wajah, clock, camera=None):
    return auth.login(db, password=password, capture_face=camera or Camera(wajah), dongle_roots=[root],
                      clock=clock, params=FAST)


# ------------------------------------------------------------ inisialisasi
def test_inisialisasi_admin(admin_ready):
    db, roots, _ = admin_ready
    admin = db.get_admin()
    assert db.is_initialized()
    assert admin.username == "admin"
    assert admin.enc_pubkey and admin.enc_privkey_wrapped
    assert db.get_key_slot(admin.id) is not None
    assert os.path.isfile(os.path.join(roots["fd_admin"], dg.DONGLE_FILENAME))


def test_inisialisasi_hanya_sekali(admin_ready):
    db, roots, _ = admin_ready
    with pytest.raises(AuthError):
        auth.initialize_admin(db, username="admin2", password=PW_ADMIN, dongle_root=roots["fd_lain"],
                              face_samples=_samples(WAJAH_ADMIN), params=FAST)


@pytest.mark.parametrize(
    "kw, err",
    [
        ({"password": "pendek"}, ValueError),
        ({"username": "  "}, ValueError),
        ({"face_samples": _samples(WAJAH_ADMIN)[:4] + [_sample(WAJAH_ORANG_LAIN)]}, FaceError),
    ],
    ids=["password-pendek", "username-kosong", "wajah-tidak-konsisten"],
)
def test_input_tidak_valid_tidak_menulis_apa_pun(env, kw, err):
    db, roots, _ = env
    args = dict(username="admin", password=PW_ADMIN, dongle_root=roots["fd_admin"],
                face_samples=_samples(WAJAH_ADMIN), params=FAST)
    args.update(kw)
    with pytest.raises(err):
        auth.initialize_admin(db, **args)
    assert not os.listdir(roots["fd_admin"])
    assert not db.is_initialized()


def test_gagal_simpan_ke_db_menghapus_dongle_yang_baru_dibuat(env, monkeypatch):
    db, roots, _ = env

    def gagal(*a, **k):
        raise sqlite3.OperationalError("disk penuh")

    monkeypatch.setattr(db, "add_user", gagal)
    with pytest.raises(sqlite3.OperationalError):
        auth.initialize_admin(db, username="admin", password=PW_ADMIN, dongle_root=roots["fd_admin"],
                              face_samples=_samples(WAJAH_ADMIN), params=FAST)
    assert not os.listdir(roots["fd_admin"])


def test_flashdisk_yang_sudah_berisi_dongle_tidak_disentuh(env):
    db, roots, _ = env
    dg.create_dongle(roots["fd_admin"])
    path = os.path.join(roots["fd_admin"], dg.DONGLE_FILENAME)
    isi = open(path, "rb").read()
    with pytest.raises(dg.DongleError):
        auth.initialize_admin(db, username="admin", password=PW_ADMIN, dongle_root=roots["fd_admin"],
                              face_samples=_samples(WAJAH_ADMIN), params=FAST)
    assert open(path, "rb").read() == isi


# ------------------------------------------------------------ login
def test_login_admin_berhasil(admin_ready):
    db, roots, clock = admin_ready
    s = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    assert s.is_admin and s.active
    assert len(s.dek) == 32
    pub = keys.load_encryption_public_key(db.get_admin().enc_pubkey)
    assert s.admin_private_key.public_key().public_numbers() == pub.public_numbers()


def test_password_salah_ditolak_dan_wajah_tidak_diminta(admin_ready):
    db, roots, clock = admin_ready
    cam = Camera(WAJAH_ADMIN)
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_admin"], "salah-sekali", WAJAH_ADMIN, clock, cam)
    assert ei.value.step == "password"
    assert cam.calls == 0
    assert db.get_admin().failed_logins == 1


def test_wajah_orang_lain_ditolak(admin_ready):
    """Skenario uji nomor 1: dongle dan password benar, wajah orang lain."""
    db, roots, clock = admin_ready
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ORANG_LAIN, clock)
    assert ei.value.step == "wajah"
    assert db.get_admin().failed_logins == 1


def test_kamera_dibatalkan_dihitung_gagal(admin_ready):
    db, roots, clock = admin_ready

    def batal():
        raise FaceError("pengambilan wajah dibatalkan")

    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_admin"], PW_ADMIN, None, clock, batal)
    assert ei.value.step == "wajah"


def test_tanpa_dongle_ditolak(admin_ready):
    """Skenario uji nomor 1: dongle dicabut."""
    db, roots, clock = admin_ready
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_kosong"], PW_ADMIN, WAJAH_ADMIN, clock)
    assert ei.value.step == "dongle"


def test_dongle_tidak_terdaftar_ditolak(admin_ready):
    db, roots, clock = admin_ready
    dg.create_dongle(roots["fd_lain"])
    with pytest.raises(AuthError, match="tidak terdaftar"):
        _login(db, roots["fd_lain"], PW_ADMIN, WAJAH_ADMIN, clock)


def test_dongle_salinan_dengan_nomor_seri_lain_ditolak(env, monkeypatch):
    db, roots, clock = env
    monkeypatch.setattr(dg, "read_volume_serial", lambda path: "AAAA-0001")
    auth.initialize_admin(db, username="admin", password=PW_ADMIN, dongle_root=roots["fd_admin"],
                          face_samples=_samples(WAJAH_ADMIN), params=FAST)
    monkeypatch.setattr(dg, "read_volume_serial", lambda path: "BBBB-0002")
    with pytest.raises(AuthError, match="nomor seri") as ei:
        _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    assert ei.value.step == "dongle"
    assert db.get_admin().failed_logins == 1


# ------------------------------------------------------------ penguncian
def test_lima_gagal_mengunci_15_menit_lalu_terbuka(admin_ready):
    db, roots, clock = admin_ready
    for i in range(4):
        with pytest.raises(AuthError, match="Sisa percobaan"):
            _login(db, roots["fd_admin"], "salah-salah", WAJAH_ADMIN, clock)
    with pytest.raises(AuthError, match="dikunci 15 menit"):
        _login(db, roots["fd_admin"], "salah-salah", WAJAH_ADMIN, clock)

    # Selama terkunci, kredensial benar pun ditolak, dan wajah tidak diminta.
    cam = Camera(WAJAH_ADMIN)
    clock.now += 14 * 60
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock, cam)
    assert ei.value.step == "terkunci"
    assert cam.calls == 0

    clock.now += 60 + 1
    s = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    assert s.user.failed_logins == 0 and s.user.locked_until is None


def test_gagal_campuran_faktor_ikut_dihitung(admin_ready):
    db, roots, clock = admin_ready
    for _ in range(2):
        with pytest.raises(AuthError):
            _login(db, roots["fd_admin"], "salah-salah", WAJAH_ADMIN, clock)
    for _ in range(3):
        with pytest.raises(AuthError):
            _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ORANG_LAIN, clock)
    assert db.get_admin().locked_until is not None


def test_login_berhasil_mengatur_ulang_hitungan(admin_ready):
    db, roots, clock = admin_ready
    for _ in range(3):
        with pytest.raises(AuthError):
            _login(db, roots["fd_admin"], "salah-salah", WAJAH_ADMIN, clock)
    _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    assert db.get_admin().failed_logins == 0


# ------------------------------------------------------------ Pelapor
@pytest.fixture
def pelapor_ready(admin_ready):
    db, roots, clock = admin_ready
    admin = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    auth.register_pelapor(db, admin, username="pelapor1", password=PW_PELAPOR, dongle_root=roots["fd_pelapor"],
                          face_samples=_samples(WAJAH_PELAPOR), params=FAST)
    return db, roots, clock, admin


def test_pelapor_login_dan_mendapat_dek_yang_sama(pelapor_ready):
    db, roots, clock, admin = pelapor_ready
    p = _login(db, roots["fd_pelapor"], PW_PELAPOR, WAJAH_PELAPOR, clock)
    assert not p.is_admin
    assert p.dek == admin.dek
    assert p.admin_private_key is None


def test_wajah_admin_tidak_bisa_login_sebagai_pelapor(pelapor_ready):
    db, roots, clock, _ = pelapor_ready
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_pelapor"], PW_PELAPOR, WAJAH_ADMIN, clock)
    assert ei.value.step == "wajah"


def test_password_admin_dengan_dongle_pelapor_ditolak(pelapor_ready):
    db, roots, clock, _ = pelapor_ready
    with pytest.raises(AuthError) as ei:
        _login(db, roots["fd_pelapor"], PW_ADMIN, WAJAH_PELAPOR, clock)
    assert ei.value.step == "password"


def test_hanya_admin_aktif_yang_bisa_mendaftarkan(pelapor_ready):
    db, roots, clock, admin = pelapor_ready
    p = _login(db, roots["fd_pelapor"], PW_PELAPOR, WAJAH_PELAPOR, clock)
    with pytest.raises(AuthError):
        auth.register_pelapor(db, p, username="x", password=PW_PELAPOR, dongle_root=roots["fd_lain"],
                              face_samples=_samples(WAJAH_ORANG_LAIN), params=FAST)
    admin.logout()
    with pytest.raises(AuthError):
        auth.register_pelapor(db, admin, username="x", password=PW_PELAPOR, dongle_root=roots["fd_lain"],
                              face_samples=_samples(WAJAH_ORANG_LAIN), params=FAST)
    assert not os.listdir(roots["fd_lain"])


def test_username_ganda_ditolak_tanpa_membuat_dongle(pelapor_ready):
    db, roots, clock, admin = pelapor_ready
    with pytest.raises(ValueError, match="sudah dipakai"):
        auth.register_pelapor(db, admin, username="pelapor1", password=PW_PELAPOR, dongle_root=roots["fd_lain"],
                              face_samples=_samples(WAJAH_ORANG_LAIN), params=FAST)
    assert not os.listdir(roots["fd_lain"])


# ------------------------------------------------------------ alur penuh dan sesi
def test_pelapor_kirim_laporan_admin_membaca(pelapor_ready):
    db, roots, clock, _ = pelapor_ready
    p = _login(db, roots["fd_pelapor"], PW_PELAPOR, WAJAH_PELAPOR, clock)
    rid = new_id()
    pub = keys.load_encryption_public_key(db.get_admin().enc_pubkey)
    pkg = textcrypto.encrypt_text("Ada pungutan liar di loket 3.", pub, rid, p.user.id)
    db.add_report(p.dek, report_id=rid, reporter_id=p.user.id, kind="teks", body_pkg=pkg)
    p.logout()

    a = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    [r] = db.list_reports(a.dek)
    assert textcrypto.decrypt_text(r.body_pkg, a.admin_private_key, r.id, r.reporter_id) == "Ada pungutan liar di loket 3."


def test_logout_menghapus_kunci(admin_ready):
    db, roots, clock = admin_ready
    s = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    s.logout()
    assert not s.active
    assert s.admin_private_key is None
    assert set(s._kek) == {0} and set(s._dek) == {0}
    with pytest.raises(AuthError):
        _ = s.dek
    with pytest.raises(AuthError):
        _ = s.kek


def test_kunci_tidak_tampil_saat_sesi_dicetak(admin_ready):
    db, roots, clock = admin_ready
    s = _login(db, roots["fd_admin"], PW_ADMIN, WAJAH_ADMIN, clock)
    teks = repr(s)
    assert "_dek" not in teks and "_kek" not in teks and "admin_private_key" not in teks
