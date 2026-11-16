import base64
import dataclasses
import json
import os
import re
import shutil
from collections import namedtuple

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from klr import dongle as dg
from klr.dongle import ChallengeIssuer, DongleError

# Tes memakai folder sementara sebagai "flashdisk", jadi bisa jalan tanpa flashdisk asli.


@pytest.fixture
def drive(tmp_path):
    root = tmp_path / "flashdisk"
    root.mkdir()
    return str(root)


@pytest.fixture
def registered(drive):
    """Dongle terdaftar. Nomor seri dipaksa ke nilai tetap supaya tes sama di semua OS."""
    record = dg.create_dongle(drive)
    record = dataclasses.replace(record, volume_serial="1A2B-3C4D")
    return dg.load_dongle(os.path.join(drive, dg.DONGLE_FILENAME)), record


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ------------------------------------------------------------ membuat dongle
def test_create_dan_load_dongle(drive):
    record = dg.create_dongle(drive)
    d = dg.load_dongle(os.path.join(drive, dg.DONGLE_FILENAME))
    assert d.dongle_id == record.dongle_id
    assert len(d.dongle_secret) == 32
    assert d.signing_key.key_size == 2048
    assert d.public_key_pem() == record.public_key_pem


def test_record_untuk_basis_data_tidak_memuat_rahasia(drive):
    record = dg.create_dongle(drive)
    d = dg.load_dongle(os.path.join(drive, dg.DONGLE_FILENAME))
    isi = json.dumps(dataclasses.asdict(record))
    assert base64.b64encode(d.dongle_secret).decode() not in isi
    assert "PRIVATE KEY" not in isi


def test_rahasia_tidak_tampil_saat_dicetak(registered):
    d, _ = registered
    teks = repr(d)
    assert repr(d.dongle_secret) not in teks  # bentuk b'...' yang dicetak Python
    assert "dongle_secret" not in teks
    assert "signing_key" not in teks
    assert d.dongle_id in teks  # yang tidak rahasia tetap tampil


def test_isi_berkas_sesuai_laporan(drive):
    dg.create_dongle(drive)
    with open(os.path.join(drive, dg.DONGLE_FILENAME), encoding="utf-8") as fh:
        data = json.load(fh)
    assert set(data) == {"format", "dongle_id", "dongle_secret", "signing_key"}
    assert data["format"] == "klr-dongle-v1"
    assert len(base64.b64decode(data["dongle_secret"])) == 32
    assert "BEGIN PRIVATE KEY" in data["signing_key"]


def test_tidak_menimpa_dongle_yang_sudah_ada(drive):
    first = dg.create_dongle(drive)
    with pytest.raises(DongleError):
        dg.create_dongle(drive)
    second = dg.create_dongle(drive, overwrite=True)
    assert second.dongle_id != first.dongle_id
    assert not os.path.exists(os.path.join(drive, dg.DONGLE_FILENAME + ".tmp"))


def test_dua_dongle_berbeda_seluruhnya(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    ra, rb = dg.create_dongle(str(a)), dg.create_dongle(str(b))
    da = dg.load_dongle(str(a / dg.DONGLE_FILENAME))
    db = dg.load_dongle(str(b / dg.DONGLE_FILENAME))
    assert ra.dongle_id != rb.dongle_id
    assert da.dongle_secret != db.dongle_secret
    assert ra.public_key_pem != rb.public_key_pem


# ------------------------------------------------------------ berkas rusak
def _tulis(drive, data):
    path = os.path.join(drive, dg.DONGLE_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data if isinstance(data, str) else json.dumps(data))
    return path


def _payload_valid(drive):
    dg.create_dongle(drive)
    with open(os.path.join(drive, dg.DONGLE_FILENAME), encoding="utf-8") as fh:
        return json.load(fh)


def _pem(key):
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


@pytest.mark.parametrize(
    "ubah",
    [
        lambda p: "bukan json",
        lambda p: [],
        lambda p: {**p, "format": "klr-dongle-v0"},
        lambda p: {k: v for k, v in p.items() if k != "dongle_secret"},
        lambda p: {**p, "dongle_id": "bukan-uuid"},
        lambda p: {**p, "dongle_id": 123},
        lambda p: {**p, "dongle_secret": base64.b64encode(bytes(31)).decode()},
        lambda p: {**p, "dongle_secret": "!!!bukan base64!!!"},
        lambda p: {**p, "signing_key": "-----BEGIN PRIVATE KEY-----\nrusak\n-----END PRIVATE KEY-----\n"},
        lambda p: {**p, "signing_key": _pem(rsa.generate_private_key(public_exponent=65537, key_size=1024))},
        lambda p: {**p, "signing_key": _pem(ec.generate_private_key(ec.SECP256R1()))},
    ],
    ids=["bukan-json", "bukan-objek", "versi-salah", "tanpa-secret", "id-rusak", "id-angka",
         "secret-31-byte", "secret-bukan-base64", "pem-rusak", "rsa-1024", "kunci-ec"],
)
def test_berkas_rusak_ditolak(drive, ubah):
    payload = _payload_valid(drive)
    path = _tulis(drive, ubah(payload))
    with pytest.raises(DongleError):
        dg.load_dongle(path)


def test_berkas_tidak_ada_ditolak(drive):
    with pytest.raises(DongleError):
        dg.load_dongle(os.path.join(drive, dg.DONGLE_FILENAME))


# ------------------------------------------------------------ mencari dongle
def test_find_dongles_di_beberapa_drive(tmp_path):
    kosong, berisi = tmp_path / "e", tmp_path / "f"
    kosong.mkdir(), berisi.mkdir()
    dg.create_dongle(str(berisi))
    tidak_ada = str(tmp_path / "g-tidak-ada")
    assert dg.find_dongles([str(kosong), str(berisi), tidak_ada]) == [str(berisi / dg.DONGLE_FILENAME)]


def test_require_single_dongle(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    with pytest.raises(DongleError, match="tidak ditemukan"):
        dg.require_single_dongle([str(a), str(b)])
    dg.create_dongle(str(a))
    assert dg.require_single_dongle([str(a), str(b)]).endswith(dg.DONGLE_FILENAME)
    dg.create_dongle(str(b))
    with pytest.raises(DongleError, match="lebih dari satu"):
        dg.require_single_dongle([str(a), str(b)])


Partition = namedtuple("Partition", "device mountpoint fstype opts")


def test_list_removable_drives_hanya_yang_removable(monkeypatch):
    fake = [
        Partition("C:\\", "C:\\", "NTFS", "rw,fixed"),
        Partition("D:\\", "D:\\", "NTFS", "rw,fixed"),
        Partition("E:\\", "E:\\", "FAT32", "rw,removable"),
        Partition("F:\\", "F:\\", "", "cdrom"),
    ]
    monkeypatch.setattr(dg.psutil, "disk_partitions", lambda all=False: fake)
    assert dg.list_removable_drives() == ["E:\\"]


def test_find_dongles_bawaan_memakai_drive_removable(monkeypatch, drive):
    dg.create_dongle(drive)
    monkeypatch.setattr(dg, "list_removable_drives", lambda: [drive])
    assert dg.find_dongles() == [os.path.join(drive, dg.DONGLE_FILENAME)]


# ------------------------------------------------------------ challenge-response
def test_tanda_tangan_challenge_valid(registered):
    d, record = registered
    ch = os.urandom(32)
    sig = d.sign_challenge(ch)
    pub = dg.public_key_from_pem(record.public_key_pem)
    assert dg.verify_challenge_signature(pub, ch, sig)


def test_tanda_tangan_ditolak_jika_challenge_atau_kunci_berbeda(registered, tmp_path):
    d, record = registered
    ch = os.urandom(32)
    sig = d.sign_challenge(ch)
    pub = dg.public_key_from_pem(record.public_key_pem)
    assert not dg.verify_challenge_signature(pub, os.urandom(32), sig)
    lain = tmp_path / "lain"
    lain.mkdir()
    pub_lain = dg.public_key_from_pem(dg.create_dongle(str(lain)).public_key_pem)
    assert not dg.verify_challenge_signature(pub_lain, ch, sig)


def test_tanda_tangan_tanpa_konteks_login_ditolak(registered):
    """Tanda tangan atas nonce mentah (misalnya dari konteks lain) tidak diterima sebagai jawaban login."""
    from cryptography.hazmat.primitives import hashes

    d, record = registered
    ch = os.urandom(32)
    sig_mentah = d.signing_key.sign(ch, dg._PSS, hashes.SHA256())
    assert not dg.verify_challenge_signature(dg.public_key_from_pem(record.public_key_pem), ch, sig_mentah)


def test_nonce_acak_32_byte():
    issuer = ChallengeIssuer()
    a, b = issuer.issue(), issuer.issue()
    assert len(a) == len(b) == 32
    assert a != b


def test_nonce_tidak_bisa_diputar_ulang(registered):
    d, record = registered
    pub = dg.public_key_from_pem(record.public_key_pem)
    issuer = ChallengeIssuer()
    ch = issuer.issue()
    sig = d.sign_challenge(ch)
    assert issuer.verify_and_consume(ch, sig, pub) is True
    assert issuer.verify_and_consume(ch, sig, pub) is False  # rekaman jawaban lama


def test_nonce_yang_tidak_pernah_dibuat_ditolak(registered):
    d, record = registered
    ch = os.urandom(32)
    assert ChallengeIssuer().verify_and_consume(ch, d.sign_challenge(ch), dg.public_key_from_pem(record.public_key_pem)) is False


def test_nonce_kedaluwarsa_ditolak(registered):
    d, record = registered
    clock = FakeClock()
    issuer = ChallengeIssuer(ttl_seconds=60, clock=clock)
    ch = issuer.issue()
    clock.now += 61
    assert issuer.verify_and_consume(ch, d.sign_challenge(ch), dg.public_key_from_pem(record.public_key_pem)) is False


def test_nonce_hangus_walau_tanda_tangan_salah(registered):
    d, record = registered
    pub = dg.public_key_from_pem(record.public_key_pem)
    issuer = ChallengeIssuer()
    ch = issuer.issue()
    assert issuer.verify_and_consume(ch, b"\x00" * 256, pub) is False
    assert issuer.verify_and_consume(ch, d.sign_challenge(ch), pub) is False


# ------------------------------------------------------------ faktor dongle lengkap
def test_authenticate_lolos(registered):
    d, record = registered
    dg.authenticate_dongle(d, record, ChallengeIssuer(), actual_serial="1A2B-3C4D")


def test_authenticate_dongle_id_tidak_terdaftar(registered):
    d, record = registered
    lain = dataclasses.replace(record, dongle_id="00000000-0000-4000-8000-000000000000")
    with pytest.raises(DongleError, match="tidak terdaftar"):
        dg.authenticate_dongle(d, lain, ChallengeIssuer(), actual_serial="1A2B-3C4D")


def test_authenticate_salinan_ke_flashdisk_lain_ditolak(registered, tmp_path):
    """dongle.key disalin utuh ke flashdisk lain: nomor seri berbeda."""
    d, record = registered
    lain = tmp_path / "flashdisk-lain"
    lain.mkdir()
    shutil.copy(d.path, lain / dg.DONGLE_FILENAME)
    salinan = dg.load_dongle(str(lain / dg.DONGLE_FILENAME))
    with pytest.raises(DongleError, match="nomor seri"):
        dg.authenticate_dongle(salinan, record, ChallengeIssuer(), actual_serial="9999-0000")
    with pytest.raises(DongleError, match="nomor seri"):
        dg.authenticate_dongle(salinan, record, ChallengeIssuer(), actual_serial=None)


def test_authenticate_kunci_dipalsukan_ditolak(registered, tmp_path):
    """dongle_id benar tetapi kunci privat diganti: challenge-response gagal."""
    d, record = registered
    lain = tmp_path / "palsu"
    lain.mkdir()
    dg.create_dongle(str(lain))
    palsu = dg.load_dongle(str(lain / dg.DONGLE_FILENAME))
    palsu = dataclasses.replace(palsu, dongle_id=record.dongle_id)
    with pytest.raises(DongleError, match="challenge-response"):
        dg.authenticate_dongle(palsu, record, ChallengeIssuer(), actual_serial="1A2B-3C4D")


# ------------------------------------------------------------ khusus Windows
@pytest.mark.skipif(os.name != "nt", reason="nomor seri volume hanya dibaca di Windows")
def test_nomor_seri_volume_format_windows(drive):
    serial = dg.read_volume_serial(drive)
    assert serial is not None
    assert re.fullmatch(r"[0-9A-F]{4}-[0-9A-F]{4}", serial)


@pytest.mark.skipif(os.name != "nt", reason="nomor seri volume hanya dibaca di Windows")
def test_create_dongle_mencatat_nomor_seri_di_windows(drive):
    record = dg.create_dongle(drive)
    assert record.volume_serial == dg.read_volume_serial(drive)
