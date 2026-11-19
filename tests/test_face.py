"""Tes faktor wajah tanpa webcam: face_recognition diganti tiruan yang bisa diatur."""
import os

import numpy as np
import pytest

from klr import face, kdf, keys
from klr.errors import DecryptionError
from klr.face import FaceError
from klr.kdf import Argon2Params

rng = np.random.default_rng(2026)


def _unit(v):
    return v / np.linalg.norm(v)


# Orang A dan orang B: dua arah acak yang berjauhan (jarak sekitar 1,4).
PERSON_A = _unit(rng.normal(size=128))
PERSON_B = _unit(rng.normal(size=128))


def _sample(base, noise=0.02):
    """Satu pembacaan wajah: embedding dasar ditambah sedikit derau."""
    return base + rng.normal(scale=noise, size=128)


class FakeFR:
    """Pengganti modul face_recognition. `faces` = daftar embedding wajah di gambar."""

    def __init__(self, faces):
        self.faces = faces
        self.calls = []

    def face_locations(self, img, model="hog"):
        self.calls.append(("locations", model))
        return [(10 * i, 50, 50, 10) for i in range(len(self.faces))]

    def face_encodings(self, img, known_face_locations=None):
        self.calls.append(("encodings", len(known_face_locations)))
        return [self.faces[i] for i in range(len(known_face_locations))]


@pytest.fixture
def fake_fr(monkeypatch):
    def install(faces):
        fr = FakeFR(faces)
        monkeypatch.setattr(face, "_fr", lambda: fr)
        return fr

    return install


IMG = np.zeros((480, 640, 3), dtype=np.uint8)


# ------------------------------------------------------------ embedding
def test_satu_wajah_menghasilkan_embedding_128(fake_fr):
    fr = fake_fr([PERSON_A])
    emb = face.extract_embedding(IMG)
    assert emb.shape == (128,)
    assert emb.dtype == np.float64
    assert ("locations", "hog") in fr.calls


def test_tidak_ada_wajah_ditolak(fake_fr):
    fake_fr([])
    with pytest.raises(FaceError, match="tidak ada wajah"):
        face.extract_embedding(IMG)


def test_dua_wajah_ditolak(fake_fr):
    """Mencegah orang lain 'menumpang' di depan kamera saat login atau pendaftaran."""
    fake_fr([PERSON_A, PERSON_B])
    with pytest.raises(FaceError, match="lebih dari satu"):
        face.extract_embedding(IMG)


@pytest.mark.parametrize("bad", [np.zeros(127), np.full(128, np.nan), np.full(128, np.inf)])
def test_embedding_tidak_valid_ditolak(fake_fr, bad):
    fake_fr([bad])
    with pytest.raises(FaceError):
        face.extract_embedding(IMG)


# ------------------------------------------------------------ template
def test_template_rata_rata_sampel():
    samples = [_sample(PERSON_A) for _ in range(5)]
    template = face.build_template(samples)
    assert np.allclose(template, np.mean(samples, axis=0))


def test_template_menolak_sampel_campuran_dua_orang():
    samples = [_sample(PERSON_A) for _ in range(4)] + [_sample(PERSON_B)]
    with pytest.raises(FaceError, match="tidak konsisten"):
        face.build_template(samples)


def test_template_butuh_minimal_tiga_sampel():
    with pytest.raises(FaceError):
        face.build_template([_sample(PERSON_A), _sample(PERSON_A)])


def test_template_ke_byte_dan_kembali():
    template = face.build_template([_sample(PERSON_A) for _ in range(5)])
    data = face.template_to_bytes(template)
    assert len(data) == 1024
    assert np.array_equal(face.template_from_bytes(data), template)


@pytest.mark.parametrize("data", [b"", bytes(1023), bytes(1025), np.full(128, np.nan).astype("<f8").tobytes()])
def test_template_rusak_ditolak(data):
    with pytest.raises(FaceError):
        face.template_from_bytes(data)


# ------------------------------------------------------------ verifikasi
def test_orang_yang_sama_lolos():
    template = face.build_template([_sample(PERSON_A) for _ in range(5)])
    ok, dist = face.verify(template, _sample(PERSON_A))
    assert ok
    assert dist < 0.5


def test_orang_lain_ditolak():
    """Skenario uji nomor 1 di laporan: login dengan wajah orang lain."""
    template = face.build_template([_sample(PERSON_A) for _ in range(5)])
    ok, dist = face.verify(template, _sample(PERSON_B))
    assert not ok
    assert dist > 1.0


def test_ambang_tepat_pada_batas_ditolak():
    """Laporan: lolos jika jarak DI BAWAH ambang. Sama dengan ambang = ditolak."""
    template = np.zeros(128)
    embedding = np.zeros(128)
    embedding[0] = 0.5
    assert face.verify(template, embedding, threshold=0.5) == (False, 0.5)
    assert face.verify(template, embedding, threshold=0.5001)[0] is True


@pytest.mark.parametrize("threshold", [0, -0.1, 2, float("nan")])
def test_ambang_tidak_masuk_akal_ditolak(threshold):
    with pytest.raises(ValueError):
        face.verify(np.zeros(128), np.zeros(128), threshold=threshold)


def test_ambang_lebih_ketat_menolak_yang_tadinya_lolos():
    template = np.zeros(128)
    embedding = np.zeros(128)
    embedding[0] = 0.45
    assert face.verify(template, embedding, threshold=0.5)[0] is True
    assert face.verify(template, embedding, threshold=0.4)[0] is False


# ------------------------------------------------------------ alur dengan modul lain
def test_alur_daftar_dan_login_dengan_template_terenkripsi(fake_fr):
    """Daftar: 5 sampel -> template -> dibungkus sub-kunci KEK. Login: buka -> bandingkan."""
    fast = Argon2Params(memory_kib=1024, time_cost=1, parallelism=1)
    salt, secret = kdf.new_salt(), os.urandom(32)
    kek = kdf.derive_kek("Pelapor!2026", salt, secret, fast)

    samples = []
    for _ in range(5):
        fake_fr([_sample(PERSON_A)])
        samples.append(face.extract_embedding(IMG))
    blob = keys.wrap_face_template(face.template_to_bytes(face.build_template(samples)), kek, "user-7")

    # Login oleh orang yang sama
    fake_fr([_sample(PERSON_A)])
    template = face.template_from_bytes(keys.unwrap_face_template(blob, kek, "user-7"))
    assert face.verify(template, face.extract_embedding(IMG))[0] is True

    # Orang lain di depan kamera
    fake_fr([_sample(PERSON_B)])
    assert face.verify(template, face.extract_embedding(IMG))[0] is False

    # Tanpa KEK yang benar, template bahkan tidak bisa dibuka
    kek_salah = kdf.derive_kek("salah", salt, secret, fast)
    with pytest.raises(DecryptionError):
        keys.unwrap_face_template(blob, kek_salah, "user-7")


def test_bgr_ke_rgb_bersambung():
    pytest.importorskip("cv2")
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # biru di OpenCV
    rgb = face.bgr_to_rgb(bgr)
    assert rgb[0, 0].tolist() == [0, 0, 255]
    assert rgb.flags["C_CONTIGUOUS"]
