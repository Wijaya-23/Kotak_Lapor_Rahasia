"""Faktor wajah (laporan subbab 3.1, langkah 3).

Alur:
    webcam -> gambar RGB -> face_recognition (dlib) -> embedding 128 angka
    pendaftaran : rata-rata 5 embedding = template, dienkripsi (keys.wrap_face_template)
    login       : jarak Euclidean(template, embedding baru) < ambang -> lolos

Wajah adalah GERBANG, bukan sumber kunci: template bukan rahasia kriptografis dan
hasil pembacaan selalu sedikit berbeda. Keamanan kunci tetap bertumpu pada password
dan dongle. Tidak ada liveness detection, jadi foto wajah bisa mengelabui faktor ini
(dicatat di tabel ancaman 4.1).

`face_recognition` dan `cv2` baru diimpor saat dipakai, supaya modul ini dan tesnya
bisa jalan tanpa webcam.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .errors import KlrError

EMBEDDING_DIM = 128
TEMPLATE_BYTES = EMBEDDING_DIM * 8  # float64
DEFAULT_THRESHOLD = 0.5  # laporan: diawali 0,5 lalu disetel dari hasil uji
ENROLL_SAMPLES = 5


class FaceError(KlrError):
    """Faktor wajah gagal: tidak ada wajah, lebih dari satu wajah, atau data rusak."""


def _fr():
    import face_recognition  # impor lambat: memuat model dlib

    return face_recognition


# --------------------------------------------------------------------------
# Kamera
# --------------------------------------------------------------------------
def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """OpenCV memberi BGR, face_recognition butuh RGB yang bersambung di memori.

    Jangan pakai frame[:, :, ::-1]: hasilnya tidak bersambung dan dlib menolaknya.
    """
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def extract_embedding(rgb_image: np.ndarray) -> np.ndarray:
    """Embedding 128 dimensi dari gambar yang berisi TEPAT satu wajah."""
    fr = _fr()
    locations = fr.face_locations(rgb_image, model="hog")
    if len(locations) == 0:
        raise FaceError("tidak ada wajah terdeteksi. Hadapkan wajah ke kamera dengan cahaya cukup.")
    if len(locations) > 1:
        raise FaceError("terdeteksi lebih dari satu wajah. Pastikan hanya satu orang di depan kamera.")
    encodings = fr.face_encodings(rgb_image, known_face_locations=locations)
    if len(encodings) != 1:
        raise FaceError("embedding wajah gagal dihitung")
    return _validate(np.asarray(encodings[0], dtype=np.float64))


def _validate(vec: np.ndarray) -> np.ndarray:
    if vec.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(vec)):
        raise FaceError("embedding tidak valid")
    return vec


# --------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------
def face_distance(template: np.ndarray, embedding: np.ndarray) -> float:
    """Jarak Euclidean antara dua embedding. Makin kecil makin mirip."""
    return float(np.linalg.norm(_validate(template) - _validate(embedding)))


def build_template(samples: Sequence[np.ndarray], threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Template = rata-rata beberapa embedding dari orang yang sama.

    Menolak sampel yang tidak konsisten (misalnya orang lain ikut terekam atau gambar
    buram), karena template yang buruk membuat login pemiliknya sendiri sering gagal.
    """
    if len(samples) < 3:
        raise FaceError("butuh minimal 3 sampel wajah")
    stack = np.stack([_validate(np.asarray(s, dtype=np.float64)) for s in samples])
    template = stack.mean(axis=0)
    worst = max(face_distance(template, s) for s in stack)
    if worst >= threshold:
        raise FaceError(f"sampel wajah tidak konsisten (jarak terjauh {worst:.3f}). Ulangi pendaftaran.")
    return template


def template_to_bytes(template: np.ndarray) -> bytes:
    """1024 byte (128 x float64 little-endian), siap dibungkus keys.wrap_face_template."""
    return _validate(np.asarray(template, dtype=np.float64)).astype("<f8").tobytes()


def template_from_bytes(data: bytes) -> np.ndarray:
    if len(data) != TEMPLATE_BYTES:
        raise FaceError("template rusak")
    return _validate(np.frombuffer(data, dtype="<f8").astype(np.float64))


# --------------------------------------------------------------------------
# Verifikasi
# --------------------------------------------------------------------------
def verify(template: np.ndarray, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float]:
    """(lolos, jarak). Lolos jika jarak DI BAWAH ambang."""
    if not (0 < threshold < 2):  # juga menolak NaN
        raise ValueError("ambang harus di antara 0 dan 2")
    distance = face_distance(template, embedding)
    return distance < threshold, distance
