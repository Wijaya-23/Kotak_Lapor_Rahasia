"""Pengambilan wajah dari webcam lewat jendela OpenCV.

Dipakai skrip uji sekarang, dan nanti oleh GUI: saat login atau pendaftaran,
GUI cukup memanggil fungsi di sini dan sebuah jendela kamera akan muncul.
SPASI = ambil, ESC = batal.
"""
from __future__ import annotations

import numpy as np

from . import face

KEY_SPACE, KEY_ESC = 32, 27


def _overlay(cv2, frame, lines, color):
    h = 28 * len(lines) + 12
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 28 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def capture_embeddings(count: int, title: str, camera_index: int = 0) -> list[np.ndarray]:
    """Ambil `count` embedding wajah. Melempar FaceError jika dibatalkan atau kamera gagal."""
    import cv2

    face._fr()  # muat model dulu supaya SPASI pertama tidak membeku
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise face.FaceError("webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")
    window = f"KLR - {title} (SPASI ambil, ESC batal)"
    result: list[np.ndarray] = []
    message, color = "", (255, 255, 255)
    try:
        while len(result) < count:
            ok, frame = cam.read()
            if not ok:
                raise face.FaceError("gagal membaca gambar dari webcam")
            shown = frame.copy()
            lines = [f"{title}: SPASI untuk ambil {len(result) + 1}/{count}"]
            if message:
                lines.append(message)
            _overlay(cv2, shown, lines, color)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise face.FaceError("pengambilan wajah dibatalkan")
            if key != KEY_SPACE:
                continue
            try:
                result.append(face.extract_embedding(face.bgr_to_rgb(frame)))
                message, color = f"diambil {len(result)}/{count}", (255, 255, 255)
            except face.FaceError as exc:
                message, color = str(exc)[:70], (0, 165, 255)
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return result


def capture_one(title: str = "Login wajah", camera_index: int = 0) -> np.ndarray:
    return capture_embeddings(1, title, camera_index)[0]
