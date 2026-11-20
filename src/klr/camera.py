"""Pengambilan wajah dari webcam lewat jendela OpenCV.

Dipakai skrip uji sekarang, dan nanti oleh GUI: saat login atau pendaftaran,
GUI cukup memanggil fungsi di sini dan sebuah jendela kamera akan muncul.
SPASI = ambil, ESC = batal.

Tombol hanya terbaca jika jendela kamera sedang aktif (fokus). Karena itu jendela
dibawa ke depan saat dibuka, dan petunjuk "klik jendela ini" ditampilkan.
"""
from __future__ import annotations

import os

import numpy as np

from . import face

KEY_SPACE, KEY_ESC = 32, 27
WHITE, ORANGE, YELLOW = (255, 255, 255), (0, 165, 255), (0, 255, 255)


def _overlay(cv2, frame, lines, color):
    h = 28 * len(lines) + 12
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 28 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _bring_to_front(cv2, window: str) -> None:
    """Upaya terbaik membawa jendela ke depan dan memberinya fokus keyboard."""
    prop = getattr(cv2, "WND_PROP_TOPMOST", None)
    if prop is not None:
        try:
            cv2.setWindowProperty(window, prop, 1)
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, window)
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass


def capture_embeddings(count: int, title: str, camera_index: int = 0) -> list[np.ndarray]:
    """Ambil `count` embedding wajah. Melempar FaceError jika dibatalkan atau kamera gagal."""
    import cv2

    face._fr()  # muat model dulu supaya SPASI pertama tidak membeku
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise face.FaceError("webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")
    window = f"KLR - {title} (SPASI ambil, ESC batal)"
    print(f"Jendela kamera '{title}' terbuka. Jika SPASI tidak bereaksi, KLIK jendela kamera dulu.")
    result: list[np.ndarray] = []
    message, color = "Klik jendela ini dulu jika SPASI tidak bereaksi", YELLOW
    first = True
    try:
        while len(result) < count:
            ok, frame = cam.read()
            if not ok:
                raise face.FaceError("gagal membaca gambar dari webcam")
            shown = frame.copy()
            _overlay(cv2, shown, [f"{title}: SPASI untuk ambil {len(result) + 1}/{count}", message], color)
            cv2.imshow(window, shown)
            if first:
                _bring_to_front(cv2, window)
                first = False
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise face.FaceError("pengambilan wajah dibatalkan")
            if key != KEY_SPACE:
                continue

            # Umpan balik langsung: deteksi wajah butuh sekitar setengah detik.
            busy = frame.copy()
            _overlay(cv2, busy, [f"{title}: memproses...", ""], WHITE)
            cv2.imshow(window, busy)
            cv2.waitKey(1)
            try:
                result.append(face.extract_embedding(face.bgr_to_rgb(frame)))
                message, color = f"diambil {len(result)}/{count}", WHITE
                print(f"  wajah diambil ({len(result)}/{count})")
            except face.FaceError as exc:
                message, color = str(exc)[:70], ORANGE
                print(f"  gagal: {exc}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return result


def capture_one(title: str = "Login wajah", camera_index: int = 0) -> np.ndarray:
    return capture_embeddings(1, title, camera_index)[0]
