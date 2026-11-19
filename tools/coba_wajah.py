"""Uji manual faktor wajah dengan webcam, sekaligus untuk menyetel ambang.

Pemakaian (dari folder proyek, venv aktif):
    python tools/coba_wajah.py              -> ambang 0.5
    python tools/coba_wajah.py --ambang 0.45

Tahap 1 DAFTAR : tekan SPASI 5 kali untuk mengambil 5 sampel wajahmu.
Tahap 2 UJI    : tekan SPASI untuk verifikasi. Coba wajahmu sendiri berkali-kali
                 (posisi dan cahaya berbeda), lalu wajah orang lain, lalu foto wajahmu di HP.
ESC atau tutup jendela untuk selesai. Ringkasan jarak dicetak di akhir.

Wajah TIDAK disimpan ke disk: template hanya ada di memori selama skrip berjalan.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2  # noqa: E402

from klr import face  # noqa: E402

WINDOW = "KLR - uji wajah (SPASI ambil, ESC keluar)"
KEY_SPACE, KEY_ESC = 32, 27


def overlay(frame, lines, color=(255, 255, 255)):
    """Tulis beberapa baris teks di atas latar gelap di pojok kiri atas."""
    h = 28 * len(lines) + 12
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 28 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambang", type=float, default=face.DEFAULT_THRESHOLD)
    parser.add_argument("--kamera", type=int, default=0)
    args = parser.parse_args(argv)

    print("Memuat model wajah (beberapa detik)...")
    face._fr()

    cam = cv2.VideoCapture(args.kamera)
    if not cam.isOpened():
        sys.exit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")

    samples, template = [], None
    hasil = []  # (lolos, jarak)
    pesan, warna = "", (255, 255, 255)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                sys.exit("Gagal membaca gambar dari webcam.")

            if template is None:
                baris = [f"DAFTAR: SPASI untuk sampel {len(samples) + 1}/{face.ENROLL_SAMPLES}"]
            else:
                baris = [f"UJI (ambang {args.ambang}): SPASI untuk verifikasi"]
            if pesan:
                baris.append(pesan)
            tampil = frame.copy()
            overlay(tampil, baris, warna)
            cv2.imshow(WINDOW, tampil)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key != KEY_SPACE:
                continue

            try:
                emb = face.extract_embedding(face.bgr_to_rgb(frame))
            except face.FaceError as exc:
                pesan, warna = str(exc)[:70], (0, 165, 255)
                print("  gagal:", exc)
                continue

            if template is None:
                samples.append(emb)
                pesan, warna = f"sampel {len(samples)} diambil", (255, 255, 255)
                print(pesan)
                if len(samples) == face.ENROLL_SAMPLES:
                    try:
                        template = face.build_template(samples, args.ambang)
                        jarak = [face.face_distance(template, s) for s in samples]
                        pesan = f"template jadi (jarak sampel {min(jarak):.3f}-{max(jarak):.3f})"
                        print(pesan)
                    except face.FaceError as exc:
                        samples = []
                        pesan, warna = "sampel tidak konsisten, ulangi", (0, 165, 255)
                        print("  ", exc)
            else:
                lolos, jarak = face.verify(template, emb, args.ambang)
                hasil.append((lolos, jarak))
                pesan = f"{'LOLOS' if lolos else 'DITOLAK'}  jarak {jarak:.3f}"
                warna = (0, 200, 0) if lolos else (0, 0, 255)
                print(f"  uji {len(hasil)}: {pesan}")
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if hasil:
        print("\nRingkasan uji:")
        for i, (lolos, jarak) in enumerate(hasil, 1):
            print(f"  {i:2d}. {'LOLOS  ' if lolos else 'DITOLAK'}  jarak {jarak:.3f}")
        print("Catat mana yang wajahmu dan mana yang bukan, lalu pilih ambang di antara keduanya.")


if __name__ == "__main__":
    main()
