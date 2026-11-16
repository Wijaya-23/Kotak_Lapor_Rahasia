"""Uji manual faktor dongle dengan flashdisk asli (Windows).

Pemakaian (dari folder proyek, venv aktif):
    python tools/coba_dongle.py               -> daftar / login dengan flashdisk
    python tools/coba_dongle.py --root D:\\x   -> pakai folder lain (simulasi dongle disalin)

Data publik dongle disimpan sementara di data/dongle_record.json sebagai pengganti
basis data. Folder data/ sudah ada di .gitignore.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from klr import dongle as dg  # noqa: E402

RECORD_PATH = os.path.join("data", "dongle_record.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="cari dongle.key di folder ini, bukan di flashdisk")
    args = parser.parse_args()

    roots = [args.root] if args.root else dg.list_removable_drives()
    print("Lokasi yang diperiksa:", roots or "(tidak ada drive removable)")
    for r in roots:
        print(f"  nomor seri volume {r}: {dg.read_volume_serial(r)}")

    if not os.path.exists(RECORD_PATH):
        if not roots:
            sys.exit("Tancapkan flashdisk dulu.")
        if len(roots) > 1:
            sys.exit("Lebih dari satu drive. Pakai --root untuk memilih.")
        if input(f"Belum ada dongle terdaftar. Buat dongle.key di {roots[0]}? (y/n) ").strip().lower() != "y":
            return
        record = dg.create_dongle(roots[0])
        os.makedirs("data", exist_ok=True)
        with open(RECORD_PATH, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(record), fh, indent=2)
        print("Dongle dibuat. dongle_id:", record.dongle_id, "| nomor seri:", record.volume_serial)
        print("Jalankan skrip ini sekali lagi untuk mencoba login.")
        return

    with open(RECORD_PATH, encoding="utf-8") as fh:
        record = dg.DongleRecord(**json.load(fh))

    try:
        path = dg.require_single_dongle(roots)
        dongle = dg.load_dongle(path)
        dg.authenticate_dongle(dongle, record, dg.ChallengeIssuer(), dg.read_volume_serial(path))
    except dg.DongleError as exc:
        print("DITOLAK:", exc)
        sys.exit(1)
    print("LOLOS: faktor dongle valid untuk", dongle)


if __name__ == "__main__":
    main()
