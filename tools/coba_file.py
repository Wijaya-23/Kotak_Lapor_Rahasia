"""Demo skenario uji nomor 4 (laporan 4.3): enkripsi file ChaCha20-Poly1305 per potongan.

Pemakaian (dari folder proyek, venv aktif):
    python tools/coba_file.py --buat 10          -> buat file acak 10 MB lalu uji
    python tools/coba_file.py "D:\\bukti\\x.pdf"   -> uji dengan file milikmu

Yang ditunjukkan:
    1. File dienkripsi menjadi .enc dengan nama acak; isi dan nama aslinya tidak terbaca.
    2. Dekripsi menghasilkan file yang SHA-256-nya sama persis dengan aslinya.
    3. Potongan ditukar, potongan terakhir dibuang, atau satu byte diubah -> ditolak.

Kunci Admin di demo ini dibuat sementara di memori, hanya untuk demo. Semua hasil ada
di data/coba_file/ (folder data/ ada di .gitignore). File aslimu tidak diubah.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from klr import filecrypto as fc  # noqa: E402
from klr import keys  # noqa: E402
from klr.errors import DecryptionError  # noqa: E402

WORK = os.path.join("data", "coba_file")
CONTEXT = "demo-lampiran"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def chunk_ranges(enc_path: str, key_bytes: int) -> tuple[bytes, list[tuple[int, int]]]:
    raw = open(enc_path, "rb").read()
    header = 4 + 4 + key_bytes + 4
    chunk = int.from_bytes(raw[header - 4 : header], "big")
    meta_len = int.from_bytes(raw[header : header + 4], "big")
    pos, ranges = header + 4 + meta_len, []
    while pos < len(raw):
        end = min(pos + chunk + fc.TAG_LEN, len(raw))
        ranges.append((pos, end))
        pos = end
    return raw, ranges


def try_decrypt(label: str, enc_path: str, key, out_dir: str) -> bool:
    try:
        fc.decrypt_file(enc_path, key, out_dir, context=CONTEXT)
    except DecryptionError as exc:
        print(f"  {label:<38} DITOLAK  ({exc})")
        return True
    print(f"  {label:<38} LOLOS  <- seharusnya ditolak!")
    return False


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="file yang akan diuji")
    parser.add_argument("--buat", type=int, metavar="MB", help="buat file acak berukuran MB megabyte")
    args = parser.parse_args(argv)

    if args.file and os.path.abspath(args.file.strip('"')).startswith(os.path.abspath(WORK)):
        sys.exit("Pilih file di luar folder data/coba_file (folder itu dikosongkan setiap demo).")
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    if args.buat:
        src = os.path.join(WORK, f"sampel_{args.buat}MB.bin")
        with open(src, "wb") as fh:
            for _ in range(args.buat):
                fh.write(os.urandom(1024 * 1024))
    elif args.file:
        src = args.file.strip('"')
    else:
        parser.error("beri path file atau --buat MB")

    print("Membuat kunci Admin sementara untuk demo...")
    admin = keys.generate_encryption_keypair()

    print("\n[1] Enkripsi")
    t = time.perf_counter()
    try:
        ef = fc.encrypt_file(src, os.path.join(WORK, "enc"), admin.public_key(), context=CONTEXT)
    except (ValueError, OSError) as exc:
        sys.exit(f"Gagal: {exc}")
    dt = time.perf_counter() - t
    _, ranges = chunk_ranges(ef.enc_path, admin.key_size // 8)
    print(f"  file asli   : {os.path.basename(src)}  {ef.meta.size:,} byte")
    print(f"  SHA-256 asli: {ef.meta.sha256}")
    print(f"  berkas .enc : {os.path.basename(ef.enc_path)}  {ef.enc_size:,} byte, {len(ranges)} potongan")
    print(f"  SHA-256 .enc: {ef.enc_sha256}   (yang nanti dicatat di blockchain)")
    print(f"  waktu       : {dt:.2f} detik")

    raw = open(ef.enc_path, "rb").read()
    print("\n[2] Isi .enc jika dibuka seperti teks (seperti di Notepad):")
    print("  " + raw[300:380].decode("latin-1").encode("ascii", "replace").decode())
    nama = os.path.basename(src).encode()
    print(f"  nama asli '{os.path.basename(src)}' muncul di dalam .enc? {'YA' if nama in raw else 'TIDAK'}")

    print("\n[3] Dekripsi")
    out, meta = fc.decrypt_file(ef.enc_path, admin, os.path.join(WORK, "hasil"), context=CONTEXT)
    same = sha256(out) == ef.meta.sha256
    print(f"  hasil       : {out}")
    print(f"  SHA-256     : {sha256(out)}")
    print(f"  sama dengan aslinya? {'SAMA' if same else 'BEDA'}")

    print("\n[4] Manipulasi potongan (salinan .enc diubah, lalu didekripsi)")
    rusak_dir, out_dir = os.path.join(WORK, "rusak"), os.path.join(WORK, "hasil_rusak")
    os.makedirs(rusak_dir)
    results = []

    def variant(name: str, data: bytes) -> str:
        p = os.path.join(rusak_dir, name + ".enc")
        with open(p, "wb") as fh:
            fh.write(data)
        return p

    if len(ranges) >= 2:
        (a0, a1), (b0, b1) = ranges[0], ranges[1]
        results.append(try_decrypt("potongan 1 dan 2 ditukar", variant("tukar", raw[:a0] + raw[b0:b1] + raw[a0:a1] + raw[b1:]), admin, out_dir))
        results.append(try_decrypt("potongan terakhir dibuang", variant("potong", raw[: ranges[-1][0]]), admin, out_dir))
    else:
        print("  (file hanya 1 potongan; pakai file > 1 MB untuk uji tukar/buang potongan)")
    tengah = bytearray(raw)
    tengah[len(raw) // 2] ^= 0x01
    results.append(try_decrypt("satu byte di tengah diubah", variant("ubah", bytes(tengah)), admin, out_dir))
    results.append(try_decrypt("dibuka dengan kunci Admin lain", ef.enc_path, keys.generate_encryption_keypair(), out_dir))
    sisa = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    print(f"  berkas hasil yang tertinggal dari percobaan gagal: {len(sisa)}")

    ok = same and all(results) and not sisa
    print(f"\nKESIMPULAN: {'SEMUA SESUAI HARAPAN' if ok else 'ADA YANG TIDAK SESUAI, kirim keluaran ini ke Claude'}")


if __name__ == "__main__":
    main()
