"""KLR versi terminal: inisialisasi Admin, login tiga faktor, daftar Pelapor,
kirim dan baca laporan teks, kirim file dan simpan lampiran. Memakai flashdisk dan webcam asli.

Pemakaian (dari folder proyek, venv aktif):
    python tools/coba_login.py
    python tools/coba_login.py --root D:\\dongle_pelapor   (dongle di folder, untuk uji tanpa flashdisk kedua)

Basis data: data/klr.db, file terenkripsi: data/files/, lampiran yang dibuka Admin: data/unduhan/
(folder data/ ada di .gitignore). Hapus berkas itu untuk mulai dari nol,
dan hapus juga dongle.key di flashdisk karena sudah tidak terdaftar.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from getpass import getpass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from klr import auth, camera, face, filecrypto, keys, textcrypto  # noqa: E402
from klr import dongle as dg  # noqa: E402
from klr.db import Database, new_id  # noqa: E402
from klr.errors import KlrError  # noqa: E402

FILES_DIR = os.path.join("data", "files")
DOWNLOAD_DIR = os.path.join("data", "unduhan")


def local_time(iso: str) -> str:
    """Waktu disimpan dalam UTC; tampilkan dalam zona waktu komputer (WIB)."""
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def ask_password(confirm: bool) -> str:
    pw = getpass("Password: ")
    if confirm and getpass("Ulangi password: ") != pw:
        raise ValueError("password tidak sama")
    return pw


def pick_dongle_root(fixed: str | None, purpose: str) -> str:
    if fixed:
        return fixed
    drives = dg.list_removable_drives()
    if len(drives) != 1:
        raise ValueError(f"tancapkan tepat satu flashdisk untuk {purpose} (terdeteksi: {drives or 'tidak ada'})")
    return drives[0]


def setup_admin(db: Database, args) -> None:
    print("\n=== Sistem belum diinisialisasi: membuat Admin pertama ===")
    username = input("Username Admin: ").strip()
    password = ask_password(confirm=True)
    root = pick_dongle_root(args.root, "dongle Admin")
    if input(f"Tulis dongle.key Admin ke {root}? (y/n) ").strip().lower() != "y":
        sys.exit("Dibatalkan.")
    print("Jendela kamera akan terbuka. Ambil 5 sampel wajah Admin.")
    samples = camera.capture_embeddings(face.ENROLL_SAMPLES, "Daftar wajah Admin")
    print("Membuat kunci (beberapa detik)...")
    user = auth.initialize_admin(db, username=username, password=password, dongle_root=root, face_samples=samples)
    print(f"Admin '{user.username}' dibuat. DEK dan pasangan kunci enkripsi Admin sudah dibuat.")


def do_login(db: Database, args) -> auth.Session | None:
    password = getpass("Password: ")
    try:
        session = auth.login(
            db,
            password=password,
            capture_face=lambda: camera.capture_one("Login wajah"),
            dongle_roots=[args.root] if args.root else None,
            threshold=args.ambang,
        )
    except auth.AuthError as exc:
        print(f"DITOLAK pada langkah [{exc.step}]: {exc}")
        return None
    print(f"LOLOS. Selamat datang, {session.user.username} ({session.user.role}).")
    print("  DEK terbuka di memori." + ("  Kunci privat enkripsi Admin terbuka." if session.is_admin else ""))
    return session


def register(db: Database, session: auth.Session) -> None:
    username = input("Username Pelapor baru: ").strip()
    password = ask_password(confirm=True)
    loc = input("Lokasi dongle Pelapor (Enter = flashdisk; atau path folder): ").strip()
    if not loc:
        input("Cabut flashdisk Admin, tancapkan flashdisk KOSONG untuk Pelapor, lalu tekan Enter...")
    root = pick_dongle_root(loc or None, "dongle Pelapor")
    print("Jendela kamera akan terbuka. Ambil 5 sampel wajah Pelapor.")
    samples = camera.capture_embeddings(face.ENROLL_SAMPLES, f"Daftar wajah {username}")
    user = auth.register_pelapor(db, session, username=username, password=password, dongle_root=root,
                                 face_samples=samples)
    print(f"Pelapor '{user.username}' terdaftar, dongle di {root}. Slot DEK baru dibungkus KEK miliknya.")


def send_report(db: Database, session: auth.Session) -> None:
    print("Tulis laporan. Akhiri dengan baris kosong.")
    lines = []
    while (line := input()) != "":
        lines.append(line)
    text = "\n".join(lines)
    if not text:
        print("Laporan kosong, dibatalkan.")
        return
    admin = db.get_admin()
    report_id = new_id()
    pkg = textcrypto.encrypt_text(text, keys.load_encryption_public_key(admin.enc_pubkey), report_id,
                                  session.user.id)
    db.add_report(session.dek, report_id=report_id, reporter_id=session.user.id, kind="teks", body_pkg=pkg)
    print(f"Terkirim. id laporan: {report_id}")
    print(f"  paket hybrid (awal): {pkg[:60]}...")


def send_file_report(db: Database, session: auth.Session) -> None:
    path = input("Path file (boleh hasil 'Copy as path'): ").strip().strip('"')
    note = input("Keterangan singkat (boleh kosong): ").strip()
    admin_pub = keys.load_encryption_public_key(db.get_admin().enc_pubkey)
    report_id, attachment_id = new_id(), new_id()
    print("Mengenkripsi file...")
    ef = filecrypto.encrypt_file(path, FILES_DIR, admin_pub, context=attachment_id)
    try:
        pkg = textcrypto.encrypt_text(note, admin_pub, report_id, session.user.id) if note else None
        db.add_report(session.dek, report_id=report_id, reporter_id=session.user.id, kind="file", body_pkg=pkg)
        db.add_attachment(session.dek, attachment_id=attachment_id, report_id=report_id, enc_path=ef.enc_path,
                          size=ef.enc_size, filename=ef.meta.name)
    except BaseException:
        os.remove(ef.enc_path)
        raise
    print(f"Terkirim. id laporan: {report_id}")
    print(f"  disimpan sebagai : {ef.enc_path}  ({ef.enc_size:,} byte)")
    print(f"  SHA-256 asli     : {ef.meta.sha256}")
    print(f"  SHA-256 .enc     : {ef.enc_sha256}")


def my_reports(db: Database, session: auth.Session) -> None:
    reports = db.list_reports(session.dek, session.user.id)
    print(f"{len(reports)} laporan. Isi hanya bisa dibaca Admin; kamu hanya melihat status.")
    for r in reports:
        print(f"  {local_time(r.created_at)}  {r.id[:8]}  {r.kind:<6}  status: {r.status}")


def read_reports(db: Database, session: auth.Session) -> None:
    reports = db.list_reports(session.dek)
    print(f"{len(reports)} laporan.")
    for r in reports:
        reporter = db.get_user(r.reporter_id)
        isi = "(tanpa isi teks)"
        if r.body_pkg:
            isi = textcrypto.decrypt_text(r.body_pkg, session.admin_private_key, r.id, r.reporter_id)
        print(f"\n  [{local_time(r.created_at)}] dari {reporter.username}  status: {r.status}  id: {r.id[:8]}")
        for line in isi.splitlines():
            print(f"    {line}")
        for a in db.list_attachments(session.dek, r.id):
            print(f"    lampiran: {a.filename}  ({a.size:,} byte terenkripsi)")


def save_attachment(db: Database, session: auth.Session) -> None:
    items = [a for r in db.list_reports(session.dek) for a in db.list_attachments(session.dek, r.id)]
    if not items:
        print("Belum ada lampiran.")
        return
    for i, a in enumerate(items, 1):
        print(f"  {i}) {a.filename}  ({a.size:,} byte terenkripsi)")
    pick = input("Nomor lampiran: ").strip()
    if not pick.isdigit() or not 1 <= int(pick) <= len(items):
        print("Nomor tidak valid.")
        return
    a = items[int(pick) - 1]
    out, meta = filecrypto.decrypt_file(a.enc_path, session.admin_private_key, DOWNLOAD_DIR, context=a.id)
    print(f"Tersimpan: {out}")
    print(f"  ukuran {meta.size:,} byte, SHA-256 {meta.sha256} (cocok dengan metadata)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.path.join("data", "klr.db"))
    parser.add_argument("--root", help="cari dongle di folder ini, bukan di flashdisk")
    parser.add_argument("--ambang", type=float, default=face.DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    with Database(args.db) as db:
        if not db.is_initialized():
            try:
                setup_admin(db, args)
            except (KlrError, ValueError) as exc:
                sys.exit(f"Gagal: {exc}")

        session: auth.Session | None = None
        while True:
            print()
            if session is None:
                menu = {"1": "Login", "0": "Keluar"}
            elif session.is_admin:
                menu = {"2": "Daftarkan Pelapor", "4": "Baca semua laporan", "7": "Simpan lampiran", "9": "Logout",
                        "0": "Keluar"}
            else:
                menu = {"3": "Kirim laporan teks", "5": "Status laporan saya", "6": "Kirim laporan file", "9": "Logout",
                        "0": "Keluar"}
            who = f"[{session.user.username}]" if session else "[belum login]"
            print(who, "  ".join(f"{k}) {v}" for k, v in menu.items()))
            choice = input("> ").strip()
            if choice not in menu:
                continue
            try:
                if choice == "0":
                    break
                if choice == "1":
                    session = do_login(db, args)
                elif choice == "2":
                    register(db, session)
                elif choice == "3":
                    send_report(db, session)
                elif choice == "4":
                    read_reports(db, session)
                elif choice == "5":
                    my_reports(db, session)
                elif choice == "6":
                    send_file_report(db, session)
                elif choice == "7":
                    save_attachment(db, session)
                elif choice == "9":
                    session.logout()
                    session = None
                    print("Logout. Kunci di memori sudah dihapus.")
            except (KlrError, ValueError, OSError) as exc:
                print(f"Gagal: {exc}")
        if session:
            session.logout()


if __name__ == "__main__":
    main()
