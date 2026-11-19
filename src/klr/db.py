"""Basis data SQLite KLR (laporan subbab 3.5).

Kolom sensitif dienkripsi AES-256-GCM dengan DEK lewat `aead.seal`. AAD berisi
nama tabel, nama kolom, dan id baris, sehingga nilai terenkripsi yang dipindah
ke baris atau kolom lain akan ditolak saat dibaca.

Kolom yang sudah dibungkus sub-kunci KEK (users.enc_privkey_wrapped dan
users.face_template_enc, lihat keys.py) disimpan apa adanya, tanpa DEK lagi.

Kolom terbuka (id, waktu, jenis, nama pengguna, kunci publik, rantai blok)
sengaja dibiarkan agar bisa di-query dan diverifikasi. Konsekuensinya, metadata
terlihat oleh pemegang berkas .db (keterbatasan di tabel ancaman 4.1).

Id pengguna, laporan, dan lampiran dibuat aplikasi (UUID) SEBELUM baris disimpan,
karena id itu dibutuhkan untuk AAD saat enkripsi.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .aead import build_aad, open_sealed, seal

SCHEMA_VERSION = 1

ROLE_ADMIN = "admin"
ROLE_PELAPOR = "pelapor"

KINDS = ("teks", "file", "gambar")
STATUSES = ("baru", "diproses", "selesai", "ditolak")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,
    username            TEXT NOT NULL UNIQUE,
    role                TEXT NOT NULL CHECK (role IN ('admin', 'pelapor')),
    pw_hash             TEXT NOT NULL,              -- Argon2id, bukan password
    kek_salt            BLOB NOT NULL,              -- 16 byte, tidak rahasia
    dongle_id           TEXT NOT NULL UNIQUE,
    dongle_serial       TEXT,
    sign_pubkey         TEXT NOT NULL,              -- PEM, kunci penandatangan dongle
    enc_pubkey          TEXT,                       -- PEM, hanya Admin
    enc_privkey_wrapped TEXT,                       -- dibungkus sub-kunci KEK, hanya Admin
    face_template_enc   TEXT NOT NULL,              -- dibungkus sub-kunci KEK
    failed_logins       INTEGER NOT NULL DEFAULT 0,
    locked_until        REAL                        -- waktu Unix, NULL = tidak terkunci
);

-- Hanya boleh ada satu Admin: semua laporan dienkripsi ke kunci publiknya.
CREATE UNIQUE INDEX IF NOT EXISTS one_admin ON users(role) WHERE role = 'admin';

CREATE TABLE IF NOT EXISTS key_slots (
    user_id     TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    wrapped_dek TEXT NOT NULL                       -- DEK dibungkus sub-kunci KEK pengguna
);

CREATE TABLE IF NOT EXISTS reports (
    id           TEXT PRIMARY KEY,
    reporter_id  TEXT NOT NULL REFERENCES users(id),
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('teks', 'file', 'gambar')),
    body_pkg_enc TEXT,                              -- DEK( paket hybrid subbab 3.3 )
    status_enc   TEXT NOT NULL                      -- DEK( status )
);

CREATE TABLE IF NOT EXISTS attachments (
    id           TEXT PRIMARY KEY,
    report_id    TEXT NOT NULL REFERENCES reports(id),
    enc_path     TEXT NOT NULL,                     -- berkas .enc dengan nama acak
    size         INTEGER NOT NULL,
    filename_enc TEXT NOT NULL                      -- DEK( nama asli )
);

CREATE TABLE IF NOT EXISTS stego_images (
    id        TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES reports(id),
    png_path  TEXT NOT NULL                         -- payload di dalam PNG sudah terenkripsi
);

-- Sengaja tanpa enkripsi agar rantai bisa diverifikasi tanpa DEK.
CREATE TABLE IF NOT EXISTS chain (
    idx        INTEGER PRIMARY KEY,
    timestamp  TEXT NOT NULL,
    actor_id   TEXT NOT NULL,
    action     TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    asset_hash TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    nonce      INTEGER NOT NULL,
    hash       TEXT NOT NULL UNIQUE,
    signature  TEXT NOT NULL
);
"""


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Baris data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    role: str
    pw_hash: str
    kek_salt: bytes
    dongle_id: str
    dongle_serial: str | None
    sign_pubkey: str
    face_template_enc: str
    enc_pubkey: str | None = None
    enc_privkey_wrapped: str | None = None
    failed_logins: int = 0
    locked_until: float | None = None


@dataclass(frozen=True)
class Report:
    id: str
    reporter_id: str
    created_at: str
    kind: str
    body_pkg: str | None  # masih paket hybrid: isinya hanya terbuka dengan kunci privat Admin
    status: str


@dataclass(frozen=True)
class Attachment:
    id: str
    report_id: str
    enc_path: str
    size: int
    filename: str


@dataclass(frozen=True)
class StegoImage:
    id: str
    report_id: str
    png_path: str


@dataclass(frozen=True)
class Block:
    idx: int
    timestamp: str
    actor_id: str
    action: str
    asset_type: str
    asset_hash: str
    prev_hash: str
    nonce: int
    hash: str
    signature: str


# --------------------------------------------------------------------------
# Basis data
# --------------------------------------------------------------------------
class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self.conn:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------- enkripsi kolom
    @staticmethod
    def _aad(table: str, column: str, row_id: str) -> bytes:
        return build_aad("klr-db", table, column, row_id)

    def _enc(self, dek: bytes, table: str, column: str, row_id: str, value: str) -> str:
        return seal(dek, value.encode("utf-8"), self._aad(table, column, row_id))

    def _dec(self, dek: bytes, table: str, column: str, row_id: str, blob: str) -> str:
        return open_sealed(dek, blob, self._aad(table, column, row_id)).decode("utf-8")

    # ---------------------------------------------------- meta
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def is_initialized(self) -> bool:
        """True jika Admin pertama sudah dibuat."""
        return self.get_admin() is not None

    # ---------------------------------------------------- users
    def add_user(self, user: UserRecord, wrapped_dek: str | None = None) -> None:
        """Simpan pengguna, dan slot DEK-nya bila diberikan, dalam SATU transaksi:
        tidak mungkin ada pengguna tanpa slot karena proses terhenti di tengah."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO users (id, username, role, pw_hash, kek_salt, dongle_id, dongle_serial,
                       sign_pubkey, enc_pubkey, enc_privkey_wrapped, face_template_enc, failed_logins, locked_until)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user.id, user.username, user.role, user.pw_hash, user.kek_salt, user.dongle_id,
                    user.dongle_serial, user.sign_pubkey, user.enc_pubkey, user.enc_privkey_wrapped,
                    user.face_template_enc, user.failed_logins, user.locked_until,
                ),
            )
            if wrapped_dek is not None:
                self.conn.execute(
                    "INSERT INTO key_slots (user_id, wrapped_dek) VALUES (?, ?)", (user.id, wrapped_dek)
                )

    def _user(self, where: str, value) -> UserRecord | None:
        row = self.conn.execute(f"SELECT * FROM users WHERE {where} = ?", (value,)).fetchone()
        return None if row is None else UserRecord(**dict(row))  # BLOB terbaca sebagai bytes

    def get_user(self, user_id: str) -> UserRecord | None:
        return self._user("id", user_id)

    def get_user_by_username(self, username: str) -> UserRecord | None:
        return self._user("username", username)

    def get_user_by_dongle(self, dongle_id: str) -> UserRecord | None:
        return self._user("dongle_id", dongle_id)

    def get_admin(self) -> UserRecord | None:
        return self._user("role", ROLE_ADMIN)

    def list_users(self) -> list[UserRecord]:
        ids = [r["id"] for r in self.conn.execute("SELECT id FROM users ORDER BY username")]
        return [self.get_user(i) for i in ids]

    def set_login_state(self, user_id: str, failed_logins: int, locked_until: float | None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                (failed_logins, locked_until, user_id),
            )

    def update_credentials(self, user_id: str, pw_hash: str, kek_salt: bytes, face_template_enc: str,
                           enc_privkey_wrapped: str | None) -> None:
        """Dipakai saat ganti password: semua yang bergantung pada KEK dibungkus ulang."""
        with self.conn:
            self.conn.execute(
                """UPDATE users SET pw_hash = ?, kek_salt = ?, face_template_enc = ?, enc_privkey_wrapped = ?
                   WHERE id = ?""",
                (pw_hash, kek_salt, face_template_enc, enc_privkey_wrapped, user_id),
            )

    # ---------------------------------------------------- key slots
    def put_key_slot(self, user_id: str, wrapped_dek: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO key_slots (user_id, wrapped_dek) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET wrapped_dek = excluded.wrapped_dek",
                (user_id, wrapped_dek),
            )

    def get_key_slot(self, user_id: str) -> str | None:
        row = self.conn.execute("SELECT wrapped_dek FROM key_slots WHERE user_id = ?", (user_id,)).fetchone()
        return row["wrapped_dek"] if row else None

    # ---------------------------------------------------- reports
    def add_report(self, dek: bytes, *, report_id: str, reporter_id: str, kind: str,
                   body_pkg: str | None, status: str = "baru", created_at: str | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"jenis laporan harus salah satu dari {KINDS}")
        if status not in STATUSES:
            raise ValueError(f"status harus salah satu dari {STATUSES}")
        body_enc = None if body_pkg is None else self._enc(dek, "reports", "body_pkg", report_id, body_pkg)
        with self.conn:
            self.conn.execute(
                "INSERT INTO reports (id, reporter_id, created_at, kind, body_pkg_enc, status_enc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (report_id, reporter_id, created_at or now_iso(), kind, body_enc,
                 self._enc(dek, "reports", "status", report_id, status)),
            )

    def _report(self, dek: bytes, row: sqlite3.Row) -> Report:
        rid = row["id"]
        body = row["body_pkg_enc"]
        return Report(
            id=rid,
            reporter_id=row["reporter_id"],
            created_at=row["created_at"],
            kind=row["kind"],
            body_pkg=None if body is None else self._dec(dek, "reports", "body_pkg", rid, body),
            status=self._dec(dek, "reports", "status", rid, row["status_enc"]),
        )

    def get_report(self, dek: bytes, report_id: str) -> Report | None:
        row = self.conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return None if row is None else self._report(dek, row)

    def list_reports(self, dek: bytes, reporter_id: str | None = None) -> list[Report]:
        if reporter_id is None:
            rows = self.conn.execute("SELECT * FROM reports ORDER BY created_at, id")
        else:
            rows = self.conn.execute(
                "SELECT * FROM reports WHERE reporter_id = ? ORDER BY created_at, id", (reporter_id,)
            )
        return [self._report(dek, r) for r in rows]

    def update_status(self, dek: bytes, report_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status harus salah satu dari {STATUSES}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE reports SET status_enc = ? WHERE id = ?",
                (self._enc(dek, "reports", "status", report_id, status), report_id),
            )
        if cur.rowcount != 1:
            raise KeyError("laporan tidak ditemukan")

    # ---------------------------------------------------- attachments
    def add_attachment(self, dek: bytes, *, attachment_id: str, report_id: str, enc_path: str,
                       size: int, filename: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO attachments (id, report_id, enc_path, size, filename_enc) VALUES (?, ?, ?, ?, ?)",
                (attachment_id, report_id, enc_path, size,
                 self._enc(dek, "attachments", "filename", attachment_id, filename)),
            )

    def list_attachments(self, dek: bytes, report_id: str) -> list[Attachment]:
        rows = self.conn.execute("SELECT * FROM attachments WHERE report_id = ? ORDER BY id", (report_id,))
        return [
            Attachment(r["id"], r["report_id"], r["enc_path"], r["size"],
                       self._dec(dek, "attachments", "filename", r["id"], r["filename_enc"]))
            for r in rows
        ]

    # ---------------------------------------------------- stego
    def add_stego_image(self, *, image_id: str, report_id: str, png_path: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO stego_images (id, report_id, png_path) VALUES (?, ?, ?)",
                (image_id, report_id, png_path),
            )

    def list_stego_images(self, report_id: str) -> list[StegoImage]:
        rows = self.conn.execute("SELECT * FROM stego_images WHERE report_id = ? ORDER BY id", (report_id,))
        return [StegoImage(r["id"], r["report_id"], r["png_path"]) for r in rows]

    # ---------------------------------------------------- chain
    def append_block(self, block: Block) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO chain (idx, timestamp, actor_id, action, asset_type, asset_hash, prev_hash, "
                "nonce, hash, signature) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (block.idx, block.timestamp, block.actor_id, block.action, block.asset_type,
                 block.asset_hash, block.prev_hash, block.nonce, block.hash, block.signature),
            )

    def get_chain(self) -> list[Block]:
        return [Block(**dict(r)) for r in self.conn.execute("SELECT * FROM chain ORDER BY idx")]

    def last_block(self) -> Block | None:
        row = self.conn.execute("SELECT * FROM chain ORDER BY idx DESC LIMIT 1").fetchone()
        return None if row is None else Block(**dict(row))
