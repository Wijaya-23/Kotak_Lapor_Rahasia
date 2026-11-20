"""Enkripsi file dengan ChaCha20-Poly1305 per potongan (laporan subbab 3.4).

Format berkas .enc:

    +---------------------+---------+-------------------------------------------+
    | penanda + versi     | 4 byte  | b"KLF" + 0x01                             |
    | prefiks nonce       | 4 byte  | acak, satu per berkas                     |
    | kunci terbungkus    | 256 B   | K_f (32 byte) dibungkus RSA-OAEP Admin    |
    | ukuran potongan     | 4 byte  | bawaan 1 MiB                              |
    +---------------------+---------+-------------------------------------------+  <- header
    | panjang metadata    | 4 byte  |                                           |
    | blok metadata       | var.    | {nama, ukuran, sha256} terenkripsi        |
    | potongan 0..n-1     | var.    | ciphertext + tag 16 byte per potongan     |
    +---------------------+---------+-------------------------------------------+

Nonce   = prefiks (4 byte) || penghitung (8 byte). Penghitung 0 = metadata,
          penghitung i+1 = potongan ke-i. K_f baru untuk setiap berkas.
AAD     = header || konteks || jenis blok (|| indeks potongan || penanda akhir)
          - header di AAD: header yang diubah (misalnya ukuran potongan) terdeteksi
          - konteks (id lampiran): berkas .enc yang ditukar antar lampiran terdeteksi
          - indeks: potongan yang ditukar urutannya terdeteksi
          - penanda akhir: potongan terakhir yang dibuang atau potongan tambahan terdeteksi
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from .aead import build_aad
from .errors import DecryptionError

MAGIC = b"KLF"
VERSION = 1
CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_FILE_SIZE = 50 * 1024 * 1024  # parameter aplikasi (laporan 3.4.3)
KEY_LEN = 32
NONCE_PREFIX_LEN = 4
TAG_LEN = 16
_MIN_CHUNK, _MAX_CHUNK = 1024, 16 * 1024 * 1024

_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
_FAIL = "berkas rusak, dimanipulasi, atau bukan untuk kunci ini"


@dataclass(frozen=True)
class FileMeta:
    name: str
    size: int
    sha256: str  # hex, isi ASLI


@dataclass(frozen=True)
class EncryptedFile:
    enc_path: str
    enc_size: int
    enc_sha256: str  # hex, berkas .enc (dicatat di blockchain)
    meta: FileMeta


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + counter.to_bytes(8, "big")


def _meta_aad(header: bytes, context: str) -> bytes:
    return header + build_aad(context) + b"META"


def _chunk_aad(header: bytes, context: str, index: int, final: bool) -> bytes:
    return header + build_aad(context) + b"DATA" + index.to_bytes(8, "big") + (b"\x01" if final else b"\x00")


def _sha256_file(path: str) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK_SIZE):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def safe_name(name: str) -> str:
    """Nama berkas aman untuk disimpan: tanpa folder, tanpa '..'.

    Metadata ditulis oleh Pelapor. Pelapor nakal bisa memberi nama seperti
    '..\\..\\Windows\\x.dll'; tanpa pembersihan, Admin yang menyimpan lampiran
    akan menulis ke luar folder tujuan.
    """
    base = name.replace("\\", "/").split("/")[-1].strip().strip(".")
    return base or "lampiran"


# --------------------------------------------------------------------------
# Enkripsi (Pelapor: hanya butuh kunci PUBLIK Admin)
# --------------------------------------------------------------------------
def encrypt_file(
    src_path: str,
    out_dir: str,
    admin_public_key: rsa.RSAPublicKey,
    *,
    context: str,
    name: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_size: int = MAX_FILE_SIZE,
) -> EncryptedFile:
    if not (_MIN_CHUNK <= chunk_size <= _MAX_CHUNK):
        raise ValueError("ukuran potongan tidak wajar")
    size = os.path.getsize(src_path)
    if size > max_size:
        raise ValueError(f"file terlalu besar ({size} byte, batas {max_size} byte)")
    content_sha, _ = _sha256_file(src_path)  # lintasan 1: hash untuk metadata

    k_f = os.urandom(KEY_LEN)
    prefix = os.urandom(NONCE_PREFIX_LEN)
    wrapped = admin_public_key.encrypt(k_f, _OAEP)
    header = MAGIC + bytes([VERSION]) + prefix + wrapped + chunk_size.to_bytes(4, "big")
    aead = ChaCha20Poly1305(k_f)

    meta = FileMeta(name=safe_name(name or os.path.basename(src_path)), size=size, sha256=content_sha)
    meta_ct = aead.encrypt(_nonce(prefix, 0), json.dumps(asdict(meta)).encode("utf-8"), _meta_aad(header, context))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, secrets.token_hex(16) + ".enc")  # nama acak
    tmp = out_path + ".tmp"
    out_hash, check_hash, total = hashlib.sha256(), hashlib.sha256(), 0
    try:
        with open(src_path, "rb") as fin, open(tmp, "wb") as fout:

            def write(data: bytes) -> None:
                fout.write(data)
                out_hash.update(data)

            write(header)
            write(len(meta_ct).to_bytes(4, "big"))
            write(meta_ct)
            index, current = 0, fin.read(chunk_size)
            while True:  # lintasan 2: enkripsi
                nxt = fin.read(chunk_size)
                final = not nxt
                check_hash.update(current)
                total += len(current)
                write(aead.encrypt(_nonce(prefix, index + 1), current, _chunk_aad(header, context, index, final)))
                if final:
                    break
                current, index = nxt, index + 1
        if total != size or check_hash.hexdigest() != content_sha:
            raise ValueError("file berubah saat sedang dienkripsi. Coba lagi.")
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return EncryptedFile(out_path, os.path.getsize(out_path), out_hash.hexdigest(), meta)


# --------------------------------------------------------------------------
# Dekripsi (Admin: butuh kunci PRIVAT)
# --------------------------------------------------------------------------
def _open(fh, admin_private_key: rsa.RSAPrivateKey, context: str):
    """Baca header dan metadata. Mengembalikan (aead, header, prefix, chunk_size, meta)."""
    wrapped_len = admin_private_key.key_size // 8
    head = fh.read(4 + NONCE_PREFIX_LEN + wrapped_len + 4)
    if len(head) != 4 + NONCE_PREFIX_LEN + wrapped_len + 4 or head[:4] != MAGIC + bytes([VERSION]):
        raise DecryptionError(_FAIL)
    prefix = head[4:8]
    wrapped = head[8 : 8 + wrapped_len]
    chunk_size = int.from_bytes(head[-4:], "big")
    if not (_MIN_CHUNK <= chunk_size <= _MAX_CHUNK):
        raise DecryptionError(_FAIL)
    try:
        k_f = admin_private_key.decrypt(wrapped, _OAEP)
        if len(k_f) != KEY_LEN:
            raise ValueError
        aead = ChaCha20Poly1305(k_f)
        meta_len = int.from_bytes(fh.read(4), "big")
        if not (TAG_LEN <= meta_len <= 64 * 1024):
            raise ValueError
        meta_raw = aead.decrypt(_nonce(prefix, 0), fh.read(meta_len), _meta_aad(head, context))
        m = json.loads(meta_raw.decode("utf-8"))
        meta = FileMeta(name=safe_name(str(m["name"])), size=int(m["size"]), sha256=str(m["sha256"]))
    except (ValueError, InvalidTag, KeyError, TypeError):
        raise DecryptionError(_FAIL) from None
    return aead, head, prefix, chunk_size, meta


def read_metadata(enc_path: str, admin_private_key: rsa.RSAPrivateKey, *, context: str) -> FileMeta:
    with open(enc_path, "rb") as fh:
        return _open(fh, admin_private_key, context)[4]


def _free_path(out_dir: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    candidate, n = os.path.join(out_dir, name), 1
    while os.path.exists(candidate):
        candidate = os.path.join(out_dir, f"{stem} ({n}){ext}")
        n += 1
    return candidate


def decrypt_file(
    enc_path: str, admin_private_key: rsa.RSAPrivateKey, out_dir: str, *, context: str
) -> tuple[str, FileMeta]:
    """Dekripsi ke `out_dir` dengan nama aslinya. Jika ada satu potongan saja yang gagal,
    tidak ada berkas hasil yang tertinggal."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, secrets.token_hex(8) + ".part")
    try:
        with open(enc_path, "rb") as fh, open(tmp, "wb") as fout:
            aead, header, prefix, chunk_size, meta = _open(fh, admin_private_key, context)
            block_len = chunk_size + TAG_LEN
            h, total, index = hashlib.sha256(), 0, 0
            block = fh.read(block_len)
            if not block:
                raise DecryptionError(_FAIL)  # minimal harus ada satu potongan akhir
            while True:
                nxt = fh.read(block_len)
                final = not nxt
                try:
                    plain = aead.decrypt(_nonce(prefix, index + 1), block, _chunk_aad(header, context, index, final))
                except InvalidTag:
                    raise DecryptionError(_FAIL) from None
                fout.write(plain)
                h.update(plain)
                total += len(plain)
                if final:
                    break
                block, index = nxt, index + 1
        if total != meta.size or h.hexdigest() != meta.sha256:  # laporan 3.4.3
            raise DecryptionError(_FAIL)
        out_path = _free_path(out_dir, meta.name)
        os.replace(tmp, out_path)
        return out_path, meta
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
