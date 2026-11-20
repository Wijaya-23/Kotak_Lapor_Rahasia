import hashlib
import os

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from klr import filecrypto as fc
from klr.errors import DecryptionError

CHUNK = 1024  # potongan kecil supaya tes cepat; satu tes memakai 1 MiB asli
CTX = "lampiran-0001"


@pytest.fixture(scope="module")
def admin_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def dirs(tmp_path):
    d = {k: tmp_path / k for k in ("src", "enc", "out")}
    for p in d.values():
        p.mkdir()
    return {k: str(v) for k, v in d.items()}


def _make(dirs, name, data):
    path = os.path.join(dirs["src"], name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _enc(dirs, key, data, name="bukti.pdf", chunk=CHUNK, ctx=CTX, **kw):
    return fc.encrypt_file(_make(dirs, name, data), dirs["enc"], key.public_key(), context=ctx, chunk_size=chunk, **kw)


def _dec(dirs, key, enc_path, ctx=CTX):
    return fc.decrypt_file(enc_path, key, dirs["out"], context=ctx)


def _header_len(key):
    return 4 + 4 + key.key_size // 8 + 4


def _layout(enc_path, key, chunk=CHUNK):
    """Posisi awal potongan data dan daftar (awal, akhir) tiap potongan."""
    raw = open(enc_path, "rb").read()
    h = _header_len(key)
    meta_len = int.from_bytes(raw[h : h + 4], "big")
    start = h + 4 + meta_len
    blocks, pos = [], start
    while pos < len(raw):
        end = min(pos + chunk + 16, len(raw))
        blocks.append((pos, end))
        pos = end
    return raw, blocks


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


# ------------------------------------------------------------ round trip
@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK, 5 * CHUNK + 17])
def test_round_trip_berbagai_ukuran(dirs, admin_key, size):
    data = os.urandom(size)
    ef = _enc(dirs, admin_key, data)
    out, meta = _dec(dirs, admin_key, ef.enc_path)
    assert open(out, "rb").read() == data
    assert (meta.name, meta.size, meta.sha256) == ("bukti.pdf", size, hashlib.sha256(data).hexdigest())


def test_potongan_1_mib_asli(dirs, admin_key):
    data = os.urandom(2 * fc.CHUNK_SIZE + 12345)
    ef = _enc(dirs, admin_key, data, chunk=fc.CHUNK_SIZE)
    out, _ = _dec(dirs, admin_key, ef.enc_path)
    assert hashlib.sha256(open(out, "rb").read()).digest() == hashlib.sha256(data).digest()


def test_format_header_sesuai_laporan(dirs, admin_key):
    ef = _enc(dirs, admin_key, b"isi", chunk=fc.CHUNK_SIZE)
    raw = open(ef.enc_path, "rb").read()
    assert raw[:4] == b"KLF\x01"  # penanda + versi, 4 byte
    assert int.from_bytes(raw[4 + 4 + 256 : 4 + 4 + 256 + 4], "big") == 1024 * 1024  # ukuran potongan


def test_nama_acak_dan_isi_tidak_terbaca(dirs, admin_key):
    data = b"LAPORAN-KEUANGAN-RAHASIA " * 200
    ef = _enc(dirs, admin_key, data, name="laporan_keuangan_gudang.xlsx")
    base = os.path.basename(ef.enc_path)
    assert base.endswith(".enc") and len(base) == 32 + 4
    raw = open(ef.enc_path, "rb").read()
    assert b"LAPORAN-KEUANGAN" not in raw
    assert b"laporan_keuangan_gudang" not in raw


def test_dua_kali_enkripsi_menghasilkan_berkas_berbeda(dirs, admin_key):
    a = _enc(dirs, admin_key, b"sama", name="a.txt")
    b = _enc(dirs, admin_key, b"sama", name="a.txt")
    assert a.enc_path != b.enc_path
    assert a.enc_sha256 != b.enc_sha256


def test_sha256_berkas_enc_benar(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(3000))
    assert ef.enc_sha256 == hashlib.sha256(open(ef.enc_path, "rb").read()).hexdigest()
    assert ef.enc_size == os.path.getsize(ef.enc_path)


def test_baca_metadata_tanpa_dekripsi_isi(dirs, admin_key):
    ef = _enc(dirs, admin_key, b"x" * 5000, name="foto bukti.jpg")
    meta = fc.read_metadata(ef.enc_path, admin_key, context=CTX)
    assert (meta.name, meta.size) == ("foto bukti.jpg", 5000)


# ------------------------------------------------------------ manipulasi (skenario uji nomor 4)
def _expect_reject(dirs, key, path):
    before = set(os.listdir(dirs["out"]))
    with pytest.raises(DecryptionError):
        _dec(dirs, key, path)
    assert set(os.listdir(dirs["out"])) == before  # tidak ada sisa berkas hasil


def test_tukar_dua_potongan_terdeteksi(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(4 * CHUNK))
    raw, blocks = _layout(ef.enc_path, admin_key)
    (a0, a1), (b0, b1) = blocks[1], blocks[2]
    _write(ef.enc_path, raw[:a0] + raw[b0:b1] + raw[a0:a1] + raw[b1:])
    _expect_reject(dirs, admin_key, ef.enc_path)


def test_potongan_terakhir_dibuang_terdeteksi(dirs, admin_key):
    """Dipotong tepat di batas potongan: tanpa penanda akhir, ini tidak akan ketahuan."""
    ef = _enc(dirs, admin_key, os.urandom(3 * CHUNK + 10))
    raw, blocks = _layout(ef.enc_path, admin_key)
    _write(ef.enc_path, raw[: blocks[-1][0]])
    _expect_reject(dirs, admin_key, ef.enc_path)


def test_dipotong_di_tengah_terdeteksi(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(3 * CHUNK))
    raw = open(ef.enc_path, "rb").read()
    _write(ef.enc_path, raw[:-100])
    _expect_reject(dirs, admin_key, ef.enc_path)


def test_potongan_tambahan_terdeteksi(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(2 * CHUNK + 5))
    raw, blocks = _layout(ef.enc_path, admin_key)
    s, e = blocks[0]
    _write(ef.enc_path, raw + raw[s:e])
    _expect_reject(dirs, admin_key, ef.enc_path)


def test_potongan_digandakan_terdeteksi(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(3 * CHUNK))
    raw, blocks = _layout(ef.enc_path, admin_key)
    (s0, e0), (s1, e1) = blocks[0], blocks[1]
    _write(ef.enc_path, raw[:s1] + raw[s0:e0] + raw[e1:])
    _expect_reject(dirs, admin_key, ef.enc_path)


def test_satu_bit_berubah_di_mana_pun_terdeteksi(dirs, admin_key):
    ef = _enc(dirs, admin_key, os.urandom(2 * CHUNK + 50))
    raw = open(ef.enc_path, "rb").read()
    h = _header_len(admin_key)
    posisi = [0, 3, 5, 100, h - 1, h + 2, h + 10, h + 4 + 40, len(raw) // 2, len(raw) - 1]
    for i in posisi:
        rusak = bytearray(raw)
        rusak[i] ^= 0x01
        _write(ef.enc_path, bytes(rusak))
        _expect_reject(dirs, admin_key, ef.enc_path)


def test_berkas_terlalu_pendek_atau_sampah(dirs, admin_key):
    path = os.path.join(dirs["enc"], "sampah.enc")
    for isi in [b"", b"KLF\x01", b"bukan berkas klr" * 100]:
        _write(path, isi)
        _expect_reject(dirs, admin_key, path)


# ------------------------------------------------------------ kunci dan konteks
def test_kunci_privat_lain_ditolak(dirs, admin_key, other_key):
    ef = _enc(dirs, admin_key, b"rahasia")
    _expect_reject(dirs, other_key, ef.enc_path)


def test_berkas_ditukar_ke_lampiran_lain_ditolak(dirs, admin_key):
    """Konteks (id lampiran) ikut di AAD: .enc milik lampiran A tidak bisa dipasang ke lampiran B."""
    ef = _enc(dirs, admin_key, b"rahasia", ctx="lampiran-A")
    with pytest.raises(DecryptionError):
        fc.decrypt_file(ef.enc_path, admin_key, dirs["out"], context="lampiran-B")
    with pytest.raises(DecryptionError):
        fc.read_metadata(ef.enc_path, admin_key, context="lampiran-B")


# ------------------------------------------------------------ batas dan nama berkas
def test_batas_ukuran(dirs, admin_key):
    with pytest.raises(ValueError, match="terlalu besar"):
        _enc(dirs, admin_key, b"x" * 2001, max_size=2000)
    assert os.listdir(dirs["enc"]) == []


def test_ukuran_potongan_tidak_wajar_ditolak(dirs, admin_key):
    with pytest.raises(ValueError):
        _enc(dirs, admin_key, b"x", chunk=10)


@pytest.mark.parametrize(
    "nama, aman",
    [
        ("..\\..\\Windows\\System32\\jahat.dll", "jahat.dll"),
        ("../../etc/passwd", "passwd"),
        ("..", "lampiran"),
        ("", "lampiran"),
        ("  laporan akhir.pdf  ", "laporan akhir.pdf"),
        ("C:\\Users\\x\\bukti.png", "bukti.png"),
    ],
)
def test_nama_berkas_dibersihkan(nama, aman):
    assert fc.safe_name(nama) == aman


def test_nama_berbahaya_tidak_keluar_folder_tujuan(dirs, admin_key):
    ef = _enc(dirs, admin_key, b"isi", name="..\\..\\..\\jahat.txt")
    out, meta = _dec(dirs, admin_key, ef.enc_path)
    assert os.path.dirname(out) == dirs["out"]
    assert meta.name == "jahat.txt"


def test_nama_yang_sudah_ada_tidak_ditimpa(dirs, admin_key):
    ef = _enc(dirs, admin_key, b"versi 1", name="sama.txt")
    out1, _ = _dec(dirs, admin_key, ef.enc_path)
    out2, _ = _dec(dirs, admin_key, ef.enc_path)
    assert out1 != out2
    assert os.path.basename(out2) == "sama (1).txt"
    assert open(out1, "rb").read() == open(out2, "rb").read() == b"versi 1"


def test_hash_isi_tidak_cocok_dengan_metadata_ditolak(dirs, admin_key):
    """Laporan 3.4.3: Pelapor nakal membuat .enc yang tag-nya sah tetapi SHA-256 di metadata palsu."""
    import json

    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    k_f, prefix = os.urandom(32), os.urandom(4)
    header = b"KLF\x01" + prefix + admin_key.public_key().encrypt(k_f, fc._OAEP) + CHUNK.to_bytes(4, "big")
    aead = ChaCha20Poly1305(k_f)
    meta = json.dumps({"name": "x.txt", "size": 3, "sha256": "00" * 32}).encode()
    meta_ct = aead.encrypt(fc._nonce(prefix, 0), meta, fc._meta_aad(header, CTX))
    chunk = aead.encrypt(fc._nonce(prefix, 1), b"abc", fc._chunk_aad(header, CTX, 0, True))
    path = os.path.join(dirs["enc"], "palsu.enc")
    _write(path, header + len(meta_ct).to_bytes(4, "big") + meta_ct + chunk)
    _expect_reject(dirs, admin_key, path)


def test_aad_potongan_membedakan_indeks_penanda_akhir_dan_konteks():
    """Pertahanan berlapis: tiap potongan terikat pada posisinya, statusnya, dan lampirannya,
    bukan hanya bergantung pada pemeriksaan SHA-256 di akhir."""
    h = b"header"
    base = fc._chunk_aad(h, "A", 0, False)
    assert base != fc._chunk_aad(h, "A", 1, False)
    assert base != fc._chunk_aad(h, "A", 0, True)
    assert base != fc._chunk_aad(h, "B", 0, False)
    assert base != fc._chunk_aad(b"header-lain", "A", 0, False)
    assert fc._nonce(b"\x00" * 4, 1) != fc._nonce(b"\x00" * 4, 2)
