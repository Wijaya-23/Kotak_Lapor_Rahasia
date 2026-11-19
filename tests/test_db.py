import os
import sqlite3

import pytest

from klr import db as dbm
from klr.db import Block, Database, UserRecord
from klr.errors import DecryptionError

DEK = bytes(range(32))


def _user(role="pelapor", username=None, **extra):
    uid = dbm.new_id()
    base = dict(
        id=uid,
        username=username or f"user-{uid[:8]}",
        role=role,
        pw_hash="$argon2id$v=19$m=65536,t=3,p=4$contoh",
        kek_salt=os.urandom(16),
        dongle_id=dbm.new_id(),
        dongle_serial="1A2B-3C4D",
        sign_pubkey="-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n",
        face_template_enc="AQ-template-terbungkus",
    )
    base.update(extra)
    return UserRecord(**base)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "klr.db")


@pytest.fixture
def db(path):
    with Database(path) as d:
        yield d


@pytest.fixture
def raw(path, db):
    """Koneksi SQLite langsung, seperti orang yang membuka berkas .db dengan DB Browser."""
    conn = sqlite3.connect(path)
    yield conn
    conn.close()


@pytest.fixture
def pelapor(db):
    u = _user(username="pelapor1")
    db.add_user(u)
    return u


def _report(db, pelapor, body="PAKET-HYBRID-BASE64", status="baru", kind="teks"):
    rid = dbm.new_id()
    db.add_report(DEK, report_id=rid, reporter_id=pelapor.id, kind=kind, body_pkg=body, status=status)
    return rid


# ------------------------------------------------------------ skema dan users
def test_skema_dibuat_dan_bisa_dibuka_ulang(path):
    with Database(path) as d:
        assert d.get_meta("schema_version") == "1"
        d.add_user(_user(username="a"))
    with Database(path) as d:  # membuka ulang tidak menghapus data
        assert d.get_user_by_username("a") is not None


def test_simpan_dan_ambil_user(db):
    u = _user(role="admin", username="admin", enc_pubkey="PEM", enc_privkey_wrapped="terbungkus")
    db.add_user(u)
    assert db.get_user(u.id) == u
    assert db.get_user_by_username("admin") == u
    assert db.get_user_by_dongle(u.dongle_id) == u
    assert db.get_admin() == u
    assert isinstance(db.get_user(u.id).kek_salt, bytes)


def test_belum_diinisialisasi_sampai_ada_admin(db):
    assert not db.is_initialized()
    db.add_user(_user(role="pelapor"))
    assert not db.is_initialized()
    db.add_user(_user(role="admin"))
    assert db.is_initialized()


def test_hanya_satu_admin(db):
    db.add_user(_user(role="admin"))
    with pytest.raises(sqlite3.IntegrityError):
        db.add_user(_user(role="admin"))


@pytest.mark.parametrize("field", ["username", "dongle_id"])
def test_username_dan_dongle_unik(db, field):
    a = _user()
    db.add_user(a)
    with pytest.raises(sqlite3.IntegrityError):
        db.add_user(_user(**{field: getattr(a, field)}))


def test_peran_tidak_dikenal_ditolak(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_user(_user(role="superuser"))


def test_status_login_dan_penguncian(db, pelapor):
    db.set_login_state(pelapor.id, 5, 1_900_000_000.0)
    u = db.get_user(pelapor.id)
    assert (u.failed_logins, u.locked_until) == (5, 1_900_000_000.0)
    db.set_login_state(pelapor.id, 0, None)
    u = db.get_user(pelapor.id)
    assert (u.failed_logins, u.locked_until) == (0, None)


def test_tabel_users_tidak_menyimpan_password_mentah(db, raw):
    """Skenario uji nomor 2: buka tabel users, hanya ada hash Argon2id."""
    db.add_user(_user(username="x", pw_hash="$argon2id$v=19$hash-saja"))
    cur = raw.execute("SELECT * FROM users")
    row = cur.fetchone()
    cols = [c[0] for c in cur.description]
    assert "password" not in cols
    assert row[cols.index("pw_hash")].startswith("$argon2id$")


# ------------------------------------------------------------ key slots
def test_key_slot_simpan_ganti_ambil(db, pelapor):
    assert db.get_key_slot(pelapor.id) is None
    db.put_key_slot(pelapor.id, "slot-1")
    db.put_key_slot(pelapor.id, "slot-2")  # ganti password: slot dibungkus ulang
    assert db.get_key_slot(pelapor.id) == "slot-2"


def test_user_dan_slot_disimpan_dalam_satu_transaksi(db):
    a = _user()
    db.add_user(a, wrapped_dek="slot-a")
    assert db.get_key_slot(a.id) == "slot-a"
    b = _user()
    with pytest.raises(sqlite3.Error):
        db.add_user(b, wrapped_dek=object())  # penyimpanan slot gagal di tengah jalan...
    assert db.get_user(b.id) is None  # ...maka pengguna juga ikut dibatalkan


def test_key_slot_harus_milik_user_yang_ada(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.put_key_slot("tidak-ada", "slot")


# ------------------------------------------------------------ laporan
def test_laporan_round_trip(db, pelapor):
    rid = _report(db, pelapor)
    r = db.get_report(DEK, rid)
    assert (r.reporter_id, r.kind, r.body_pkg, r.status) == (pelapor.id, "teks", "PAKET-HYBRID-BASE64", "baru")


def test_daftar_laporan_per_pelapor(db, pelapor):
    lain = _user(username="pelapor2")
    db.add_user(lain)
    a, b = _report(db, pelapor), _report(db, pelapor)
    _report(db, lain)
    assert {r.id for r in db.list_reports(DEK, pelapor.id)} == {a, b}
    assert len(db.list_reports(DEK)) == 3


def test_ubah_status(db, pelapor):
    rid = _report(db, pelapor)
    db.update_status(DEK, rid, "diproses")
    assert db.get_report(DEK, rid).status == "diproses"
    with pytest.raises(ValueError):
        db.update_status(DEK, rid, "dihapus")
    with pytest.raises(KeyError):
        db.update_status(DEK, "tidak-ada", "selesai")


def test_laporan_tanpa_isi_teks(db, pelapor):
    """Laporan berjenis file/gambar boleh tanpa paket teks."""
    rid = _report(db, pelapor, body=None, kind="file")
    assert db.get_report(DEK, rid).body_pkg is None


@pytest.mark.parametrize("kw", [{"kind": "video"}, {"status": "rahasia"}])
def test_jenis_dan_status_divalidasi(db, pelapor, kw):
    with pytest.raises(ValueError):
        db.add_report(DEK, report_id=dbm.new_id(), reporter_id=pelapor.id,
                      kind=kw.get("kind", "teks"), body_pkg="x", status=kw.get("status", "baru"))


def test_laporan_harus_dari_user_yang_ada(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_report(DEK, report_id=dbm.new_id(), reporter_id="tidak-ada", kind="teks", body_pkg="x")


# ------------------------------------------------------------ enkripsi kolom
def test_berkas_db_tidak_memuat_plaintext(db, pelapor, path):
    """Skenario uji nomor 5: berkas SQLite dibuka langsung, kolom sensitif berupa ciphertext."""
    rid = _report(db, pelapor, body="ISI-PAKET-RAHASIA-123", status="diproses")
    db.add_attachment(DEK, attachment_id=dbm.new_id(), report_id=rid, enc_path="files/acak.enc",
                      size=10, filename="bukti_penyimpangan_gudang.xlsx")
    db.close()
    raw = open(path, "rb").read()
    for rahasia in [b"ISI-PAKET-RAHASIA-123", b"bukti_penyimpangan_gudang", b"diproses"]:
        assert rahasia not in raw
    assert b"files/acak.enc" in raw  # kolom terbuka memang terbuka


def test_dek_salah_tidak_bisa_membaca(db, pelapor):
    rid = _report(db, pelapor)
    with pytest.raises(DecryptionError):
        db.get_report(bytes(32), rid)


def test_nilai_terenkripsi_dipindah_ke_laporan_lain_ditolak(db, pelapor, raw):
    """Penyerang dengan akses berkas menukar status_enc laporan A ke laporan B."""
    a = _report(db, pelapor, status="ditolak")
    b = _report(db, pelapor, status="baru")
    sa = raw.execute("SELECT status_enc FROM reports WHERE id = ?", (a,)).fetchone()[0]
    with raw:
        raw.execute("UPDATE reports SET status_enc = ? WHERE id = ?", (sa, b))
    with pytest.raises(DecryptionError):
        db.get_report(DEK, b)


def test_nilai_dipindah_ke_kolom_lain_ditolak(db, pelapor, raw):
    """status_enc ditempel ke body_pkg_enc pada baris yang sama: AAD kolom berbeda."""
    a = _report(db, pelapor)
    with raw:
        raw.execute("UPDATE reports SET body_pkg_enc = status_enc WHERE id = ?", (a,))
    with pytest.raises(DecryptionError):
        db.get_report(DEK, a)


def test_isi_terenkripsi_diubah_terdeteksi(db, pelapor, raw):
    a = _report(db, pelapor)
    blob = raw.execute("SELECT body_pkg_enc FROM reports WHERE id = ?", (a,)).fetchone()[0]
    rusak = blob[:-6] + ("A" if blob[-6] != "A" else "B") + blob[-5:]
    with raw:
        raw.execute("UPDATE reports SET body_pkg_enc = ? WHERE id = ?", (rusak, a))
    with pytest.raises(DecryptionError):
        db.get_report(DEK, a)


# ------------------------------------------------------------ lampiran, stego, rantai
def test_lampiran_round_trip(db, pelapor):
    rid = _report(db, pelapor, body=None, kind="file")
    aid = dbm.new_id()
    db.add_attachment(DEK, attachment_id=aid, report_id=rid, enc_path="files/x.enc", size=1234,
                      filename="Laporan Keuangan Q3.pdf")
    [att] = db.list_attachments(DEK, rid)
    assert (att.id, att.enc_path, att.size, att.filename) == (aid, "files/x.enc", 1234, "Laporan Keuangan Q3.pdf")


def test_stego_round_trip(db, pelapor):
    rid = _report(db, pelapor, body=None, kind="gambar")
    db.add_stego_image(image_id="img-1", report_id=rid, png_path="stego/img-1.png")
    assert [s.png_path for s in db.list_stego_images(rid)] == ["stego/img-1.png"]


def _block(i, prev="0" * 64):
    return Block(idx=i, timestamp="2026-11-19T00:00:00+00:00", actor_id="u", action="kirim_laporan",
                 asset_type="laporan", asset_hash="ab" * 32, prev_hash=prev, nonce=42,
                 hash=f"{i:064x}", signature="sig")


def test_rantai_urut_dan_blok_terakhir(db):
    assert db.last_block() is None
    for i in range(3):
        db.append_block(_block(i))
    assert [b.idx for b in db.get_chain()] == [0, 1, 2]
    assert db.last_block().idx == 2


def test_rantai_idx_tidak_boleh_ganda(db):
    db.append_block(_block(0))
    with pytest.raises(sqlite3.IntegrityError):
        db.append_block(_block(0))
