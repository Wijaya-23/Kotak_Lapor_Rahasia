import base64
import os

import pytest
from cryptography.hazmat.primitives import serialization

from klr import aead, keys, kdf, textcrypto
from klr.errors import DecryptionError
from klr.kdf import Argon2Params

FAST = Argon2Params(memory_kib=1024, time_cost=1, parallelism=1)


@pytest.fixture(scope="module")
def admin():
    """Admin lengkap: password, dongle_secret, salt KEK, KEK, dan pasangan kunci enkripsi."""
    a = {"id": "user-admin", "password": "Admin#Kuat-2026", "secret": os.urandom(32), "salt": kdf.new_salt()}
    a["kek"] = kdf.derive_kek(a["password"], a["salt"], a["secret"], FAST)
    a["privkey"] = keys.generate_encryption_keypair()
    return a


@pytest.fixture(scope="module")
def pelapor():
    p = {"id": "user-pelapor", "password": "Pelapor!2026", "secret": os.urandom(32), "salt": kdf.new_salt()}
    p["kek"] = kdf.derive_kek(p["password"], p["salt"], p["secret"], FAST)
    return p


# ------------------------------------------------------------ kunci privat Admin
def test_wrap_unwrap_kunci_privat(admin):
    blob = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    back = keys.unwrap_private_key(blob, admin["kek"], admin["id"])
    assert back.private_numbers() == admin["privkey"].private_numbers()


def test_bungkusan_tidak_memuat_kunci_privat_polos(admin):
    der = admin["privkey"].private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    blob = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    assert "PRIVATE" not in blob
    assert der[40:80] not in base64.b64decode(blob)


def test_kunci_privat_tidak_terbuka_dengan_password_salah(admin):
    blob = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    kek_salah = kdf.derive_kek("password-salah", admin["salt"], admin["secret"], FAST)
    with pytest.raises(DecryptionError):
        keys.unwrap_private_key(blob, kek_salah, admin["id"])


def test_kunci_privat_tidak_terbuka_tanpa_dongle_yang_benar(admin):
    """Password benar, dongle lain: KEK berbeda, kunci privat tetap tertutup."""
    blob = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    kek_dongle_lain = kdf.derive_kek(admin["password"], admin["salt"], os.urandom(32), FAST)
    with pytest.raises(DecryptionError):
        keys.unwrap_private_key(blob, kek_dongle_lain, admin["id"])


def test_kunci_privat_terikat_pada_user_id(admin):
    blob = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_private_key(blob, admin["kek"], "user-lain")


def test_kunci_publik_pem(admin):
    pem = keys.encryption_public_key_pem(admin["privkey"])
    assert pem.startswith("-----BEGIN PUBLIC KEY-----")
    pub = keys.load_encryption_public_key(pem)
    assert pub.public_numbers() == admin["privkey"].public_key().public_numbers()


# ------------------------------------------------------------ DEK dan key slot
def test_dek_acak_32_byte():
    a, b = keys.new_dek(), keys.new_dek()
    assert len(a) == len(b) == 32
    assert a != b


def test_tiap_pengguna_membuka_dek_yang_sama_dengan_kek_masing_masing(admin, pelapor):
    dek = keys.new_dek()
    slot_admin = keys.wrap_dek(dek, admin["kek"], admin["id"])
    slot_pelapor = keys.wrap_dek(dek, pelapor["kek"], pelapor["id"])
    assert slot_admin != slot_pelapor
    assert keys.unwrap_dek(slot_admin, admin["kek"], admin["id"]) == dek
    assert keys.unwrap_dek(slot_pelapor, pelapor["kek"], pelapor["id"]) == dek


def test_slot_orang_lain_tidak_bisa_dibuka(admin, pelapor):
    slot_admin = keys.wrap_dek(keys.new_dek(), admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_dek(slot_admin, pelapor["kek"], pelapor["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_dek(slot_admin, pelapor["kek"], admin["id"])


def test_slot_dipindah_ke_baris_lain_ditolak(admin):
    """Walau KEK sama, slot yang ditempel ke user_id lain ditolak karena AAD."""
    slot = keys.wrap_dek(keys.new_dek(), admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_dek(slot, admin["kek"], "user-pelapor")


def test_ganti_password_cukup_bungkus_ulang_slot(pelapor):
    """Ganti password tidak mengubah DEK, jadi basis data tidak perlu dienkripsi ulang."""
    dek = keys.new_dek()
    slot_lama = keys.wrap_dek(dek, pelapor["kek"], pelapor["id"])
    salt_baru = kdf.new_salt()
    kek_baru = kdf.derive_kek("Password-Baru!", salt_baru, pelapor["secret"], FAST)
    slot_baru = keys.wrap_dek(keys.unwrap_dek(slot_lama, pelapor["kek"], pelapor["id"]), kek_baru, pelapor["id"])
    assert keys.unwrap_dek(slot_baru, kek_baru, pelapor["id"]) == dek
    with pytest.raises(DecryptionError):
        keys.unwrap_dek(slot_baru, pelapor["kek"], pelapor["id"])


def test_panjang_dek_divalidasi(admin):
    with pytest.raises(ValueError):
        keys.wrap_dek(bytes(16), admin["kek"], admin["id"])


# ------------------------------------------------------------ pemisahan sub-kunci
def test_bungkusan_tidak_bisa_dibuka_sebagai_fungsi_lain(admin):
    """Satu KEK, tiga sub-kunci: slot DEK tidak bisa dibuka sebagai template wajah, dan sebaliknya."""
    slot = keys.wrap_dek(keys.new_dek(), admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_face_template(slot, admin["kek"], admin["id"])
    tmpl = keys.wrap_face_template(os.urandom(1024), admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_dek(tmpl, admin["kek"], admin["id"])
    with pytest.raises(DecryptionError):
        keys.unwrap_private_key(tmpl, admin["kek"], admin["id"])


@pytest.mark.parametrize(
    "wrap, info, purpose",
    [
        (lambda k, u: keys.wrap_dek(bytes(32), k, u), kdf.INFO_DEK_SLOT, "klr-dek-slot"),
        (lambda k, u: keys.wrap_face_template(b"tmpl", k, u), kdf.INFO_FACE_TEMPLATE, "klr-face-template"),
    ],
    ids=["slot-dek", "template-wajah"],
)
def test_tiap_bungkusan_memakai_sub_kunci_miliknya_bukan_kek_langsung(admin, wrap, info, purpose):
    blob = wrap(admin["kek"], admin["id"])
    aad = aead.build_aad(purpose, admin["id"])
    aead.open_sealed(kdf.derive_subkey(admin["kek"], info), blob, aad)  # sub-kunci yang benar: terbuka
    with pytest.raises(DecryptionError):
        aead.open_sealed(admin["kek"], blob, aad)  # KEK mentah: tertutup
    with pytest.raises(DecryptionError):
        aead.open_sealed(kdf.derive_subkey(admin["kek"], kdf.INFO_PRIVKEY_ENC), blob, aad)


def test_template_wajah_round_trip(admin):
    template = os.urandom(128 * 8)  # 128 angka float64
    blob = keys.wrap_face_template(template, admin["kek"], admin["id"])
    assert keys.unwrap_face_template(blob, admin["kek"], admin["id"]) == template


# ------------------------------------------------------------ alur penuh
def test_alur_penuh_pelapor_kirim_admin_baca(admin, pelapor):
    """Pelapor hanya punya kunci PUBLIK Admin. Admin membaca setelah KEK dihitung dari password + dongle."""
    # Saat inisialisasi: kunci privat Admin disimpan terbungkus, publiknya terbuka.
    enc_privkey_wrapped = keys.wrap_private_key(admin["privkey"], admin["kek"], admin["id"])
    enc_pubkey_pem = keys.encryption_public_key_pem(admin["privkey"])

    # Pelapor mengirim laporan.
    paket = textcrypto.encrypt_text(
        "Ada dugaan penyimpangan di gudang B.",
        keys.load_encryption_public_key(enc_pubkey_pem),
        "rpt-9",
        pelapor["id"],
    )

    # Pelapor tidak bisa membuka kunci privat Admin dengan KEK-nya sendiri.
    with pytest.raises(DecryptionError):
        keys.unwrap_private_key(enc_privkey_wrapped, pelapor["kek"], admin["id"])

    # Admin login: KEK dihitung ulang dari password + dongle_secret + salt yang tersimpan.
    kek_login = kdf.derive_kek(admin["password"], admin["salt"], admin["secret"], FAST)
    privkey = keys.unwrap_private_key(enc_privkey_wrapped, kek_login, admin["id"])
    assert textcrypto.decrypt_text(paket, privkey, "rpt-9", pelapor["id"]) == "Ada dugaan penyimpangan di gudang B."
