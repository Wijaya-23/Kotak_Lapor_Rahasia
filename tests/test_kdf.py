import pytest

from klr import kdf
from klr.kdf import Argon2Params

# Parameter murah supaya tes cepat. Satu tes terpisah memakai parameter asli.
FAST = Argon2Params(memory_kib=1024, time_cost=1, parallelism=1)

SALT = bytes(range(16))
DONGLE = bytes(range(32))


def test_parameter_bawaan_sesuai_laporan():
    p = kdf.DEFAULT_PARAMS
    assert p.memory_kib == 64 * 1024  # 64 MiB
    assert p.time_cost == 3
    assert p.parallelism == 4
    assert kdf.SALT_LEN == 16


def test_kek_dengan_parameter_asli_32_byte_dan_deterministik():
    a = kdf.derive_kek("password-uji", SALT, DONGLE)
    b = kdf.derive_kek("password-uji", SALT, DONGLE)
    assert len(a) == 32
    assert a == b


def test_kek_berubah_jika_password_berubah():
    a = kdf.derive_kek("password-a", SALT, DONGLE, FAST)
    b = kdf.derive_kek("password-b", SALT, DONGLE, FAST)
    assert a != b


def test_kek_berubah_jika_dongle_secret_berubah():
    """Password benar tapi dongle salah: KEK berbeda, jadi data tidak terbuka."""
    a = kdf.derive_kek("pw", SALT, DONGLE, FAST)
    b = kdf.derive_kek("pw", SALT, bytes(32), FAST)
    assert a != b


def test_kek_berubah_jika_salt_berubah():
    a = kdf.derive_kek("pw", SALT, DONGLE, FAST)
    b = kdf.derive_kek("pw", bytes(16), DONGLE, FAST)
    assert a != b


def test_password_dinormalisasi_unicode():
    """'é' satu karakter dan 'e' + aksen terpisah harus menghasilkan KEK yang sama."""
    satu = "café"
    terpisah = "café"
    assert satu != terpisah
    assert kdf.derive_kek(satu, SALT, DONGLE, FAST) == kdf.derive_kek(terpisah, SALT, DONGLE, FAST)


@pytest.mark.parametrize("salt", [b"", bytes(15), bytes(17)])
def test_salt_salah_panjang_ditolak(salt):
    with pytest.raises(ValueError):
        kdf.derive_kek("pw", salt, DONGLE, FAST)


@pytest.mark.parametrize("secret", [b"", bytes(31), bytes(33)])
def test_dongle_secret_salah_panjang_ditolak(secret):
    with pytest.raises(ValueError):
        kdf.derive_kek("pw", SALT, secret, FAST)


def test_password_kosong_ditolak():
    with pytest.raises(ValueError):
        kdf.derive_kek("", SALT, DONGLE, FAST)
    with pytest.raises(ValueError):
        kdf.hash_password("", FAST)


def test_subkey_berbeda_per_label_dan_berbeda_dari_kek():
    kek = kdf.derive_kek("pw", SALT, DONGLE, FAST)
    labels = [kdf.INFO_PRIVKEY_ENC, kdf.INFO_DEK_SLOT, kdf.INFO_FACE_TEMPLATE]
    subs = [kdf.derive_subkey(kek, lb) for lb in labels]
    assert all(len(s) == 32 for s in subs)
    assert len({*subs, kek}) == 4  # semuanya berbeda


def test_subkey_deterministik():
    kek = kdf.derive_kek("pw", SALT, DONGLE, FAST)
    assert kdf.derive_subkey(kek, kdf.INFO_DEK_SLOT) == kdf.derive_subkey(kek, kdf.INFO_DEK_SLOT)


def test_subkey_validasi_input():
    with pytest.raises(ValueError):
        kdf.derive_subkey(bytes(31), kdf.INFO_DEK_SLOT)
    with pytest.raises(ValueError):
        kdf.derive_subkey(bytes(32), b"")


def test_hash_password_unik_dan_terverifikasi():
    h1 = kdf.hash_password("rahasia", FAST)
    h2 = kdf.hash_password("rahasia", FAST)
    assert h1.startswith("$argon2id$")
    assert h1 != h2  # salt acak berbeda tiap hash
    assert kdf.verify_password(h1, "rahasia")
    assert kdf.verify_password(h2, "rahasia")


def test_verify_password_salah_atau_hash_rusak_bernilai_false():
    h = kdf.hash_password("rahasia", FAST)
    assert kdf.verify_password(h, "salah") is False
    assert kdf.verify_password("bukan-hash", "rahasia") is False
    assert kdf.verify_password(h, "") is False


def test_new_salt_acak_16_byte():
    a, b = kdf.new_salt(), kdf.new_salt()
    assert len(a) == len(b) == 16
    assert a != b
