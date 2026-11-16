"""Exception khusus KLR."""


class KlrError(Exception):
    """Induk semua error KLR."""


class DecryptionError(KlrError):
    """Dekripsi gagal.

    Sengaja hanya satu pesan untuk semua penyebab (kunci salah, data diubah,
    format rusak, AAD tidak cocok). Pesan yang berbeda-beda bisa membocorkan
    kepada penyerang tahap mana yang gagal.
    """
