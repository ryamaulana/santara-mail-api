"""Template balasan surat resmi.

Paragraf lengkap dirakit di sini dari field pendek yang dihasilkan LLM
(`jenis_balasan` + `poin_balasan`), bukan ditulis penuh oleh model — ini
memangkas token input (system prompt jadi jauh lebih pendek) sekaligus
token output (model cukup mengeluarkan 1-2 kalimat, bukan satu paragraf).
"""

REPLY_TEMPLATES = {
    "konfirmasi_hadir": (
        "Dengan hormat, kami mengucapkan terima kasih atas surat {perihal} "
        "dengan nomor {nomor_surat} yang telah kami terima dari {pengirim}. "
        "{poin_balasan} Demikian konfirmasi ini kami sampaikan, atas perhatian "
        "dan kerja samanya kami ucapkan terima kasih."
    ),
    "pengiriman_dokumen": (
        "Dengan hormat, terima kasih telah menghubungi kami terkait {perihal}. "
        "{poin_balasan} Mohon informasi lebih lanjut apabila terdapat kendala "
        "atau berkas tambahan yang perlu kami lengkapi. Atas perhatian Anda, "
        "kami ucapkan terima kasih."
    ),
    "permohonan_waktu": (
        "Dengan hormat, terima kasih atas surat yang Anda kirimkan mengenai "
        "{perihal}. {poin_balasan} Kami akan segera memberikan kabar terbaru "
        "kepada Anda dalam waktu dekat. Terima kasih atas pengertian dan "
        "kesabaran Anda."
    ),
    "umum": (
        "Dengan hormat, sehubungan dengan surat {perihal} dari {pengirim}, "
        "{poin_balasan} Demikian balasan ini kami sampaikan, atas perhatian "
        "dan kerja samanya kami ucapkan terima kasih."
    ),
}


def build_draf_balasan(parsed: dict) -> str:
    """Rakit paragraf draf_balasan lengkap dari field jenis_balasan + poin_balasan."""
    jenis_balasan = parsed.get("jenis_balasan") or "umum"
    template = REPLY_TEMPLATES.get(jenis_balasan, REPLY_TEMPLATES["umum"])

    poin_balasan = (parsed.get("poin_balasan") or "").strip()
    perihal = parsed.get("perihal") or "yang Anda maksud"
    nomor_surat = parsed.get("nomor_surat") or "-"
    pengirim = parsed.get("pengirim") or "Bapak/Ibu"

    return template.format(
        perihal=perihal,
        nomor_surat=nomor_surat,
        pengirim=pengirim,
        poin_balasan=poin_balasan,
    ).strip()
