import json
import logging
from typing import Dict, Any
from groq import AsyncGroq
from app.config import settings
from app.inference.reply_templates import build_draf_balasan

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Inisialisasi klien LLM untuk berinteraksi dengan Groq API.
        Args:
            model (str): Nama model di Groq yang akan digunakan.
        """
        self.model = model
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        logger.info(f"Menginisialisasi LLMClient dengan model '{self.model}' via Groq API")

    def get_system_prompt(self) -> str:
        """
        Mengembalikan System Prompt agar Llama 3 mengekstrak data surat sekolah
        menjadi JSON. Draf balasan TIDAK ditulis penuh oleh model di sini — model
        hanya memilih jenis balasan + inti kalimatnya; paragraf lengkap dirakit
        oleh kode dari template tetap (lihat reply_templates.py). Ini membuat
        prompt jauh lebih pendek dan output model jauh lebih singkat.
        """
        return """Anda adalah asisten AI ahli dalam memproses dokumen resmi dan surat menyurat instansi sekolah di Indonesia.
Tugas Anda adalah menganalisis teks hasil OCR dari sebuah surat sekolah dan mengekstraknya menjadi format JSON yang valid.

Isi struktur JSON di bawah ini sesuai aturan berikut:

{
  "nomor_surat": "Nomor surat resmi lengkap. Jika tidak ada, null.",
  "tanggal_surat": "Tanggal surat diterbitkan. Jika tidak ada, null.",
  "perihal": "Perihal/hal/subjek surat. Jika tidak ada, null.",
  "pengirim": "Nama instansi/komite/individu pengirim surat. Jika tidak ada, null.",
  "ditujukan_kepada": "Pihak tujuan surat (mis. Kepala Sekolah, Orang Tua Siswa). Jika tidak ada, null.",
  "sifat_surat": "Salah satu persis: 'Biasa', 'Penting', atau 'Rahasia'.",
  "ringkasan": "Ringkasan padat inti & tujuan surat dalam 2-3 kalimat, hanya berdasarkan fakta di teks OCR.",
  "jenis_balasan": "Salah satu persis: 'konfirmasi_hadir' (undangan rapat/acara), 'pengiriman_dokumen' (permintaan dokumen/data), 'permohonan_waktu' (perlu koordinasi/peninjauan internal), atau 'umum' (tidak cocok kategori lain).",
  "poin_balasan": "1-2 kalimat SINGKAT berisi inti balasan berdasarkan fakta di surat (contoh: 'Kami menyatakan bersedia hadir pada acara tersebut.'). JANGAN tulis kalimat pembuka/penutup formal (Dengan hormat, terima kasih, dst) — itu ditambahkan otomatis oleh sistem."
}

ATURAN KETAT OUTPUT:
1. Output Anda MURNI harus berupa objek JSON yang valid.
2. JANGAN sertakan teks basa-basi sebelum atau sesudah JSON (seperti "Berikut adalah hasilnya...").
3. JANGAN membungkus JSON dengan backticks Markdown. Langsung mulai dengan kurung kurawal '{' dan akhiri dengan '}'.
"""

    async def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Mengirim teks OCR ke Groq API untuk dianalisis dan diformat menjadi JSON.
        Fungsi ini bersifat asynchronous (async) agar tidak memblokir FastAPI.
        
        Args:
            ocr_text (str): Teks mentah hasil pembacaan OCR.
            
        Returns:
            Dict[str, Any]: Data struktur JSON yang berisi informasi surat.
        """
        system_prompt = self.get_system_prompt()
        
        # Menggabungkan instruksi sistem dengan input teks OCR dari pengguna
        user_prompt = f"--- TEKS OCR SURAT ---\n{ocr_text}\n----------------------"
        
        logger.info(f"Mengirim teks (panjang: {len(ocr_text)} karakter) ke Groq API ({self.model})...")
        
        try:
            # Memanggil API Groq secara asinkron
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model,
                temperature=0.1,
                # Fitur krusial untuk memaksa model mengeluarkan JSON yang valid
                response_format={"type": "json_object"},
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Mengubah string JSON dari Groq menjadi objek Dictionary Python
            parsed_json = json.loads(response_text)

            # Rakit paragraf draf_balasan lengkap dari jenis_balasan + poin_balasan
            # (model tidak lagi menulis paragraf penuh — lihat get_system_prompt()).
            parsed_json["draf_balasan"] = build_draf_balasan(parsed_json)

            logger.info("Berhasil memproses respons JSON dari Groq API.")
            return parsed_json
                
        except json.JSONDecodeError as e:
            logger.error(f"Gagal menerjemahkan respons Groq ke JSON. Respons mentah: {response_text}")
            raise Exception("Model tidak mengembalikan format JSON yang valid.")
        except Exception as e:
            logger.error(f"Terjadi kesalahan tak terduga: {e}")
            raise
