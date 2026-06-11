import json
import logging
from typing import Dict, Any
from groq import AsyncGroq
from app.config import settings

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
        Mengembalikan System Prompt dengan teknik Few-Shot Examples agar Llama 3
        menghasilkan draf balasan sekolah yang memiliki struktur formal dan baku.
        """
        return """Anda adalah asisten AI ahli dalam memproses dokumen resmi dan surat menyurat instansi sekolah di Indonesia.
Tugas Anda adalah menganalisis teks hasil OCR dari sebuah surat sekolah dan mengekstraknya menjadi format JSON yang valid.

Analisis teks OCR tersebut dengan teliti, lalu isi struktur JSON di bawah ini sesuai aturan berikut:

{
  "nomor_surat": "Tulis nomor surat resmi yang tertera secara lengkap. Jika benar-benar tidak ada nomornya, isi null.",
  "tanggal_surat": "Tulis tanggal surat diterbitkan. Jika tidak ada, isi null.",
  "perihal": "Ekstrak perihal, hal, atau subjek surat dengan jelas. Jika tidak ada, isi null.",
  "pengirim": "Nama instansi sekolah, komite, organisasi, atau individu yang mengirim surat. Jika tidak ada, isi null.",
  "ditujukan_kepada": "Pihak yang menjadi tujuan surat (misal: Kepala Sekolah, Orang Tua Siswa, Guru). Jika tidak ada, isi null.",
  "sifat_surat": "Tentukan sifat surat dengan memilih salah satu persis antara: 'Biasa', 'Penting', atau 'Rahasia' berdasarkan urgensi dan konteks surat.",
  "ringkasan": "Buat ringkasan padat mengenai inti dan tujuan utama surat dalam 2-3 kalimat berdasarkan fakta di teks OCR. Jangan menambah informasi di luar teks.",
  "draf_balasan": "Buat draf balasan resmi bahasa Indonesia yang formal, sopan, dan terstruktur baku khas instansi pendidikan. Sesuaikan logika balasan berdasarkan contoh di bawah ini."
}

[CONTOH STRUKTUR FORMAL DRAF BALASAN SEKOLAH]

Contoh 1: Jika Surat Berupa Undangan Rapat / Acara
Draf_balasan: "Dengan hormat, kami mengucapkan terima kasih atas surat undangan [Perihal Surat] dengan nomor [Nomor Surat] yang telah kami terima. Melalui surat ini, kami bermaksud menyampaikan konfirmasi bahwa pihak [Ditujukan Kepada] bersedia dan siap menghadiri agenda tersebut sesuai dengan waktu dan tempat yang telah ditentukan. Demikian konfirmasi ini kami sampaikan, atas perhatian dan kerja samanya kami ucapkan terima kasih."

Contoh 2: Jika Surat Berupa Permintaan Dokumen / Data Siswa
Draf_balasan: "Selamat pagi/siang, terima kasih telah menghubungi kami. Menindaklanjuti surat permintaan dokumen terkait [Perihal Surat], bersama surat ini kami lampirkan berkas data yang Anda perlukan. Mohon informasi lebih lanjut apabila terdapat kendala atau berkas tambahan lain yang perlu kami lengkapi. Atas perhatian Anda, kami ucapkan terima kasih."

Contoh 3: Jika Surat Memerlukan Waktu untuk Koordinasi Internal
Draf_balasan: "Dengan hormat, terima kasih atas surat yang Anda kirimkan mengenai [Perihal Surat]. Kami informasikan bahwa saat ini permohonan tersebut sedang dalam tahap peninjauan dan koordinasi dengan pihak manajemen sekolah. Kami akan segera memberikan keputusan atau kabar terbaru kepada Anda paling lambat dalam waktu dekat. Terima kasih atas pengertian dan kesabaran Anda."

ATURAN KETAT OUTPUT:
1. Output Anda MURNI harus berupa objek JSON yang valid.
2. JANGAN sertakan teks basa-basi sebelum atau sesudah JSON (seperti "Berikut adalah hasilnya...").
3. JANGAN membungkus JSON dengan backticks Markdown. Langsung mulai dengan kurung kurawal '{' dan akhiri dengan '}'.
4. Ubah placeholder seperti [Perihal Surat], [Nomor Surat], atau [Ditujukan Kepada] pada contoh di atas menggunakan data asli yang Anda temukan di teks OCR!
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
            logger.info("Berhasil memproses respons JSON dari Groq API.")
            return parsed_json
                
        except json.JSONDecodeError as e:
            logger.error(f"Gagal menerjemahkan respons Groq ke JSON. Respons mentah: {response_text}")
            raise Exception("Model tidak mengembalikan format JSON yang valid.")
        except Exception as e:
            logger.error(f"Terjadi kesalahan tak terduga: {e}")
            raise
