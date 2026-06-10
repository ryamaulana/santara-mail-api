import httpx
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        """
        Inisialisasi klien LLM untuk berinteraksi dengan Ollama secara lokal.
        Args:
            base_url (str): Base URL untuk API Ollama. Default ke localhost.
            model (str): Nama model di Ollama yang akan digunakan (misal: 'llama3' atau 'llama3.1').
        """
        self.base_url = base_url
        self.model = model
        self.generate_endpoint = f"{self.base_url}/api/generate"
        logger.info(f"Menginisialisasi LLMClient dengan model '{self.model}' di '{self.base_url}'")

    def get_system_prompt(self) -> str:
        """
        Mengembalikan System Prompt khusus dalam bahasa Indonesia untuk memandu Llama 3 
        mengekstrak data dan merespons hanya dengan format JSON.
        """
        return """Anda adalah asisten AI ahli dalam memproses dokumen resmi dan surat menyurat instansi sekolah di Indonesia.
Tugas Anda adalah menganalisis teks berantakan hasil OCR dari sebuah surat dan merangkumnya menjadi format JSON yang valid.

Ekstrak dan hasilkan informasi berikut HANYA dalam bentuk JSON dengan key (kunci) persis seperti di bawah ini:
{
  "nomor_surat": "Nomor surat yang tertera. Jika tidak ada, isi null.",
  "tanggal_surat": "Tanggal surat diterbitkan. Jika tidak ada, isi null.",
  "perihal": "Perihal, hal, atau subjek surat. Jika tidak ada, isi null.",
  "pengirim": "Nama instansi, organisasi, atau individu yang mengirim surat. Jika tidak ada, isi null.",
  "ditujukan_kepada": "Pihak yang menjadi tujuan surat (misal: Kepala Sekolah, Orang Tua Siswa). Jika tidak ada, isi null.",
  "ringkasan": "Ringkasan padat mengenai isi dan tujuan utama surat dalam 2-3 kalimat.",
  "draf_balasan": "Draf otomatis untuk membalas surat ini. Gunakan bahasa Indonesia yang formal, sopan, dan profesional khas institusi pendidikan."
}

PENTING:
- Pastikan output Anda murni HANYA JSON.
- Jangan menambahkan teks apa pun sebelum atau sesudah JSON (seperti "Berikut adalah hasilnya" atau blok kode Markdown tambahan jika API tidak mendukung).
- Jika ada atribut yang tidak dapat ditemukan dalam teks OCR, berikan nilai null.
"""

    async def parse_document(self, ocr_text: str) -> Dict[str, Any]:
        """
        Mengirim teks OCR ke Ollama untuk dianalisis dan diformat menjadi JSON.
        Fungsi ini bersifat asynchronous (async) agar tidak memblokir FastAPI.
        
        Args:
            ocr_text (str): Teks mentah hasil pembacaan OCR.
            
        Returns:
            Dict[str, Any]: Data struktur JSON yang berisi informasi surat.
        """
        system_prompt = self.get_system_prompt()
        
        # Menggabungkan instruksi sistem dengan input teks OCR dari pengguna
        prompt = f"{system_prompt}\n\n--- TEKS OCR SURAT ---\n{ocr_text}\n----------------------"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            # Fitur krusial dari Ollama untuk memaksa output selalu valid secara sintaksis JSON
            "format": "json", 
            "stream": False,
            # Menurunkan temperature membuat jawaban model lebih deterministik dan fokus
            "options": {
                "temperature": 0.1 
            }
        }
        
        logger.info(f"Mengirim teks (panjang: {len(ocr_text)} karakter) ke Ollama ({self.model})...")
        
        try:
            # Menggunakan httpx.AsyncClient karena ini akan berjalan di dalam FastAPI nantinya
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.generate_endpoint, json=payload)
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get("response", "")
                
                # Mengubah string JSON dari Ollama menjadi objek Dictionary Python
                parsed_json = json.loads(response_text)
                logger.info("Berhasil memproses respons JSON dari Ollama.")
                return parsed_json
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error dari Ollama: {e.response.text}")
            raise Exception(f"Gagal menghubungi Ollama API: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Gagal menerjemahkan respons Ollama ke JSON. Respons mentah: {response_text}")
            raise Exception("Model tidak mengembalikan format JSON yang valid.")
        except Exception as e:
            logger.error(f"Terjadi kesalahan tak terduga: {e}")
            raise
