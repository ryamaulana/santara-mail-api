import os
# Disable oneDNN/MKLDNN to prevent PaddlePaddle 3.3.x crash on CPU
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, lang='id'):
        """
        Inisialisasi engine PaddleOCR.
        Args:
            lang (str): Kode bahasa. Default adalah 'id' (Indonesia).
                        ('en' juga seringkali memberikan hasil yang baik untuk dokumen campuran).
        """
        logger.info(f"Menginisialisasi PaddleOCR dengan bahasa: {lang}")
        # use_angle_cls=True sangat membantu jika hasil scan/foto agak miring
        # show_log=False untuk mengurangi log berlebih dari PaddleOCR di terminal
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        logger.info("Inisialisasi PaddleOCR selesai.")

    def extract_text(self, image_path: str) -> str:
        """
        Mengekstrak teks dari gambar menggunakan PaddleOCR.
        Args:
            image_path (str): Path absolut atau relatif ke file gambar surat.
        Returns:
            str: Teks mentah hasil ekstraksi dari gambar.
        """
        logger.info(f"Mengekstrak teks dari gambar: {image_path}")
        try:
            # Jalankan OCR pada gambar
            result = self.ocr.ocr(image_path)
            
            extracted_text = ""
            # result adalah list, biasanya result[0] berisi list deteksi baris teks
            if result and result[0]:
                for line in result[0]:
                    # format line: [[koordinat kotak], (string_teks, skor_kepercayaan)]
                    text = line[1][0]
                    extracted_text += text + "\n"
                    
            logger.info(f"Berhasil mengekstrak teks dari gambar. Hasil OCR (100 char pertama): {extracted_text[:100]!r}")
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error saat proses ekstraksi OCR: {e}")
            raise
