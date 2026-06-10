from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
from app.inference.ocr_engine import OCREngine
from app.inference.llm_client import LLMClient

router = APIRouter(prefix="/api")

# Inisialisasi engine OCR dan klien LLM saat aplikasi berjalan
ocr_engine = OCREngine()
llm_client = LLMClient()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/extract-surat")
async def extract_surat(file: UploadFile = File(...)):
    """
    Endpoint untuk mengunggah gambar surat.
    Sistem akan membaca teks dengan OCR lalu merangkumnya menjadi JSON dengan LLM.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG)")

    # 1. Simpan file gambar sementara
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Ekstrak Teks menggunakan PaddleOCR
        raw_ocr_result = ocr_engine.extract_text(file_path)
        
        extracted_text = raw_ocr_result

        if not extracted_text.strip():
            return {"status": "success", "data": None, "message": "Tidak ada teks yang terdeteksi di gambar"}

        # 3. Analisis dengan Llama 3
        parsed_data = await llm_client.parse_document(extracted_text)

        return {
            "status": "success",
            "message": "Dokumen berhasil diproses",
            "raw_text": extracted_text,
            "parsed_data": parsed_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Bersihkan file setelah selesai diproses agar storage tidak penuh
        if os.path.exists(file_path):
            os.remove(file_path)
