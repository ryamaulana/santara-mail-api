from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import shutil
import os
import uuid
from app.inference.ocr_engine import OCREngine
from app.inference.llm_client import LLMClient
from app.inference.batch_processor import process_batch_task

router = APIRouter(prefix="/api")

# Inisialisasi engine OCR dan klien LLM saat aplikasi berjalan
ocr_engine = OCREngine()
llm_client = LLMClient()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global dictionary untuk menyimpan status antrean batch
BATCH_STATUS = {}

@router.post("/extract-surat")
async def extract_surat(file: UploadFile = File(...)):
    """
    Endpoint untuk mengunggah gambar surat (Single Upload).
    Sistem akan membaca teks dengan OCR lalu merangkumnya menjadi JSON dengan LLM.
    """
    if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG) atau PDF")

    # 1. Simpan file gambar sementara
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Jika file adalah PDF, kita tolak di single upload atau diproses?
        # Untuk kesederhanaan, arahkan user ke batch upload jika itu PDF
        if file_extension.lower() == "pdf":
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Untuk file PDF silakan gunakan endpoint /batch-extract")

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


@router.post("/batch-extract")
async def batch_extract(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Endpoint untuk mengunggah banyak surat sekaligus (Bulk Upload) atau multi-page PDF.
    Mengembalikan batch_id dan langsung memproses di latar belakang (BackgroundTasks).
    """
    batch_id = str(uuid.uuid4())
    
    file_paths = []
    # Simpan semua file ke folder upload
    for file in files:
        if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
            continue
            
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_paths.append(file_path)

    if not file_paths:
        raise HTTPException(status_code=400, detail="Tidak ada file valid yang diunggah.")

    # Inisialisasi status batch
    BATCH_STATUS[batch_id] = {
        "status": "processing",
        "total": len(file_paths),
        "completed": 0,
        "results": [],
        "error": None
    }

    # Daftarkan proses ke BackgroundTasks
    background_tasks.add_task(
        process_batch_task, 
        batch_id, 
        file_paths, 
        BATCH_STATUS, 
        UPLOAD_DIR,
        ocr_engine,
        llm_client
    )

    return {"status": "202 Accepted", "batch_id": batch_id, "message": "Dokumen sedang diproses di latar belakang."}


@router.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Endpoint untuk mengecek status dan hasil dari antrean batch.
    """
    if batch_id not in BATCH_STATUS:
        raise HTTPException(status_code=404, detail="Batch ID tidak ditemukan")
        
    return BATCH_STATUS[batch_id]
