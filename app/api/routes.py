from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List
import logging
import time
import os
import uuid
from app.inference.ocr_engine import OCREngine
from app.inference.llm_client import LLMClient
from app.inference.batch_processor import process_batch_task
from app.api.deps import require_internal_api_key
from app.utils import read_and_validate_upload
from app.config import settings

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api", dependencies=[Depends(require_internal_api_key)])

# Inisialisasi engine OCR dan klien LLM saat aplikasi berjalan
ocr_engine = OCREngine()
llm_client = LLMClient(model=settings.GROQ_MODEL)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "../santara-mail-app/public/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global dictionary untuk menyimpan status antrean batch (proses tunggal,
# tidak persisten — lihat catatan skalabilitas di deployment_guide.md).
BATCH_STATUS: dict[str, dict] = {}


def _sweep_expired_batches():
    """Buang entri batch yang sudah lebih tua dari TTL agar dict ini tidak
    tumbuh tanpa batas selama proses berjalan lama."""
    cutoff = time.time() - settings.BATCH_STATUS_TTL_SECONDS
    expired = [bid for bid, entry in BATCH_STATUS.items() if entry.get("created_at", cutoff) < cutoff]
    for bid in expired:
        del BATCH_STATUS[bid]


async def _save_upload(content: bytes, extension: str) -> tuple[str, str]:
    unique_filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    return file_path, unique_filename


@router.post("/extract-surat")
@limiter.limit("10/minute")
async def extract_surat(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mengunggah gambar surat (Single Upload).
    Sistem akan membaca teks dengan OCR lalu merangkumnya menjadi JSON dengan LLM.
    """
    content, extension = await read_and_validate_upload(file)

    if extension == "pdf":
        raise HTTPException(status_code=400, detail="Untuk file PDF silakan gunakan endpoint /batch-extract")

    file_path, unique_filename = await _save_upload(content, extension)

    try:
        # Ekstrak Teks menggunakan PaddleOCR
        raw_ocr_result = ocr_engine.extract_text(file_path)
        extracted_text = raw_ocr_result

        if not extracted_text.strip():
            return {
                "status": "success",
                "data": None,
                "message": "Tidak ada teks yang terdeteksi di gambar",
                "file_url": f"/uploads/{unique_filename}"
            }

        # Analisis dengan Llama 3
        parsed_data, usage = await llm_client.parse_document(extracted_text)

        return {
            "status": "success",
            "message": "Dokumen berhasil diproses",
            "raw_text": extracted_text,
            "parsed_data": parsed_data,
            "usage": usage,
            "file_url": f"/uploads/{unique_filename}"
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Gagal memproses extract-surat untuk %s", unique_filename)
        raise HTTPException(status_code=500, detail="Gagal memproses dokumen.")


@router.post("/batch-extract")
@limiter.limit("5/minute")
async def batch_extract(request: Request, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Endpoint untuk mengunggah banyak surat sekaligus (Bulk Upload) atau multi-page PDF.
    Mengembalikan batch_id dan langsung memproses di latar belakang (BackgroundTasks).
    """
    _sweep_expired_batches()

    if len(files) > settings.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maksimum {settings.MAX_BATCH_FILES} berkas per batch.",
        )

    batch_id = str(uuid.uuid4())

    file_paths = []
    for file in files:
        try:
            content, extension = await read_and_validate_upload(file)
        except HTTPException:
            continue
        file_path, _ = await _save_upload(content, extension)
        file_paths.append(file_path)

    if not file_paths:
        raise HTTPException(status_code=400, detail="Tidak ada file valid yang diunggah.")

    BATCH_STATUS[batch_id] = {
        "status": "processing",
        "total": len(file_paths),
        "completed": 0,
        "results": [],
        "error": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        process_batch_task,
        batch_id,
        file_paths,
        BATCH_STATUS,
        UPLOAD_DIR,
        ocr_engine,
        llm_client,
        settings.MAX_PDF_PAGES,
    )

    return {"status": "202 Accepted", "batch_id": batch_id, "message": "Dokumen sedang diproses di latar belakang."}


@router.get("/batch-status/{batch_id}")
@limiter.limit("60/minute")
async def get_batch_status(request: Request, batch_id: str):
    """
    Endpoint untuk mengecek status dan hasil dari antrean batch.
    """
    if batch_id not in BATCH_STATUS:
        raise HTTPException(status_code=404, detail="Batch ID tidak ditemukan")

    return BATCH_STATUS[batch_id]
