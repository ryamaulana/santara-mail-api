from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List
import logging
import time
import os
import re
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

DOCUMENTS_DIR = settings.DOCUMENTS_DIR
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# user_id datang dari Next.js (sudah diverifikasi sesi login di sana) — kita
# tetap validasi bentuknya di sini karena dipakai langsung sebagai komponen
# path filesystem.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
# Nama file yang kita hasilkan sendiri: <uuid>.<ext> atau <uuid>_page_<n>.<ext>
_FILENAME_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(_page_\d+)?\.(jpg|jpeg|png|webp|pdf)$",
    re.IGNORECASE,
)
_MEDIA_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "webp": "image/webp", "pdf": "application/pdf",
}

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


def _user_dir(user_id: str) -> str:
    if not _UUID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="user_id tidak valid.")
    path = os.path.join(DOCUMENTS_DIR, user_id)
    os.makedirs(path, exist_ok=True)
    return path


async def _save_upload(content: bytes, extension: str, user_id: str) -> tuple[str, str]:
    """Menyimpan berkas ke {DOCUMENTS_DIR}/{user_id}/{uuid}.{ext} — di luar
    folder public Next.js. Mengembalikan (file_path, document_key), di mana
    document_key ("{user_id}/{filename}") itulah yang disimpan ke DB dan
    dipakai untuk membangun URL lewat endpoint /api/documents/* yang dijaga
    autentikasi (bukan URL statis yang bisa diakses siapa saja)."""
    unique_filename = f"{uuid.uuid4()}.{extension}"
    user_dir = _user_dir(user_id)
    file_path = os.path.join(user_dir, unique_filename)
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    return file_path, f"{user_id}/{unique_filename}"


@router.post("/extract-surat")
@limiter.limit("10/minute")
async def extract_surat(request: Request, file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Endpoint untuk mengunggah gambar surat (Single Upload).
    Sistem akan membaca teks dengan OCR lalu merangkumnya menjadi JSON dengan LLM.
    """
    content, extension = await read_and_validate_upload(file)

    if extension == "pdf":
        raise HTTPException(status_code=400, detail="Untuk file PDF silakan gunakan endpoint /batch-extract")

    file_path, document_key = await _save_upload(content, extension, user_id)

    try:
        # Ekstrak Teks menggunakan PaddleOCR
        raw_ocr_result = ocr_engine.extract_text(file_path)
        extracted_text = raw_ocr_result

        if not extracted_text.strip():
            return {
                "status": "success",
                "data": None,
                "message": "Tidak ada teks yang terdeteksi di gambar",
                "file_url": document_key
            }

        # Analisis dengan Llama 3
        parsed_data, usage = await llm_client.parse_document(extracted_text)

        return {
            "status": "success",
            "message": "Dokumen berhasil diproses",
            "raw_text": extracted_text,
            "parsed_data": parsed_data,
            "usage": usage,
            "file_url": document_key
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Gagal memproses extract-surat untuk %s", document_key)
        raise HTTPException(status_code=500, detail="Gagal memproses dokumen.")


@router.post("/batch-extract")
@limiter.limit("5/minute")
async def batch_extract(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
):
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

    user_dir = _user_dir(user_id)
    batch_id = str(uuid.uuid4())

    file_paths = []
    for file in files:
        try:
            content, extension = await read_and_validate_upload(file)
        except HTTPException:
            continue
        file_path, _ = await _save_upload(content, extension, user_id)
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
        user_dir,
        user_id,
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


@router.get("/files/{user_id}/{filename}")
@limiter.limit("120/minute")
async def get_file(request: Request, user_id: str, filename: str):
    """
    Menyajikan satu berkas dokumen. Endpoint ini hanya boleh dipanggil
    server-to-server oleh Next.js (lewat X-Internal-Api-Key, sudah dijaga
    di level router) SETELAH Next.js memverifikasi pemilik dokumen — lihat
    app/api/documents/[userId]/[filename]/route.ts di santara-mail-app.
    Endpoint ini sendiri tidak tahu siapa end-user-nya, jadi jangan pernah
    diekspos langsung ke browser tanpa proxy tersebut.
    """
    if not _UUID_RE.match(user_id) or not _FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Path berkas tidak valid.")

    base = os.path.realpath(DOCUMENTS_DIR)
    file_path = os.path.realpath(os.path.join(DOCUMENTS_DIR, user_id, filename))
    if not file_path.startswith(base + os.sep) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Berkas tidak ditemukan.")

    extension = filename.rsplit(".", 1)[-1].lower()
    return FileResponse(file_path, media_type=_MEDIA_TYPES.get(extension, "application/octet-stream"))
