from fastapi import UploadFile, HTTPException
from app.config import settings

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "pdf"}

# Magic-byte signatures for the file types this API accepts. Content-Type
# headers are client-supplied and trivially spoofable, so we sniff the
# actual bytes instead of trusting them.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "webp": (b"RIFF",),  # followed by "WEBP" at offset 8, checked separately
    "pdf": (b"%PDF-",),
}


def _extension_of(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def validate_file_header(filename: str, header: bytes) -> str:
    """Validates filename extension + magic bytes. Returns the lowercased
    extension on success, raises HTTPException(400) otherwise."""
    extension = _extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Jenis berkas tidak didukung. Gunakan JPEG, PNG, WEBP, atau PDF.",
        )

    if extension == "webp":
        ok = header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    else:
        ok = any(header.startswith(sig) for sig in _SIGNATURES[extension])

    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Isi berkas tidak sesuai dengan ekstensinya.",
        )

    return extension


async def read_and_validate_upload(file: UploadFile) -> tuple[bytes, str]:
    """Reads an UploadFile fully into memory while enforcing the max upload
    size and validating its magic bytes. Returns (content, extension)."""
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Berkas melebihi batas maksimum {settings.MAX_UPLOAD_MB} MB.",
            )
        chunks.append(chunk)

    content = b"".join(chunks)
    extension = validate_file_header(file.filename or "", content[:16])
    return content, extension
