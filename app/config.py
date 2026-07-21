from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Shared secret the Next.js app must send in the X-Internal-Api-Key header.
    # Required in production — see docker-compose.yml / deployment_guide.md.
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

    # Comma-separated list of origins allowed to call this API via CORS
    # (browser-based calls only; server-to-server calls from Next.js aren't
    # subject to CORS at all). Defaults to no cross-origin access.
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "")

    # Root direktori tempat dokumen surat disimpan, di-mount dari disk host
    # (lihat docker-compose.yml) — TIDAK di dalam folder public Next.js, karena
    # dokumen ini sensitif dan hanya boleh diserve lewat endpoint /api/files/*
    # yang dijaga require_internal_api_key + otorisasi kepemilikan di Next.js.
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./storage/documents")

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "30"))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "50"))
    MAX_BATCH_FILES: int = int(os.getenv("MAX_BATCH_FILES", "30"))
    BATCH_STATUS_TTL_SECONDS: int = int(os.getenv("BATCH_STATUS_TTL_SECONDS", str(6 * 3600)))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

settings = Settings()
