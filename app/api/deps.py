import hmac
import logging
from fastapi import Header, HTTPException
from app.config import settings

logger = logging.getLogger(__name__)

if not settings.INTERNAL_API_KEY:
    logger.warning(
        "INTERNAL_API_KEY belum di-set — endpoint /api/* berjalan TANPA autentikasi. "
        "Wajib diisi sebelum deploy ke production (lihat deployment_guide.md)."
    )


async def require_internal_api_key(x_internal_api_key: str = Header(default="")):
    """Shared-secret check between the Next.js app and this backend.

    This is not end-user auth (Next.js already handles that) — it exists so
    this API isn't callable directly by anyone who can reach its port,
    bypassing the Next.js app's login/quota checks entirely.
    """
    if not settings.INTERNAL_API_KEY:
        return  # dev mode: no key configured, skip enforcement
    if not hmac.compare_digest(x_internal_api_key, settings.INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
