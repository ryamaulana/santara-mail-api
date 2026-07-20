import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api.routes import router
from app.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Santara Mail Reader API",
    description="API untuk membaca surat menggunakan PaddleOCR dan Llama 3",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS hanya relevan untuk pemanggilan langsung dari browser (mis. dokumentasi
# /docs). Pemanggilan server-to-server dari Next.js tidak tunduk pada CORS
# sama sekali. Default: tidak ada origin yang diizinkan kecuali dikonfigurasi.
_origins = settings.allowed_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Internal-Api-Key"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error saat memproses %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Terjadi kesalahan internal pada server."})


@app.get("/")
def root():
    return {"message": "Santara Mail Reader API berjalan. Kunjungi /docs untuk melihat dokumentasi."}


@app.get("/health")
def health():
    return {"status": "ok"}
