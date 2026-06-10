from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Santara Mail Reader API",
    description="API untuk membaca surat menggunakan PaddleOCR dan Llama 3",
    version="1.0.0"
)

# Konfigurasi CORS agar API bisa diakses dari frontend (misal React/Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Masukkan router dari routes.py
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Santara Mail Reader API berjalan. Kunjungi /docs untuk melihat dokumentasi."}
