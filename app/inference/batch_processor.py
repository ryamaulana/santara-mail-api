import logging
import os
from app.inference.ocr_engine import OCREngine
from app.inference.llm_client import LLMClient
from app.inference.pdf_splitter import split_pdf_to_images

logger = logging.getLogger(__name__)


async def _process_pdf(file_path, batch_id, batch_status_dict, user_dir, user_id, ocr_engine, llm_client, max_pdf_pages):
    """Satu PDF (berapapun jumlah halamannya) diperlakukan sebagai SATU dokumen:
    semua halaman di-OCR lalu teksnya digabung sebelum dikirim ke LLM sekali saja,
    supaya hasilnya jadi satu entri, bukan pecah per halaman."""
    try:
        pdf_images = split_pdf_to_images(file_path, user_dir, max_pdf_pages)
    except Exception as e:
        batch_status_dict[batch_id]["results"].append({
            "file": os.path.basename(file_path),
            "status": "error",
            "message": f"Gagal membaca PDF: {str(e)}"
        })
        batch_status_dict[batch_id]["completed"] += 1
        return

    page_texts = []
    for image_path in pdf_images:
        try:
            text = ocr_engine.extract_text(image_path)
            if text.strip():
                page_texts.append(text.strip())
        except Exception:
            logger.exception("Gagal OCR salah satu halaman PDF %s", file_path)

    # Semua halaman disimpan agar bisa dilihat satu per satu di frontend,
    # bukan cuma halaman pertama.
    page_urls = [f"{user_id}/{os.path.basename(p)}" for p in pdf_images]
    first_page_url = page_urls[0] if page_urls else None
    # PDF asli juga disimpan (tidak dihapus) supaya bisa diunduh utuh lagi nanti
    # dari halaman Surat Masuk — bukan cuma pratinjau halaman pertama.
    original_file_url = f"{user_id}/{os.path.basename(file_path)}"
    combined_text = "\n\n".join(page_texts)

    try:
        if not combined_text.strip():
            batch_status_dict[batch_id]["results"].append({
                "file": os.path.basename(file_path),
                "file_url": first_page_url,
                "page_urls": page_urls,
                "original_file_url": original_file_url,
                "status": "success",
                "data": None,
                "message": "Tidak ada teks yang terdeteksi di dokumen"
            })
        else:
            parsed_data, usage = await llm_client.parse_document(combined_text)
            batch_status_dict[batch_id]["results"].append({
                "file": os.path.basename(file_path),
                "file_url": first_page_url,
                "page_urls": page_urls,
                "original_file_url": original_file_url,
                "status": "success",
                "raw_text": combined_text,
                "parsed_data": parsed_data,
                "usage": usage
            })
    except Exception as e:
        batch_status_dict[batch_id]["results"].append({
            "file": os.path.basename(file_path),
            "status": "error",
            "message": str(e)
        })
    finally:
        batch_status_dict[batch_id]["completed"] += 1


async def _process_image(file_path, batch_id, batch_status_dict, user_id, ocr_engine, llm_client):
    try:
        extracted_text = ocr_engine.extract_text(file_path)

        if not extracted_text.strip():
            batch_status_dict[batch_id]["results"].append({
                "file": os.path.basename(file_path),
                "file_url": f"{user_id}/{os.path.basename(file_path)}",
                "status": "success",
                "data": None,
                "message": "Tidak ada teks yang terdeteksi di gambar"
            })
        else:
            parsed_data, usage = await llm_client.parse_document(extracted_text)
            batch_status_dict[batch_id]["results"].append({
                "file": os.path.basename(file_path),
                "file_url": f"{user_id}/{os.path.basename(file_path)}",
                "status": "success",
                "raw_text": extracted_text,
                "parsed_data": parsed_data,
                "usage": usage
            })
    except Exception as e:
        batch_status_dict[batch_id]["results"].append({
            "file": os.path.basename(file_path),
            "status": "error",
            "message": str(e)
        })
    finally:
        batch_status_dict[batch_id]["completed"] += 1


async def process_batch_task(
    batch_id: str,
    file_paths: list[str],
    batch_status_dict: dict,
    user_dir: str,
    user_id: str,
    ocr_engine: OCREngine,
    llm_client: LLMClient,
    max_pdf_pages: int | None = None
):
    """
    Fungsi worker asinkron untuk memproses antrean file di latar belakang.
    Menerima list of file paths (gambar atau PDF), mengekstrak teks, dan memanggil LLM.
    Setiap file asli (termasuk PDF multi-halaman) menghasilkan tepat satu entri hasil.
    """
    try:
        for file_path in file_paths:
            is_pdf = file_path.lower().endswith(".pdf")

            if is_pdf:
                # PDF asli sengaja tidak dihapus (lihat _process_pdf) agar bisa
                # diunduh utuh lagi nanti dari halaman Surat Masuk.
                await _process_pdf(file_path, batch_id, batch_status_dict, user_dir, user_id, ocr_engine, llm_client, max_pdf_pages)
            else:
                await _process_image(file_path, batch_id, batch_status_dict, user_id, ocr_engine, llm_client)

        # Tandai selesai
        batch_status_dict[batch_id]["status"] = "completed"

    except Exception as e:
        batch_status_dict[batch_id]["status"] = "failed"
        batch_status_dict[batch_id]["error"] = str(e)
