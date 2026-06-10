import os
import asyncio
from app.inference.ocr_engine import OCREngine
from app.inference.llm_client import LLMClient
from app.inference.pdf_splitter import split_pdf_to_images

async def process_batch_task(
    batch_id: str, 
    file_paths: list[str], 
    batch_status_dict: dict, 
    upload_dir: str,
    ocr_engine: OCREngine,
    llm_client: LLMClient
):
    """
    Fungsi worker asinkron untuk memproses antrean file di latar belakang.
    Menerima list of file paths (gambar atau PDF), mengekstrak teks, dan memanggil LLM.
    """
    try:
        # Update total file (ini belum termasuk pemisahan PDF menjadi halaman, 
        # kita akan hitung setiap file asli sebagai 1 entitas, atau bisa juga update total jika file itu PDF)
        
        for file_path in file_paths:
            # Periksa apakah file adalah PDF
            is_pdf = file_path.lower().endswith(".pdf")
            
            images_to_process = []
            if is_pdf:
                try:
                    # Split PDF into images
                    pdf_images = split_pdf_to_images(file_path, upload_dir)
                    images_to_process.extend(pdf_images)
                    
                    # Update total in status to reflect multiple pages
                    # Subtract 1 for the original PDF, add the number of pages
                    batch_status_dict[batch_id]["total"] += (len(pdf_images) - 1)
                except Exception as e:
                    batch_status_dict[batch_id]["results"].append({
                        "file": os.path.basename(file_path),
                        "status": "error",
                        "message": f"Gagal membaca PDF: {str(e)}"
                    })
                    batch_status_dict[batch_id]["completed"] += 1
                    continue
            else:
                images_to_process.append(file_path)
            
            # Proses setiap gambar (bisa dari file gambar asli atau halaman PDF)
            for image_path in images_to_process:
                try:
                    # 1. OCR
                    extracted_text = ocr_engine.extract_text(image_path)
                    
                    if not extracted_text.strip():
                        batch_status_dict[batch_id]["results"].append({
                            "file": os.path.basename(image_path),
                            "status": "success",
                            "data": None,
                            "message": "Tidak ada teks yang terdeteksi di gambar"
                        })
                    else:
                        # 2. LLM Parsing
                        parsed_data = await llm_client.parse_document(extracted_text)
                        
                        batch_status_dict[batch_id]["results"].append({
                            "file": os.path.basename(image_path),
                            "status": "success",
                            "raw_text": extracted_text,
                            "parsed_data": parsed_data
                        })
                except Exception as e:
                    batch_status_dict[batch_id]["results"].append({
                        "file": os.path.basename(image_path),
                        "status": "error",
                        "message": str(e)
                    })
                finally:
                    # Update completed count
                    batch_status_dict[batch_id]["completed"] += 1
                    
                    # Clean up temporary image
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception:
                            pass
            
            # Clean up original PDF file if it was split
            if is_pdf and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        # Tandai selesai
        batch_status_dict[batch_id]["status"] = "completed"

    except Exception as e:
        batch_status_dict[batch_id]["status"] = "failed"
        batch_status_dict[batch_id]["error"] = str(e)
