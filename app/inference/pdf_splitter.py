import fitz  # PyMuPDF
import os
import uuid

def split_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """
    Membaca file PDF dan mengubah setiap halamannya menjadi file gambar (PNG).
    Mengembalikan daftar path file gambar yang dihasilkan.
    """
    image_paths = []
    
    # Buka dokumen PDF
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Atur resolusi (zoom) untuk kualitas OCR yang lebih baik
        # zoom 2.0 = 144 DPI (standar 72 DPI)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Buat nama file unik untuk halaman ini
        image_filename = f"{uuid.uuid4()}_page_{page_num + 1}.png"
        image_path = os.path.join(output_dir, image_filename)
        
        # Simpan gambar
        pix.save(image_path)
        image_paths.append(image_path)
        
    pdf_document.close()
    
    return image_paths
