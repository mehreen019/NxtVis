import fitz 
import json
import pytesseract
import cv2
import datetime
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
import numpy as np


TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def read_pdf(file_path: Path) -> Optional[List[str]]:
    """Extract text from a PDF file, using OCR for images if needed."""
    try:
        pages = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text("text").strip() 
                
                
                image_texts = []
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)  
                    img_bytes = base_image["image"]  
                    img_array = np.frombuffer(img_bytes, np.uint8)  
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  

                    
                    ocr_text = pytesseract.image_to_string(image).strip()
                    if ocr_text:
                        image_texts.append(ocr_text)

                full_text = text + "\n" + "\n".join(image_texts) if image_texts else text
                pages.append(full_text.strip())

        return pages if pages else None
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None

def save_to_json(pages: List[str], output_path: Path, source_file: Path) -> None:
    """Save extracted pages to JSON file with metadata."""
    data = {
        "source_file": str(source_file),
        "page_count": len(pages),
        "pages": pages,
        "extracted_on": datetime.datetime.now().isoformat()
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_cv_pdfs(cv_folder: Path) -> None:
    """Process all CV PDFs in the given folder and save the results in a separate output folder."""
    if not cv_folder.is_dir():
        print(f"Error: {cv_folder} is not a valid directory.")
        return

    output_folder = cv_folder / "output"
    output_folder.mkdir(exist_ok=True)  

    pdf_files = list(cv_folder.glob("*.pdf")) 
    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pages = read_pdf(pdf_file)
        if pages:
            output_path = output_folder / f"{pdf_file.stem}.json"
            save_to_json(pages, output_path, pdf_file)
            print(f"Extracted {len(pages)} pages to {output_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_reader.py <cv-folder>")
        sys.exit(1)

    cv_folder = Path(sys.argv[1])
    process_cv_pdfs(cv_folder)
