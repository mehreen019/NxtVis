import fitz  # PyMuPDF
import json
import pytesseract
import cv2
import datetime
import numpy as np
import os
import faiss
import requests
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import time


TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

model = SentenceTransformer('all-MiniLM-L6-v2') 

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

def save_to_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save data to JSON file."""
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_cv_pdfs(cv_folder: Path) -> Dict[str, str]:
    """Process all CV PDFs in the given folder and save the results in a separate output folder.
    Returns a dictionary mapping CV names to their full text content."""
    if not cv_folder.is_dir():
        print(f"Error: {cv_folder} is not a valid directory.")
        return {}

    output_folder = cv_folder / "output"
    output_folder.mkdir(exist_ok=True)  

    pdf_files = list(cv_folder.glob("*.pdf")) 
    if not pdf_files:
        print("No PDF files found.")
        return {}

    cv_contents = {}
    for pdf_file in tqdm(pdf_files, desc="Processing CVs", unit="file"):
        pages = read_pdf(pdf_file)
        if pages:
            full_text = "\n".join(pages)
            cv_contents[pdf_file.stem] = full_text
            
            # Save individual CV data
            output_path = output_folder / f"{pdf_file.stem}.json"
            cv_data = {
                "source_file": str(pdf_file),
                "page_count": len(pages),
                "pages": pages,
                "full_text": full_text,
                "extracted_on": datetime.datetime.now().isoformat()
            }
            save_to_json(cv_data, output_path)
            print(f"Extracted {len(pages)} pages to {output_path}")
    
    return cv_contents

def process_job_descriptions(job_folder: Path) -> Dict[str, str]:
    """Process all job description files in the given folder.
    Returns a dictionary mapping job names to their text content."""
    if not job_folder.is_dir():
        print(f"Error: {job_folder} is not a valid directory.")
        return {}

    job_contents = {}
    
    # Process text files
    for text_file in job_folder.glob("*.txt"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                job_contents[text_file.stem] = content
                print(f"Processed job description: {text_file.name}")
        except Exception as e:
            print(f"Error reading job description {text_file}: {e}")
    
    # Process PDF files
    for pdf_file in job_folder.glob("*.pdf"):
        pages = read_pdf(pdf_file)
        if pages:
            full_text = "\n".join(pages)
            job_contents[pdf_file.stem] = full_text
            print(f"Processed job description PDF: {pdf_file.name}")
    
    return job_contents

def create_embeddings(texts: Dict[str, str]) -> Tuple[Dict[str, int], np.ndarray]:
    """Create embeddings for a dictionary of texts.
    Returns a mapping of text names to indices, and the embedding matrix."""
    # Create mapping of text names to indices
    name_to_idx = {name: idx for idx, name in enumerate(texts.keys())}
    
    # Create embeddings
    embeddings = []
    for name, text in tqdm(texts.items(), desc="Creating embeddings", unit="text"):
        embedding = model.encode(text)
        embeddings.append(embedding)
    
    return name_to_idx, np.array(embeddings)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index for the embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    
    # Normalize the vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add the vectors to the index
    index.add(embeddings)
    return index


def extract_cv_info_with_openai(cv_text: str, api_key: str) -> Dict[str, str]:
    """Extract key information from CV text using OpenAI API.
    Returns a dictionary with name, email, phone, and LinkedIn info."""

    if not api_key or " " in api_key:
        raise ValueError("Invalid API key. Please provide a valid OpenAI API key.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}"
    }

    truncated_text = cv_text[:4000]

    data = {
        "model": "gpt-3.5-turbo", 
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts information from CVs."},
            {"role": "user", "content": f"""
            Extract the following information from this CV. Return ONLY a JSON object with the keys: name, email, phone, linkedin.
            If any information is not found, use null as the value.
            
            CV:
            {truncated_text}
            
            Return a JSON object only, with no additional text:
            """}
        ],
        "max_tokens": 250,
        "temperature": 0.1
    }

    def call_openai_with_retries():
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too many requests
                    wait_time = 2 ** i
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        return None  # If all retries fail

    result = call_openai_with_retries()
    if not result:
        return {
            "name": None,
            "email": None,
            "phone": None,
            "linkedin": None,
            "error": "Failed to fetch data from OpenAI"
        }

    try:
        content = result['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Could not parse JSON from API response: {content}")
        return {
            "name": None,
            "email": None,
            "phone": None,
            "linkedin": None,
            "error": "Could not parse API response"
        }



def compare_cvs_with_job(cv_embeddings: np.ndarray, job_embedding: np.ndarray, 
                       cv_idx_map: Dict[str, int]) -> List[Dict[str, Any]]:
    """Compare CV embeddings with a job description embedding and rank candidates.
    Returns a sorted list of candidates with their similarity scores."""
    # Normalize the job embedding for cosine similarity
    job_embedding_norm = job_embedding / np.linalg.norm(job_embedding)
    
    # Calculate cosine similarity
    similarity_scores = np.dot(cv_embeddings, job_embedding_norm)
    
    # Create a list of candidates with scores
    candidates = []
    idx_to_cv = {idx: name for name, idx in cv_idx_map.items()}
    
    for idx, score in enumerate(similarity_scores):
        cv_name = idx_to_cv[idx]
        
        candidates.append({
            "cv_name": cv_name,
            "score": float(score) 
        })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return candidates

def main(cv_folder: Path, job_folder: Path, output_folder: Path, 
         api_type: str = None, api_key: str = None):
    """Main function to process CVs and job descriptions, compare them, and save results."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    print("Processing CV PDFs...")
    cv_contents = process_cv_pdfs(cv_folder)
    
    print("\nProcessing job descriptions...")
    job_contents = process_job_descriptions(job_folder)
    
    if not cv_contents:
        print("No CV contents found. Exiting.")
        return
    
    if not job_contents:
        print("No job descriptions found. Exiting.")
        return
    
    print("\nCreating CV embeddings...")
    cv_idx_map, cv_embeddings = create_embeddings(cv_contents)
    
    print("\nCreating job description embeddings...")
    job_idx_map, job_embeddings = create_embeddings(job_contents)
    

    print("\nBuilding FAISS index...")
    normalized_cv_embeddings = cv_embeddings.copy() 
    cv_index = build_faiss_index(normalized_cv_embeddings)
    
    print("\nComparing CVs with job descriptions...")
    top_matches = {}
    
    for job_name, job_idx in job_idx_map.items():
        job_embedding = job_embeddings[job_idx]
        
        # Get matches
        matches = compare_cvs_with_job(
            normalized_cv_embeddings, 
            job_embedding,
            cv_idx_map
        )
        
        if matches:
            top_match = matches[0]  # Get the top-ranked CV
            top_matches[job_name] = top_match
            print(f"Top match for job '{job_name}': {top_match['cv_name']} (score: {top_match['score']:.4f})")
    

    print("\nExtracting CV information for top matches...")
    all_results = {}
    
    for job_name, top_match in top_matches.items():
        cv_name = top_match["cv_name"]
        cv_text = cv_contents[cv_name]
        
        print(f"Extracting info for CV '{cv_name}' (top match for job '{job_name}')")
        
        # Extract CV information using the specified API
        if api_type and api_key:
            if api_type.lower() == "openai":
                info = extract_cv_info_with_openai(cv_text, api_key)
            else:
                print(f"Unknown API type: {api_type}. Skipping info extraction.")
                info = {
                    "name": None,
                    "email": None,
                    "phone": None,
                    "linkedin": None,
                    "error": f"Unknown API type: {api_type}"
                }
        else:
            print("No API type or key provided. Skipping info extraction.")
            info = {
                "name": None,
                "email": None,
                "phone": None,
                "linkedin": None,
                "error": "No API information provided"
            }
        
        top_match.update(info)
        

        all_results[job_name] = top_match

        job_results_path = output_folder / f"{job_name}_top_match.json"
        save_to_json(top_match, job_results_path)
        print(f"Saved top match for job '{job_name}' to {job_results_path}")
    

    all_results_path = output_folder / "all_job_top_matches.json"
    save_to_json(all_results, all_results_path)
    print(f"\nSaved all job top matching results to {all_results_path}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="CV and Job Description Matching System")
    parser.add_argument("cv_folder", type=str, help="Folder containing CV PDFs")
    parser.add_argument("job_folder", type=str, help="Folder containing job descriptions")
    parser.add_argument("--output", type=str, default="./results", help="Output folder for results")
    parser.add_argument("--api_type", type=str, choices=["openai", "anthropic"], 
                        help="Type of API to use (openai or anthropic)")
    parser.add_argument("--api_key", type=str, help="API key for the selected service")
    
    args = parser.parse_args()
    
    cv_folder = Path(args.cv_folder)
    job_folder = Path(args.job_folder)
    output_folder = Path(args.output)
    
    main(cv_folder, job_folder, output_folder, args.api_type, args.api_key)