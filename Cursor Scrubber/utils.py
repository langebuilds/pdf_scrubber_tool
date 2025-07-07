"""
Utility functions for the PDF Redactor Tool
"""

import os
import re
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
from config import PROCESSING, TECHNICAL_PRESERVE


def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file is a valid PDF and meets size requirements.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > PROCESSING['max_file_size_mb']:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({PROCESSING['max_file_size_mb']}MB)"
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return False, "File is not a PDF"
        
        # Try to open with PyMuPDF to validate PDF structure
        try:
            doc = fitz.open(file_path)
            doc.close()
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
        
        return True, "Valid PDF file"
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"


def create_temp_directory() -> str:
    """
    Create a temporary directory for processing files.
    
    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="pdf_redactor_")
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up temporary directory and its contents.
    
    Args:
        temp_dir: Path to the temporary directory
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not cleanup temp directory {temp_dir}: {e}")


def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """
    Extract basic information about a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF information
    """
    try:
        doc = fitz.open(file_path)
        info = {
            'page_count': len(doc),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'has_text': False,
            'has_images': False,
            'dimensions': []
        }
        
        # Check each page for text and images
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            info['dimensions'].append({
                'width': rect.width,
                'height': rect.height
            })
            
            # Check for text
            text = page.get_text()  # type: ignore
            if text.strip():
                info['has_text'] = True
            
            # Check for images
            image_list = page.get_images()
            if image_list:
                info['has_images'] = True
        
        doc.close()
        return info
        
    except Exception as e:
        return {
            'error': f"Could not extract PDF info: {str(e)}",
            'page_count': 0,
            'file_size_mb': 0,
            'has_text': False,
            'has_images': False,
            'dimensions': []
        }


def is_technical_content(text: str) -> bool:
    """
    Check if text contains technical content that should be preserved.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains technical content to preserve
    """
    text = text.strip()
    if not text:
        return False
    
    # Check against technical preservation patterns
    for pattern in TECHNICAL_PRESERVE:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Additional checks for technical content
    technical_indicators = [
        # Dimensions and measurements
        r'\d+\.\d+',  # Decimal numbers
        r'\d+\s*(?:mm|cm|in|ft|m|°|deg)',  # Measurements with units
        r'±\d+',  # Tolerances
        r'[A-Z]{2,4}\d{4}',  # Part numbers
        
        # Technical terms
        r'\b(?:diameter|radius|length|width|height|thickness|angle|tolerance|spec|material|steel|aluminum|copper|wire|gauge)\b',
        
        # CAD/Engineering terms
        r'\b(?:dimension|measurement|scale|drawing|part|assembly|component|bolt|screw|nut|washer|bearing)\b'
    ]
    
    for pattern in technical_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def extract_text_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from PDF with their positions and properties.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text block dictionaries with position and content info
    """
    text_blocks = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks
            blocks = page.get_text("dict")  # type: ignore
            
            for block in blocks.get("blocks", []):
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_block = {
                                'page': page_num + 1,
                                'text': span["text"],
                                'bbox': span["bbox"],
                                'font': span["font"],
                                'size': span["size"],
                                'color': span["color"],
                                'flags': span["flags"]
                            }
                            text_blocks.append(text_block)
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting text blocks: {e}")
    
    return text_blocks


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF pages to images for OCR processing.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion
        
    Returns:
        List of numpy arrays representing page images
    """
    images = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Calculate zoom factor for desired DPI
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to image
            pix = page.get_pixmap(matrix=mat)  # type: ignore
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            images.append(img)
        
        doc.close()
        
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    
    return images


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Deskew if needed
    # (Simple deskew implementation - can be enhanced)
    coords = np.column_stack(np.where(enhanced > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) > 0.5:
            (h, w) = enhanced.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return enhanced


def save_audit_log(audit_data: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Save audit log to JSON file.
    
    Args:
        audit_data: List of audit entries
        output_path: Path to save the audit log
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving audit log: {e}")
        return False


def generate_output_filename(input_path: str, suffix: str = "_redacted") -> str:
    """
    Generate output filename based on input file.
    
    Args:
        input_path: Path to input file
        suffix: Suffix to add to filename
        
    Returns:
        Generated output filename
    """
    input_file = Path(input_path)
    output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
    return str(input_file.parent / output_name)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = int(size_bytes / 1024.0)
    return f"{size_bytes:.1f} TB" 