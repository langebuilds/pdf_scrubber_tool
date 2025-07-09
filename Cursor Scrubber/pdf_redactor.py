"""
Main PDF Redaction Engine
"""

import os
import sys

# Try multiple ways to import PyMuPDF
try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF imported successfully as 'fitz'")
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import PyMuPDF as fitz
        print("✅ PyMuPDF imported successfully as 'PyMuPDF'")
        FITZ_AVAILABLE = True
    except ImportError:
        try:
            from PyMuPDF import fitz
            print("✅ PyMuPDF imported successfully from 'PyMuPDF'")
            FITZ_AVAILABLE = True
        except ImportError:
            print("⚠️ PyMuPDF not available, using pdfplumber as fallback")
            FITZ_AVAILABLE = False
            try:
                import pdfplumber
                print("✅ pdfplumber imported successfully as fallback")
            except ImportError:
                print("❌ Neither PyMuPDF nor pdfplumber available")
                sys.exit(1)

import numpy as np
import cv2
from typing import List, Dict, Any
from utils import (
    validate_pdf_file, extract_text_blocks, create_temp_directory, cleanup_temp_directory,
    convert_pdf_to_images, generate_output_filename, save_audit_log
)
from text_processor import TextProcessor
from logo_detector import LogoDetector
from audit_logger import AuditLogger
from PIL import Image
import io
import base64

TEMPLATE_DIR = "templates"
THUMBNAIL_SIZE = (120, 60)


def load_logo_templates() -> List[np.ndarray]:
    """Load all logo templates from the templates/ directory as grayscale images."""
    templates = []
    template_dir = "templates"
    
    if not os.path.exists(template_dir):
        print(f"Warning: Template directory '{template_dir}' not found")
        return templates
    
    for fname in os.listdir(template_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(template_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append(img)
                print(f"Loaded logo template: {fname} (size: {img.shape})")
            else:
                print(f"Failed to load logo template: {fname}")
    
    print(f"Total logo templates loaded: {len(templates)}")
    return templates


def get_thumbnail(image: np.ndarray, bbox: List[int]) -> str:
    """Crop and resize a region from the image, return as base64 PNG string."""
    x0, y0, x1, y1 = map(int, bbox)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    thumb = cv2.resize(crop, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(thumb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def redact_pdf(
    pdf_path: str,
    output_dir: str = "redacted_output",
    method: str = "auto",
    logo_padding: int = 0  # Remove padding - use exact logo boundaries
) -> Dict[str, Any]:
    """
    Redact sensitive info from a PDF, save redacted PDF and audit log.
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid, msg = validate_pdf_file(pdf_path)
    if not valid:
        return {"error": msg}

    if not FITZ_AVAILABLE:
        return {"error": "PyMuPDF is required for PDF processing. Please install with: pip install PyMuPDF"}

    doc = fitz.open(pdf_path)
    
    # Try different text extraction methods
    print(f"PDF has {len(doc)} pages")
    
    # Method 1: Get text blocks using get_text("dict")
    text_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            # Try the dict method first
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
        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
            # Fallback: try simple text extraction
            try:
                text = page.get_text()  # type: ignore
                if text.strip():
                    rect = page.rect
                    text_block = {
                        'page': page_num + 1,
                        'text': text.strip(),
                        'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                        'font': 'unknown',
                        'size': 12,
                        'color': 0,
                        'flags': 0
                    }
                    text_blocks.append(text_block)
            except Exception as e2:
                print(f"Fallback text extraction also failed: {e2}")
    
    print(f"Extracted {len(text_blocks)} text blocks from PDF")
    
    # If no text blocks found, try OCR
    if len(text_blocks) == 0:
        print("No text blocks found, attempting OCR...")
        try:
            import pytesseract
            page_images = convert_pdf_to_images(pdf_path, dpi=300)
            for page_num, img in enumerate(page_images):
                # Convert to grayscale for OCR
                if len(img.shape) == 3:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = img.copy()
                
                # Perform OCR
                text = pytesseract.image_to_string(gray_img)
                if text.strip():
                    # Get image dimensions for bbox
                    h, w = gray_img.shape
                    text_block = {
                        'page': page_num + 1,
                        'text': text.strip(),
                        'bbox': [0, 0, w, h],
                        'font': 'ocr',
                        'size': 12,
                        'color': 0,
                        'flags': 0
                    }
                    text_blocks.append(text_block)
                    print(f"OCR found text on page {page_num + 1}: {text[:100]}...")
        except Exception as e:
            print(f"OCR failed: {e}")
    
    # Show some sample text blocks
    for i, block in enumerate(text_blocks[:5]):
        print(f"  Text block {i+1}: '{block.get('text', '')[:50]}...' at bbox {block.get('bbox', [])}")
    
    text_processor = TextProcessor()
    logo_detector = LogoDetector()
    audit = AuditLogger(filename=os.path.basename(pdf_path))
    logo_templates = load_logo_templates()
    temp_dir = create_temp_directory()
    page_images = convert_pdf_to_images(pdf_path, dpi=400)

    # Redaction overlays to apply: {page_num: [ (bbox, reason, type, thumbnail) ]}
    overlays = {i: [] for i in range(len(doc))}

    # --- Text Redaction ---
    redaction_candidates = text_processor.get_text_blocks_for_redaction(text_blocks)
    print(f"Text redaction candidates found: {len(redaction_candidates)}")
    
    for i, candidate in enumerate(redaction_candidates[:10]):  # Show first 10
        print(f"  Text candidate {i+1}: '{candidate['text']}' - {candidate['reason']}")
    
    merged_redactions = text_processor.merge_overlapping_redactions(redaction_candidates)
    print(f"After merging: {len(merged_redactions)} text redactions")
    
    for red in merged_redactions:
        page_idx = red['page'] - 1
        bbox = red['bbox']
        overlays[page_idx].append((bbox, red['reason'], 'text', None))
        # Thumbnail
        thumb = get_thumbnail(page_images[page_idx], bbox)
        audit.add_text_redaction(
            page_number=red['page'],
            original_text=red['text'],
            bbox=bbox,
            reason=red['reason'],
            method='text_layer',
            confidence=red.get('confidence', 1.0),
            font_info={"font": red.get('font'), "size": red.get('size')},
            color_info={"color": red.get('color')}
        )
        audit.entries[-1].thumbnail_b64 = thumb  # Add thumbnail to audit entry

    # --- Logo Redaction ---
    print(f"Processing {len(page_images)} pages for logo detection...")
    
    for page_idx, img in enumerate(page_images):
        print(f"Processing page {page_idx + 1} for logos...")
        
        # Get PDF page dimensions for coordinate conversion
        pdf_page = doc[page_idx]
        pdf_rect = pdf_page.rect
        
        # Calculate scaling factors from image to PDF coordinates (store for later use)
        img_h, img_w = img.shape[:2] if len(img.shape) == 3 else img.shape
        pdf_h, pdf_w = pdf_rect.height, pdf_rect.width
        scale_x = pdf_w / img_w
        scale_y = pdf_h / img_h
        print(f"    Page image size: {img.shape}")
        print(f"    PDF page dimensions: {pdf_rect}")
        print(f"    pdf_w: {pdf_w}, pdf_h: {pdf_h}")
        print(f"    Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        # Template matching for each logo template
        for i, template in enumerate(logo_templates):
            # Convert page image to grayscale for template matching
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img.copy()
            
            print(f"    Template {i+1} original size: {template.shape}")
            
            # Try multiple scales for better detection
            scales = [0.6, 0.8, 1.0, 1.2]  # Optimized scale range for common logo sizes
            best_matches = []
            
            for scale in scales:
                # Resize template
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Skip if template becomes too large or too small
                if new_h >= gray_img.shape[0] or new_w >= gray_img.shape[1] or new_h < 20 or new_w < 20:
                    continue
                
                template_resized = cv2.resize(template, (new_w, new_h))
                print(f"    Trying scale {scale}: template size {template_resized.shape}")
                
                # Perform template matching
                res = cv2.matchTemplate(gray_img, template_resized, cv2.TM_CCOEFF_NORMED)
                
                # Find best and second-best match for this scale
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                res_flat = res.flatten()
                if res_flat.size > 1:
                    res_flat[np.argmax(res_flat)] = -1  # Exclude the best
                    second_best = np.max(res_flat)
                else:
                    second_best = 0
                print(f"      Best match: confidence={max_val:.3f}, location={max_loc}, second best: {second_best:.3f}")
                
                # If we found a good match, add it
                threshold = 0.28  # Balanced threshold - not too low, not too high
                if max_val >= threshold:
                    # Store BOTH image and PDF coordinates
                    img_bbox = [max_loc[0], max_loc[1], max_loc[0] + new_w, max_loc[1] + new_h]
                    pdf_bbox = [
                        img_bbox[0] * scale_x,
                        img_bbox[1] * scale_y,
                        img_bbox[2] * scale_x,
                        img_bbox[3] * scale_y
                    ]
                    print(f"      Image bbox: {img_bbox}")
                    print(f"      PDF bbox: {pdf_bbox}")
                    
                    # Additional validation to filter out false positives
                    bbox_width = pdf_bbox[2] - pdf_bbox[0]
                    bbox_height = pdf_bbox[3] - pdf_bbox[1]
                    bbox_area = bbox_width * bbox_height
                    
                    # Filter out very large detections (likely false positives)
                    max_area = 50000  # Increased to allow larger logos
                    if bbox_area > max_area:
                        print(f"      ❌ Detection too large (area: {bbox_area:.1f}), likely false positive")
                        continue
                    
                    # Filter out detections that are too wide (likely false positives)
                    if bbox_width > 600:  # Increased to allow wider logos
                        print(f"      ❌ Detection too wide (width: {bbox_width:.1f}), likely false positive")
                        continue
                    
                    # Check if bbox is within page bounds
                    if (0 <= pdf_bbox[0] < pdf_w and 0 <= pdf_bbox[1] < pdf_h and 
                        pdf_bbox[2] <= pdf_w and pdf_bbox[3] <= pdf_h and
                        (pdf_bbox[2] - pdf_bbox[0]) > 10 and (pdf_bbox[3] - pdf_bbox[1]) > 10):
                        # --- Spatial filtering for logo templates ---
                        expected_regions = {
                            1: [  # Template 2: atom_power_full_logo.png
                                (0, pdf_h * 0.6, pdf_w * 0.5, pdf_h),      # bottom left
                                (pdf_w * 0.5, pdf_h * 0.6, pdf_w, pdf_h),  # bottom right
                                (pdf_w * 0.3, pdf_h * 0.4, pdf_w * 0.7, pdf_h * 0.8),  # middle region (optional)
                            ],
                            2: [  # Template 3: atom_power_logo.png
                                (pdf_w * 0.5, pdf_h * 0.6, pdf_w, pdf_h),  # bottom right
                                (pdf_w * 0.3, pdf_h * 0.4, pdf_w * 0.7, pdf_h * 0.8),  # middle region (optional)
                            ],
                            3: [  # Template 4: mcintosh_logo.png
                                (pdf_w * 0.7, pdf_h * 0.7, pdf_w, pdf_h),  # bottom right
                            ],
                            0: [  # Template 1: mcintosh_full_logo.png
                                (pdf_w * 0.7, pdf_h * 0.7, pdf_w, pdf_h),  # bottom right
                            ],
                        }
                        regions = expected_regions.get(i, [])
                        if regions:
                            center_x = (pdf_bbox[0] + pdf_bbox[2]) / 2
                            center_y = (pdf_bbox[1] + pdf_bbox[3]) / 2
                            print(f"        Center of bbox: ({center_x:.2f}, {center_y:.2f})")
                            for idx, (x0, y0, x1, y1) in enumerate(regions):
                                print(f"        Region {idx+1} for template {i+1}: x0={x0:.2f}, y0={y0:.2f}, x1={x1:.2f}, y1={y1:.2f}")
                            in_region = any(
                                (x0 <= center_x <= x1 and y0 <= center_y <= y1)
                                for (x0, y0, x1, y1) in regions
                            )
                            print(f"        In expected region: {in_region}")
                            # More permissive spatial filtering - allow logos in broader areas
                            if not in_region:
                                # For template 1 (mcintosh_full_logo), require it to be in the bottom-right quadrant
                                if i == 0 and center_x > pdf_w * 0.5 and center_y > pdf_h * 0.5:
                                    print(f"        ✅ Template 1 allowed in bottom-right quadrant")
                                else:
                                    print(f"      ❌ Detection for template {i+1} outside expected region, skipping")
                                    continue
                        # --- Additional ratio-based size filter for Atom Power logos ---
                        if i in [1, 2]:  # Template 2 or 3 (atom_power)
                            min_width_ratio = 0.2  # Must be at least 20% of page width
                            min_height_ratio = 0.05  # Must be at least 5% of page height
                            if bbox_width < min_width_ratio * pdf_w or bbox_height < min_height_ratio * pdf_h:
                                print(f"      ❌ Atom Power logo detection too small (width: {bbox_width:.1f}, height: {bbox_height:.1f}), skipping")
                                continue
                        # --- Additional upper bound and aspect ratio filter for Atom Power logos ---
                        if i in [1, 2]:  # Template 2 or 3 (atom_power)
                            max_width_ratio = 0.35  # No wider than 35% of page width
                            max_height_ratio = 0.12  # No taller than 12% of page height
                            min_aspect = 2.5  # Real logo is wide, but not extremely so
                            max_aspect = 7.0  # Real logo is not extremely wide
                            aspect = bbox_width / bbox_height if bbox_height > 0 else 0
                            if bbox_width > max_width_ratio * pdf_w or bbox_height > max_height_ratio * pdf_h:
                                print(f"      ❌ Atom Power logo detection too large (width: {bbox_width:.1f}, height: {bbox_height:.1f}), skipping")
                                continue
                            if not (min_aspect <= aspect <= max_aspect):
                                print(f"      ❌ Atom Power logo detection aspect ratio {aspect:.2f} out of range, skipping")
                                continue
                        best_matches.append({
                            'bbox': pdf_bbox,
                            'img_bbox': img_bbox,  # Store original image bbox
                            'confidence': max_val,
                            'scale': scale,
                            'size': (new_w, new_h),
                            'area': bbox_area,
                            'width': bbox_width
                        })
                        print(f"      ✅ Valid match found: pdf_bbox={pdf_bbox}, confidence={max_val:.3f}, area={bbox_area:.1f}")
                    else:
                        print(f"      ❌ Match out of bounds: pdf_bbox={pdf_bbox}")
            # Use up to 2 best matches from all scales
            if best_matches:
                # Sort by confidence and take up to 2
                best_matches.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Only use the BEST match, not multiple matches from different scales
                # This prevents overlapping redactions and ensures we get the most accurate detection
                best_match = best_matches[0]  # Only use the single best match
                print(f"  Using best match: confidence={best_match['confidence']:.3f}, bbox={best_match['bbox']}, scale={best_match['scale']}")
                
                # Add padding to make redaction more visible and ensure full coverage
                # Template 3 (mcintosh_logo.png) is smaller (672x156) and needs more padding
                # to ensure it fully covers the actual logo in the PDF
                if i == 3:  # Template 3 (mcintosh_logo.png)
                    padding = 15  # Extra padding for the smaller template
                else:
                    padding = logo_padding  # Default padding for other templates
                
                padded_bbox = [
                    best_match['bbox'][0] - padding,
                    best_match['bbox'][1] - padding,
                    best_match['bbox'][2] + padding,
                    best_match['bbox'][3] + padding
                ]
                # Ensure padded bbox is within page bounds
                padded_bbox[0] = max(0, padded_bbox[0])
                padded_bbox[1] = max(0, padded_bbox[1])
                padded_bbox[2] = min(pdf_w, padded_bbox[2])
                padded_bbox[3] = min(pdf_h, padded_bbox[3])
                print(f"  Original bbox: {best_match['bbox']}")
                print(f"  Padded bbox: {padded_bbox}")
                
                # Store BOTH the padded PDF bbox AND the original image bbox
                overlays[page_idx].append((
                    padded_bbox, 
                    f'Logo detected (template {i+1})', 
                    'logo', 
                    best_match['img_bbox']  # Store original image bbox for direct use
                ))
                
                # Use original image bbox for thumbnail
                thumb = get_thumbnail(img, best_match['img_bbox'])
                audit.add_logo_redaction(
                    page_number=page_idx+1,
                    logo_description=f"Template {i+1} match (scale {best_match['scale']})",
                    bbox=padded_bbox,  # Use padded bbox for audit
                    confidence=best_match['confidence']
                )
                audit.entries[-1].thumbnail_b64 = thumb
            else:
                print(f"  No valid matches found for template {i+1} across all scales")
        
        # --- Best Match Wins System ---
        # Since there should only be one logo per page, keep only the best detection
        logo_overlays = [(i, bbox, reason, typ, img_bbox) for i, (bbox, reason, typ, img_bbox) in enumerate(overlays[page_idx]) if typ == 'logo']
        if len(logo_overlays) > 1:
            print(f"  Found {len(logo_overlays)} logo detections, selecting best match...")
            
            # Find the best detection based on confidence and location
            best_idx = None
            best_score = -1
            
            for i, (orig_idx, bbox, reason, typ, img_bbox) in enumerate(logo_overlays):
                # Extract confidence from audit entries
                confidence = 0.3  # Default confidence
                try:
                    for entry in audit.entries:
                        if entry.content_type == "logo" and entry.original_content == reason:
                            confidence = entry.confidence
                            break
                except:
                    pass
                
                # Calculate location preference (prefer bottom-right)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Location score: prefer bottom-right (higher values)
                location_score = (center_x / pdf_w) + (center_y / pdf_h)
                
                # Combined score: 80% confidence + 20% location preference
                score = 0.8 * confidence + 0.2 * location_score
                
                print(f"    {reason}: confidence={confidence:.3f}, location_score={location_score:.3f}, total_score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Remove all but the best detection
            for i, (orig_idx, bbox, reason, typ, img_bbox) in enumerate(logo_overlays):
                if i != best_idx:
                    print(f"    Removing inferior detection: {reason}")
                    
                    # Remove from overlays
                    overlays[page_idx].pop(orig_idx)
                    
                    # Remove from audit entries
                    for j, entry in enumerate(audit.entries):
                        if entry.content_type == "logo" and entry.original_content == reason:
                            audit.entries.pop(j)
                            break
        
        # Print summary of overlays for this page
        print(f"  Total overlays for page {page_idx + 1}: {len(overlays[page_idx])}")
        for j, (bbox, reason, typ, img_bbox) in enumerate(overlays[page_idx]):
            print(f"    Overlay {j+1}: {reason} at bbox {bbox}")

    # --- Apply Redactions ---
    total_redactions_applied = 0
    
    # Create a new document for the redacted PDF
    new_doc = fitz.open()
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_redactions = len(overlays[page_idx])
        total_redactions_applied += page_redactions
        print(f"Applying {page_redactions} redactions to page {page_idx + 1}")
        
        # Use the SAME resolution as logo detection (400 DPI)
        zoom = 400 / 72  # Same as convert_pdf_to_images with dpi=400
        mat = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # type: ignore
        img_data = mat.tobytes("png")
        
        # Convert to OpenCV format
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Apply redactions to the image
        for bbox, reason, typ, img_bbox in overlays[page_idx]:
            print(f"  Adding redaction: {reason} at bbox {bbox}")
            
            if typ == 'logo' and img_bbox is not None:
                # For logos, use the stored original image coordinates directly
                img_x, img_y, img_x2, img_y2 = img_bbox
                img_w_redact = img_x2 - img_x
                img_h_redact = img_y2 - img_y
                
                # Add minimal buffer to ensure full coverage (1 pixel on each side)
                buffer = 1  # Reduced from 2 to 1
                img_x = max(0, img_x - buffer)
                img_y = max(0, img_y - buffer)
                img_w_redact = min(img.shape[1] - img_x, img_w_redact + 2 * buffer)
                img_h_redact = min(img.shape[0] - img_y, img_h_redact + 2 * buffer)
                
                print(f"    Using stored image coordinates: ({img_x}, {img_y}) with size ({img_w_redact}, {img_h_redact})")
            else:
                # For text redactions, use coordinate conversion
                img_h, img_w = img.shape[:2] if len(img.shape) == 3 else img.shape
                pdf_h, pdf_w = page.rect.height, page.rect.width
                scale_x = img_w / pdf_w
                scale_y = img_h / pdf_h
                
                img_x = int(bbox[0] * scale_x)
                img_y = int(bbox[1] * scale_y)
                img_w_redact = int((bbox[2] - bbox[0]) * scale_x)
                img_h_redact = int((bbox[3] - bbox[1]) * scale_y)
                
                print(f"    Converted coordinates: ({img_x}, {img_y}) with size ({img_w_redact}, {img_h_redact})")
            
            # Draw black rectangle on the image
            cv2.rectangle(img, (img_x, img_y), (img_x + img_w_redact, img_y + img_h_redact), (0, 0, 0), -1)
            print(f"    ✅ Black rectangle drawn on image")
        
        # Convert back to PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Create new page with same dimensions as original
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)  # type: ignore
        
        # Convert PIL image to PDF page
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Insert the redacted image
        new_page.insert_image(new_page.rect, stream=img_bytes.getvalue())  # type: ignore
    
    print(f"\n=== REDACTION SUMMARY ===")
    print(f"Total redactions applied: {total_redactions_applied}")
    print(f"Text redactions: {len(merged_redactions)}")
    print(f"Logo redactions: {total_redactions_applied - len(merged_redactions)}")
    print(f"Audit entries: {len(audit.entries)}")

    # --- Save Redacted PDF ---
    out_pdf = os.path.join(output_dir, generate_output_filename(os.path.basename(pdf_path), suffix="_redacted"))
    new_doc.save(out_pdf)
    new_doc.close()
    doc.close()

    # --- Save Audit Log ---
    input_file = os.path.basename(pdf_path)
    base_name = os.path.splitext(input_file)[0]
    audit_filename = f"{base_name}_audit_log.json"
    out_audit = os.path.join(output_dir, audit_filename)
    audit_data = [entry.__dict__ for entry in audit.entries]
    # No need to move thumbnail_b64, it's already in the dataclass
    with open(out_audit, 'w', encoding='utf-8') as f:
        import json
        json.dump(audit_data, f, indent=2)

    cleanup_temp_directory(temp_dir)
    return {
        "redacted_pdf": out_pdf,
        "audit_log": out_audit,
        "total_redactions": len(audit.entries)
    }


def batch_redact_pdfs(pdf_paths: List[str], output_dir: str = "redacted_output") -> List[Dict[str, Any]]:
    """Redact a batch of PDFs and return a list of results."""
    results = []
    for pdf_path in pdf_paths:
        result = redact_pdf(pdf_path, output_dir=output_dir)
        results.append(result)
    return results 