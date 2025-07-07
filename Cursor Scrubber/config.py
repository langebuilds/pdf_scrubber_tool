"""
Configuration settings for the PDF Redactor Tool
"""

# Company-specific terms to redact
COMPANY_TERMS = [
    "McIntosh",
    "McIntosh Laboratory",
    "McIntosh Laboratories",
    "McIntosh Laboratory Inc",
    "McIntosh Laboratories Inc",
    "McIntosh Lab",
    "McIntosh Labs",
]

# Drawing number patterns (regex)
DRAWING_PATTERNS = [
    r'\b\d{6}[A-Z]?\b',  # 015056A, 015061D
    r'\b\d{6}_\d{2}_[A-Z]_\d{4}\.cad\b',  # 015056_01_D_1536.cad
    r'\b[A-Z]{2,4}\d{4}\b',  # MA2375, VBW
]

# Email pattern
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Phone number patterns
PHONE_PATTERNS = [
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
    r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
    r'\b\d{10}\b',  # 1234567890
]

# Address patterns
ADDRESS_PATTERNS = [
    r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Way|Terrace|Ter)\b',
    r'\b[A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',  # City, State ZIP
]

# Technical content to preserve (regex patterns)
TECHNICAL_PRESERVE = [
    r'\b\d+\.\d+\b',  # Dimensions like 0.105
    r'\b±\d+\.?\d*°?\b',  # Tolerances like ±0.5°
    r'\b[A-Z]{2,4}\d{4}\b',  # Part numbers like MA2375, VBW
    r'\b\d+\s*(?:mm|cm|in|ft|m)\b',  # Measurements with units
    r'\b[A-Za-z]+\s+\d+\b',  # Material specs like "Steel 1018"
]

# Logo detection settings
LOGO_DETECTION = {
    'min_confidence': 0.30,  # Lowered threshold to catch more logo matches
    'template_matching_threshold': 0.30,  # Lowered threshold to catch more logo matches
    'edge_detection_sensitivity': 0.1,
    'min_logo_size': (50, 50),  # minimum width, height in pixels
    'max_logo_size': (500, 500),  # maximum width, height in pixels
}

# Redaction appearance
REDACTION_STYLE = {
    'fill_color': (1, 1, 1),  # White
    'border_color': (0, 0, 0),  # Black border
    'border_width': 1,
    'opacity': 1.0,
}

# File naming conventions
OUTPUT_SETTINGS = {
    'redacted_suffix': '_redacted',
    'audit_suffix': '_audit_log',
    'output_format': 'pdf',
    'audit_format': 'json',
}

# OCR settings
OCR_SETTINGS = {
    'language': 'eng',
    'config': '--oem 3 --psm 6',
    'dpi': 300,
    'preprocessing': {
        'denoise': True,
        'deskew': True,
        'contrast_enhancement': True,
    }
}

# Processing settings
PROCESSING = {
    'max_file_size_mb': 50,
    'supported_formats': ['.pdf'],
    'temp_dir': 'temp',
    'batch_size': 5,
    'timeout_seconds': 300,
}

# Audit log settings
AUDIT_SETTINGS = {
    'include_timestamp': True,
    'include_user_info': True,
    'log_level': 'INFO',
    'max_log_size_mb': 10,
    'backup_count': 5,
} 