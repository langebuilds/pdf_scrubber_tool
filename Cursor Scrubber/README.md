# PDF Logo Redaction Tool

A Python tool for automatically detecting and redacting logos from engineering drawings and other PDF documents.

## Features

- **Automated Logo Detection**: Uses OpenCV template matching to detect logos in PDFs
- **Batch Processing**: Process multiple PDF files automatically
- **High-Quality Redaction**: Rasterizes PDFs and applies precise redactions
- **Multiple Logo Support**: Can be configured for different logo templates
- **Audit Logging**: Tracks all redaction activities

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Logo Templates

Place your logo template images in the `templates/` folder:
- `templates/mcintosh_logo.png` (example template)
- Add additional templates as needed

### 3. Run Automated Redaction

**Process all PDFs in current directory:**
```bash
python automated_logo_redaction.py
```

**Process specific folders:**
```python
from automated_logo_redaction import automated_logo_redaction

automated_logo_redaction(
    input_folder="path/to/pdfs", 
    output_folder="path/to/output"
)
```

## Output

- **Redacted PDFs**: `filename_redacted.pdf`
- **Files without logos**: `filename_no_logo.pdf`
- **Output folder**: `redacted_output/`

## Configuration

### Logo Detection Parameters

The tool uses optimized parameters for McIntosh logos:
- **Template Scale**: 0.06 (40x9 pixels)
- **Correlation Threshold**: 0.3
- **Redaction Scale**: 6x expansion
- **Position Offset**: (-20, -15) pixels

### Customization

Modify `automated_logo_redaction.py` to adjust:
- `target_scale`: Template size for detection
- `threshold`: Detection sensitivity
- `scale_factor`: Redaction area size
- `offset_x/offset_y`: Position adjustments

## File Structure

```
├── automated_logo_redaction.py  # Main automation script
├── logo_detector.py             # Logo detection engine
├── pdf_redactor.py              # PDF processing utilities
├── text_processor.py            # Text extraction and processing
├── audit_logger.py              # Audit logging functionality
├── app.py                       # Streamlit web interface
├── utils.py                     # Utility functions
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── templates/                   # Logo template images
│   └── mcintosh_logo.png
└── redacted_output/             # Output folder for redacted PDFs
```

## How It Works

1. **PDF Rasterization**: Converts PDF pages to high-quality images
2. **Template Matching**: Uses OpenCV to find logo matches
3. **Coordinate Calculation**: Determines precise logo locations
4. **Redaction Application**: Draws white rectangles over logo areas
5. **PDF Recreation**: Saves redacted images as new PDFs

## Technical Details

- **PDF Processing**: PyMuPDF (fitz)
- **Image Processing**: OpenCV
- **Template Matching**: Normalized cross-correlation
- **Image Conversion**: PIL (Pillow)
- **Web Interface**: Streamlit (optional)

## Troubleshooting

### Logo Not Detected
- Check template quality and size
- Adjust `threshold` parameter
- Verify logo is visible in PDF

### Redaction Too Small/Large
- Modify `scale_factor` in the script
- Adjust `offset_x` and `offset_y` for positioning

### Processing Errors
- Ensure PDFs are not password-protected
- Check file permissions
- Verify all dependencies are installed

## License

This tool is designed for internal use in redacting sensitive company information from engineering drawings. 