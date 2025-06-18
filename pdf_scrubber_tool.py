import re
import os
import json
from pathlib import Path
import streamlit as st

# Use alternative to PyMuPDF if unavailable
try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("Error: PyMuPDF (fitz) is not installed or not available in this environment.")

# Define redaction terms (expand as needed)
REDACTION_PATTERNS = [
    r"McIntosh(?: Laboratories| Laboratory)?(?: Inc\\.?| LLC)?",
    r"Atom Power,? Inc\\.?",
    r"\\d{1,5}\\s+\\w+\\s+(Street|St|Ave|Avenue|Blvd|Boulevard|Road|Rd|Drive|Dr)\\b",
    r"CHARLOTTE,? NC \\d{5}",
    r"\\(\\d{3}\\) \\d{3}-\\d{4}",
    r"\\b[A-Z]{2,10}\\s+LABS?\\b",
]

st.title("PDF Drawing Scrubber Tool")
st.write("Upload engineering PDFs to redact company names, addresses, and sensitive information.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    output_dir = Path("scrubbed")
    output_dir.mkdir(exist_ok=True)
    audit_log = {}

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            st.error(f"Failed to open {file_name}: {e}")
            continue

        redactions = []

        for page_num, page in enumerate(doc):
            for pattern in REDACTION_PATTERNS:
                try:
                    matches = page.search_for(pattern, flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_IGNORECASE)
                except Exception as e:
                    st.warning(f"Error searching on page {page_num + 1} in {file_name}: {e}")
                    continue

                for match in matches:
                    try:
                        preview = page.get_textbox(match)
                    except Exception:
                        preview = ""
                    redactions.append({
                        "page": page_num + 1,
                        "pattern": pattern,
                        "bbox": list(match),
                        "content_preview": preview[:100]
                    })
                    page.add_redact_annot(match, fill=(1, 1, 1))

        try:
            doc.apply_redactions()
            save_path = output_dir / file_name
            doc.save(str(save_path), garbage=4, deflate=True, clean=True)
            audit_log[file_name] = redactions
            st.success(f"Redacted and saved: {file_name}")
            with open(save_path, "rb") as f:
                st.download_button("Download Redacted PDF", f.read(), file_name=file_name)
        except Exception as e:
            st.error(f"Failed to save redacted {file_name}: {e}")
            audit_log[file_name] = []

    st.download_button(
        label="Download Audit Log (JSON)",
        data=json.dumps(audit_log, indent=2),
        file_name="audit_log.json",
        mime="application/json"
    )
