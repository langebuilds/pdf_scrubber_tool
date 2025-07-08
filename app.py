import streamlit as st
import os
import tempfile
import sys

# Add the Cursor Scrubber directory to the path
sys.path.append("Cursor Scrubber")

from pdf_redactor import redact_pdf
from PIL import Image
import shutil
import time
from datetime import datetime
import zipfile
import io

# Import database from the correct location
try:
    from database import db
except ImportError:
    # If not found in current directory, try Cursor Scrubber directory
    sys.path.insert(0, "Cursor Scrubber")
    from database import db

TEMPLATE_DIR = "templates"
OUTPUT_DIR = "redacted_output"

# Initialize session state
if 'redaction_results' not in st.session_state:
    st.session_state.redaction_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

st.set_page_config(page_title="PDF Redactor Tool", layout="wide")
st.title("🔒 PDF Redactor Tool for Engineering Drawings")
st.markdown("""
Automate removal of company/client-specific information from engineering drawings while preserving all technical content.

- **Redacts:** Company names, emails, phone numbers, addresses, drawing numbers, and logos
- **Preserves:** Dimensions, part numbers, tolerances, materials, and all technical data
- **Batch:** Upload and process multiple PDFs at once
- **Audit:** Download a JSON log with thumbnails of all redactions
""")

# --- Sidebar: Logo Template Management ---
st.sidebar.header("Logo Template Management")
logo_files = st.sidebar.file_uploader(
    "Upload new logo templates (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="logo_upload"
)
if logo_files:
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    for logo_file in logo_files:
        logo_path = os.path.join(TEMPLATE_DIR, logo_file.name)
        with open(logo_path, "wb") as f:
            f.write(logo_file.read())
    st.sidebar.success(f"Uploaded {len(logo_files)} logo template(s).")

# Show current logo templates
if os.path.exists(TEMPLATE_DIR):
    st.sidebar.markdown("**Current Logo Templates:**")
    for fname in os.listdir(TEMPLATE_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            st.sidebar.image(os.path.join(TEMPLATE_DIR, fname), width=100, caption=fname)

# --- Main: PDF Redaction ---
st.header("1. Upload PDF Files")
uploaded_files = st.file_uploader(
    "Select one or more PDF files to redact:",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_upload"
)

st.header("2. Redaction Settings")
col1, col2 = st.columns(2)

with col1:
    precision_mode = st.selectbox(
        "Redaction Precision:",
        ["Precise (Recommended)", "Standard", "Conservative"],
        help="Precise: Minimal padding around logos. Standard: Slight padding. Conservative: More padding for safety."
    )

with col2:
    st.markdown("**Current Settings:**")
    if precision_mode == "Precise (Recommended)":
        logo_padding = 3
        st.markdown("• Logo padding: 3 points")
        st.markdown("• Optimal for most cases")
    elif precision_mode == "Standard":
        logo_padding = 5
        st.markdown("• Logo padding: 5 points")
        st.markdown("• Balanced approach")
    else:  # Conservative
        logo_padding = 8
        st.markdown("• Logo padding: 8 points")
        st.markdown("• Extra safety margin")

# Add a "New Processing" button to clear previous results
if st.button("🔄 Start New Processing", disabled=not uploaded_files):
    # Clear previous results
    st.session_state.redaction_results = []
    st.session_state.processing_complete = False
    
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        st.info("Processing... This may take a few moments for large files.")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded files to temp dir
        temp_dir = tempfile.mkdtemp(prefix="pdf_redactor_ui_")
        input_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            input_paths.append(file_path)
        
        # Process each file
        results = []
        total_files = len(input_paths)
        
        for i, file_path in enumerate(input_paths):
            filename = os.path.basename(file_path)
            status_text.text(f"Processing {i+1}/{total_files}: {filename}")
            
            try:
                # Use optimized redact_pdf function
                result = redact_pdf(
                    pdf_path=file_path,
                    output_dir=OUTPUT_DIR,
                    logo_padding=logo_padding
                )
                
                if "error" not in result:
                    # Add to database
                    file_size = os.path.getsize(result['redacted_pdf']) if os.path.exists(result['redacted_pdf']) else 0
                    db.add_processed_file(
                        original_filename=filename,
                        redacted_filename=result['redacted_pdf'],
                        audit_filename=result['audit_log'],
                        total_redactions=result['total_redactions'],
                        file_size_bytes=file_size
                    )
                    results.append(result)
                    st.success(f"✅ {filename}: {result['total_redactions']} redactions applied")
                else:
                    st.error(f"❌ {filename}: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ {filename}: Error - {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.redaction_results = results
        st.session_state.processing_complete = True
        
        # Clean up temp input dir
        shutil.rmtree(temp_dir)
        
        # Rerun to show results
        st.rerun()

# Show results if processing is complete
if st.session_state.processing_complete and st.session_state.redaction_results:
    st.success(f"🎉 Redaction complete! Successfully processed {len(st.session_state.redaction_results)} file(s).")
    
    # Summary
    total_redactions = sum(r['total_redactions'] for r in st.session_state.redaction_results)
    st.metric("Total Redactions Applied", total_redactions)
    
    # Download options
    st.subheader("📥 Download Results")
    
    for i, result in enumerate(st.session_state.redaction_results):
        filename = os.path.basename(result['redacted_pdf'])
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(result['redacted_pdf']):
                with open(result['redacted_pdf'], "rb") as f:
                    st.download_button(
                        label=f"📄 {filename}",
                        data=f,
                        file_name=filename,
                        mime="application/pdf",
                        key=f"pdf_download_{i}"  # Unique key for each button
                    )
            else:
                st.error(f"File not found: {filename}")
        
        with col2:
            audit_filename = os.path.basename(result['audit_log'])
            if os.path.exists(result['audit_log']):
                with open(result['audit_log'], "rb") as f:
                    st.download_button(
                        label=f"📊 Audit Log",
                        data=f,
                        file_name=audit_filename,
                        mime="application/json",
                        key=f"audit_download_{i}"  # Unique key for each button
                    )
            else:
                st.error(f"Audit log not found: {audit_filename}")

    # Download All functionality
    st.subheader("📦 Batch Download")
    
    # Create a function to generate the zip file
    def create_redacted_pdfs_zip():
        zip_buffer = io.BytesIO()
        pdf_count = 0
        audit_count = 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for result in st.session_state.redaction_results:
                # Add redacted PDF
                if os.path.exists(result['redacted_pdf']):
                    pdf_filename = os.path.basename(result['redacted_pdf'])
                    with open(result['redacted_pdf'], 'rb') as f:
                        zip_file.writestr(pdf_filename, f.read())
                    pdf_count += 1
                
                # Add audit log
                if os.path.exists(result['audit_log']):
                    audit_filename = os.path.basename(result['audit_log'])
                    with open(result['audit_log'], 'rb') as f:
                        zip_file.writestr(audit_filename, f.read())
                    audit_count += 1
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue(), pdf_count, audit_count
    
    # Download All button
    if st.session_state.redaction_results:
        zip_data, pdf_count, audit_count = create_redacted_pdfs_zip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"redacted_pdfs_{timestamp}.zip"
        
        st.download_button(
            label="📦 Download All Redacted PDFs & Audit Logs",
            data=zip_data,
            file_name=zip_filename,
            mime="application/zip",
            help="Downloads all redacted PDFs and their corresponding audit logs in a single ZIP file",
            key="download_all_button"
        )
        
        # Calculate and display file size
        zip_size_mb = len(zip_data) / (1024 * 1024)
        st.info(f"📋 The ZIP file contains {pdf_count} redacted PDF(s) and {audit_count} audit log(s) ({zip_size_mb:.1f} MB)")

elif not uploaded_files:
    st.info("Upload PDF files and click '🔄 Start New Processing' to begin.")

# --- File Browser Section ---
st.markdown("---")
st.header("📁 File Browser")
st.markdown("Browse and download previously processed files:")

# Get all processed files from database
all_files = db.get_all_processed_files()

if all_files:
    # Show file statistics
    stats = db.get_file_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files Processed", stats['total_files'])
    with col2:
        st.metric("Total Redactions", stats['total_redactions'])
    with col3:
        st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")
    
    # File browser
    st.subheader("📋 Recent Files")
    for file_info in all_files[:10]:  # Show last 10 files
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        
        with col1:
            st.write(f"**{file_info['original_filename']}**")
            st.caption(f"Processed: {file_info['processing_date']}")
        
        with col2:
            st.write(f"{file_info['total_redactions']} redactions")
        
        with col3:
            file_size_mb = file_info['file_size_bytes'] / (1024 * 1024)
            st.write(f"{file_size_mb:.1f} MB")
        
        with col4:
            if os.path.exists(file_info['redacted_filename']):
                with open(file_info['redacted_filename'], 'rb') as f:
                    st.download_button(
                        label="📄 Download",
                        data=f.read(),
                        file_name=os.path.basename(file_info['redacted_filename']),
                        mime="application/pdf",
                        key=f"browser_pdf_{file_info['id']}"
                    )
else:
    st.info("No files have been processed yet.")

st.markdown("---")
st.caption("Internal use only - CurrahTech") 