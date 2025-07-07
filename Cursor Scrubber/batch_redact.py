#!/usr/bin/env python3
"""
Production-ready batch PDF redaction for automated processing
Optimized for accuracy and reliability across multiple PDFs
"""

import os
import glob
from pdf_redactor import redact_pdf
import time
from datetime import datetime

def batch_redact_pdfs_automated(
    input_folder=".",
    output_folder="redacted_output",
    logo_padding=3,  # Optimized for precision
    file_pattern="*.pdf"
):
    """
    Automated batch redaction with optimized settings for accuracy.
    
    Args:
        input_folder: Folder containing PDFs to process
        output_folder: Folder to save redacted PDFs
        logo_padding: Padding around detected logos (3 = optimal precision)
        file_pattern: File pattern to match (default: *.pdf)
    """
    
    print(f"🚀 Automated Batch PDF Redaction")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print(f"🎯 Logo padding: {logo_padding} points")
    print(f"🔍 File pattern: {file_pattern}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(input_folder, file_pattern))
    
    if not pdf_files:
        print(f"❌ No PDF files found matching pattern '{file_pattern}' in {input_folder}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")
    
    # Process statistics
    total_files = len(pdf_files)
    successful_redactions = 0
    failed_files = []
    processing_times = []
    
    print(f"\n🔄 Starting batch processing...")
    print("-" * 60)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_file)
        start_time = time.time()
        
        print(f"\n[{i}/{total_files}] Processing: {filename}")
        
        try:
            # Process the PDF with optimized settings
            result = redact_pdf(
                pdf_path=pdf_file,
                output_dir=output_folder,
                logo_padding=logo_padding
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if "error" in result:
                print(f"  ❌ Failed: {result['error']}")
                failed_files.append((filename, result['error']))
            else:
                print(f"  ✅ Success: {result['total_redactions']} redactions applied")
                print(f"  📄 Output: {os.path.basename(result['redacted_pdf'])}")
                print(f"  📊 Audit: {os.path.basename(result['audit_log'])}")
                print(f"  ⏱️  Time: {processing_time:.2f}s")
                successful_redactions += 1
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"  ❌ Error: {error_msg}")
            failed_files.append((filename, error_msg))
    
    # Summary report
    print("\n" + "=" * 60)
    print(f"📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"📄 Total files processed: {total_files}")
    print(f"✅ Successful redactions: {successful_redactions}")
    print(f"❌ Failed files: {len(failed_files)}")
    print(f"📈 Success rate: {(successful_redactions/total_files)*100:.1f}%")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        total_time = sum(processing_times)
        print(f"⏱️  Average processing time: {avg_time:.2f}s")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"\n📁 Output folder: {output_folder}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        "total_files": total_files,
        "successful": successful_redactions,
        "failed": len(failed_files),
        "success_rate": (successful_redactions/total_files)*100,
        "failed_files": failed_files
    }

if __name__ == "__main__":
    # Production settings for automated processing
    batch_redact_pdfs_automated(
        input_folder=".",  # Current directory
        output_folder="redacted_output",
        logo_padding=3,  # Optimal precision
        file_pattern="*.pdf"
    )
    
    # Example usage for different scenarios:
    # batch_redact_pdfs_automated(input_folder="engineering_drawings", logo_padding=3)
    # batch_redact_pdfs_automated(input_folder="client_docs", logo_padding=5)  # More conservative 