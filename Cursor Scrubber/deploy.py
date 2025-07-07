#!/usr/bin/env python3
"""
Deployment script for PDF Redactor Tool
Sets up the application for web hosting
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_deployment():
    """Setup the application for deployment"""
    print("🚀 Setting up PDF Redactor Tool for deployment...")
    
    # Create necessary directories
    directories = [
        "redacted_output",
        "templates", 
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Initialize database
    try:
        from database import db
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import streamlit
        import cv2
        import fitz
        import PIL
        import numpy
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n🎉 Deployment setup complete!")
    print("\nTo run the application:")
    print("1. Local development: streamlit run app.py")
    print("2. Production: streamlit run app.py --server.port 8501 --server.address 0.0.0.0")
    
    return True

def run_app():
    """Run the Streamlit app"""
    print("🌐 Starting PDF Redactor Tool...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_app()
    else:
        setup_deployment() 