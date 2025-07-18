from setuptools import setup, find_packages

setup(
    name="pdf-scrubber-tool",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "PyMuPDF==1.23.8",
        "opencv-python-headless>=4.8.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "regex>=2023.8.8",
        "python-docx>=0.8.11",
        "pdfplumber>=0.10.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    python_requires=">=3.8",
) 