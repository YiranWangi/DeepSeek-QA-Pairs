# 🧠 AutoQA: Extracting Medical Q&A Pairs from PDFs via OCR + LLM

A Python-based pipeline that converts scanned or image-based medical PDF documents into structured question-answer pairs using OCR (Tesseract) and DeepSeek's LLM API.

## Project Overview

This project is designed to help medical researchers or AI evaluators generate diverse, medically accurate QA datasets directly from PDF files (e.g., MSD Manual entries).

This tool is designed for:

- **Medical researchers**, educators, or AI developers  
- Who want to convert **non-selectable, image-based PDFs** (e.g., MSD Manual scans)  
- Into high-quality, diverse, structured **Q&A datasets**

The workflow includes:

1. **OCR** text extraction using Tesseract  
2. **Text chunking** (to handle long input)  
3. **Prompt construction** with QA instructions  
4. **LLM querying** using the DeepSeek API  
5. **Automatic JSON parsing and saving**

## Example

- **Input PDF**: `MSD diabetes.pdf` (scanned medical document)  
- **Output file**: `diabetes_qa_pairs.json` (list of QA pairs like `{"question": ..., "answer": ...}`)


## Requirements
- Python 3.8+
- Tesseract (macOS: brew install tesseract)
- Poppler (macOS: brew install poppler)

## File Structure
.
├── 从文档里提取问答题.py            # Main script: generates QAs and saves to JSON
├── test tessarcat.py                # Tesseract environment test
├── MSD diabetes.pdf                 # Sample input PDF (scanned)
├── diabetes_qa_pairs.json          # Generated output QA file
├── README.md

## Quick Start

### Run the main QA generation script:
```bash
python 从文档里提取问答题.py
```
### Test if OCR is configured properly:
```bash
python test tessarcat.py
```
## API Configuration
Before running, set your DeepSeek API Key in the script:
```bash
DEEPSEEK_API_KEY = "your_api_key_here"
```

