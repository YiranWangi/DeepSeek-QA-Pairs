# AutoQA: Extracting Medical Q&A Pairs from PDF Using OCR and DeepSeek

A Python-based pipeline that converts scanned or image-based medical PDF documents into structured question-answer pairs using OCR (Tesseract) and DeepSeek's LLM API.

## Project Overview

This project is designed to help medical researchers or AI evaluators generate diverse, medically accurate QA datasets directly from PDF files (e.g., MSD Manual entries). The process includes:

OCR extraction using Tesseract
Text chunking
Prompt construction
LLM querying via DeepSeek API
Parsing and saving QA pairs in JSON format

Example input file: MSD diabetes.pdf
Example output: diabetes_qa_pairs.json

## Requirements
Python 3.8+
Tesseract (macOS: brew install tesseract)
Poppler (macOS: brew install poppler)

## File Structure
.
├── 从文档里提取问答题.py            # Main script: generates QAs and saves to JSON
├── test tessarcat.py                # Tesseract environment test
├── MSD diabetes.pdf                 # Sample input PDF (scanned)
├── diabetes_qa_pairs.json          # Generated output QA file

## Run the script and test OCR
```bash
# train
python 从文档里提取问答题.py
# test
python test tessarcat.py
```

