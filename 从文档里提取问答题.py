import os
import requests
import json
import time
import re
from typing import List, Dict
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from tqdm import tqdm

# 设置 Tesseract 和 Poppler 路径
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
POPLER_PATH = "/opt/homebrew/bin"

# DeepSeek API 配置
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class PDFToQAGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"📷 Using OCR to extract text from {pdf_path} (English)...")
        text = ""
        try:
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPLER_PATH)
            for i, page_image in enumerate(pages):
                page_text = pytesseract.image_to_string(page_image, lang='eng')
                print(f"🧾 Page {i+1} OCR Preview: {page_text[:150].replace(chr(10), ' ')}...\n")
                text += page_text + "\n"
            return text
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            raise

    def build_prompt(self, chunk: str) -> str:
        return (
            "Please generate diverse, medically accurate question-answer pairs from the following text. "
            "Cover areas such as symptoms, diagnosis, treatment, causes, etc. "
            "Return only a JSON array like this (no Markdown or code fences):\n"
            "[{\"question\": \"...\", \"answer\": \"...\"}, ...]\n\n"
            f"Text:\n{chunk}\n"
            "Please generate at least 5 question-answer pairs:"
        )

    def generate_qa_pairs(self, text: str, output_path: str) -> List[Dict]:
        print("🧠 Generating as many QA pairs as possible from the document...")
        chunks = self._chunk_text(text, chunk_size=2000)
        qa_pairs = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks"), 1):
            print(f"→ Chunk {i} Preview: {chunk[:200].replace(chr(10), ' ')}...\n")

            prompt = self.build_prompt(chunk)

            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000
            }

            try:
                response = requests.post(
                    DEEPSEEK_API_URL,
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()

                try:
                    content = response.json()["choices"][0]["message"]["content"]
                    print(f"📨 Model response preview: {content[:300]}...\n")
                    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
                    if match:
                        content = match.group(1)
                    else:
                        print("⚠️ No Markdown wrapper. Trying full content as JSON...")
                    generated = json.loads(content)
                    if isinstance(generated, list):
                        qa_pairs.extend(generated)
                        print(f"✅ Chunk {i} generated {len(generated)} QAs, total so far: {len(qa_pairs)}")
                        self.save_to_file(qa_pairs, output_path)
                except Exception as e:
                    print("⚠️ Failed to parse JSON.")
                    print("Raw content:", content[:500])
                    print("Raw response:", response.text)
                    continue

                time.sleep(1)

            except Exception as e:
                print(f"⚠️ Chunk {i} failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(e.response.text)
                continue

        return qa_pairs

    def _chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def save_to_file(self, data: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(data)} QA pairs to {output_path}")

def main():
    PDF_PATH = "MSD diabetes.pdf"
    OUTPUT_QA_PATH = "diabetes_qa_pairs.json"

    generator = PDFToQAGenerator(DEEPSEEK_API_KEY)

    try:
        text = generator.extract_text_from_pdf(PDF_PATH)
        qa_pairs = generator.generate_qa_pairs(text, output_path=OUTPUT_QA_PATH)

        print("\n✅ Completed!")
        print(f"- Total QAs generated: {len(qa_pairs)}")
        print(f"- Output file: {OUTPUT_QA_PATH}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()