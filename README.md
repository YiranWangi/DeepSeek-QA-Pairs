# Medical PDF QA extraction

A small pipeline for turning scanned PDF pages into question-answer drafts with Tesseract OCR and the DeepSeek API.

The pipeline keeps page boundaries, generates questions from one source chunk at a time, and attaches an original-text excerpt to each accepted pair. These are **drafts for manual review**: matching an excerpt does not establish that the answer is medically correct or fully supported.

## Workflow

1. OCR each page locally and save its text, page number and the PDF's SHA-256 hash.
2. Split text within each page into bounded chunks.
3. Request a JSON object containing question, answer and evidence fields.
4. Reject malformed or empty fields, duplicate questions and evidence absent from the source chunk.
5. Save accepted drafts, raw responses and a report of failed chunks or rejected items.

A run with failures or rejected items is marked `partial` and exits with a nonzero status. Each accepted draft retains its source page, chunk ID, document hash and `unreviewed` state. Page numbers refer to the PDF page order, starting at 1.

## Run locally

Python 3.10+ is sufficient for the Python code; no Python packages are required. PDF extraction needs **Tesseract** (including the requested language) and **Poppler** on PATH.

On macOS:

```bash
brew install tesseract poppler
python check_environment.py
```

### Extract pages without an API call

```bash
python autoqa.py extract "MSD diabetes.pdf" --output runs/diabetes_pages.json
```

Use `--language` for another installed Tesseract language and `--dpi` to adjust image resolution. Blank pages are retained in the source file and skipped during generation.

### Generate QA drafts

Set your key in the `DEEPSEEK_API_KEY` environment variable, then pass a model ID supported by your account:

```bash
python autoqa.py generate runs/diabetes_pages.json \
  --model YOUR_MODEL_ID --output runs/diabetes_drafts.json
```

This step sends OCR chunks to DeepSeek and uses the paid API. The key is read from the environment. The request uses [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/) and checks for a completed response. Choose a new output filename for each run.

### Try the offline example

The example uses invented study notes and two manually written responses. It checks the pipeline without calling a model or demonstrating model performance.

```bash
python autoqa.py generate examples/pages.json \
  --responses_dir examples/responses --output runs/offline_demo.json
python autoqa.py validate runs/offline_demo.json
python -m unittest discover -s tests -v
```

The offline mode expects `chunk_0001.json`, `chunk_0002.json`, etc. in the response directory. Each file contains the response content as `{"qa_pairs": [...]}`; use the same source text and chunk size that produced those responses. See [examples/sample_run.json](examples/sample_run.json) for a complete saved example.

## Existing sample

`MSD diabetes.pdf` is the original example input. `diabetes_qa_pairs.json` contains **207 historical question-answer pairs**, without page references or evidence excerpts. Their source links and clinical correctness have not been verified; this update does not add guessed citations to them.

```bash
python autoqa.py validate diabetes_qa_pairs.json
```

The historical array remains unchanged. New runs use an object with source metadata, `qa_pairs`, `source_chunks`, `chunk_results` and a summary.

## Code and limitations

`autoqa.py` contains extraction, generation and validation. The original Chinese script name forwards to the same CLI, and `test tessarcat.py` forwards to the environment check.

OCR can misread units and terminology. Page-local chunks may lose surrounding context. Evidence matching checks that a quoted excerpt exists; a reviewer still needs to check the PDF and decide whether it supports the answer. No clinical accuracy benchmark is reported.

Tests cover page tracking, schema errors, duplicate questions, unsupported excerpts, partial runs and offline replay. No paid API request is made by the tests.
