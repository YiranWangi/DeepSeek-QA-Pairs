"""Regression tests use invented source text and never call an API."""

import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoqa import extract_pdf, generate, inspect_output, make_chunks, parse_response, request_response, validate_pairs

PAGE_ONE = "The study measured blood pressure at two visits."
PAGE_TWO = "The sample included twelve volunteers."
SOURCE = {
    "schema_version": 1,
    "document": {"file_name": "invented_source.txt",
                 "sha256": hashlib.sha256((PAGE_ONE + "\n" + PAGE_TWO + "\n").encode()).hexdigest(),
                 "page_count": 2, "extraction_method": "invented software-test fixture"},
    "pages": [{"page": 1, "text": PAGE_ONE}, {"page": 2, "text": PAGE_TWO}],
}
PAIR = {"question": "How many visits were recorded?", "answer": "Two visits.", "evidence": PAGE_ONE}


class ChunkTests(unittest.TestCase):
    def test_page_boundaries_survive_chunking(self):
        chunks = make_chunks(SOURCE, 2000)
        self.assertEqual([chunk["page"] for chunk in chunks], [1, 2])
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], ["chunk_0001", "chunk_0002"])
        self.assertEqual(chunks[0]["text"], PAGE_ONE)

    def test_chunks_retain_all_normalized_source_words(self):
        chunks = make_chunks(SOURCE, 24)
        for page in (1, 2):
            joined = " ".join(chunk["text"] for chunk in chunks if chunk["page"] == page)
            self.assertEqual(joined, SOURCE["pages"][page - 1]["text"])
        self.assertTrue(all(len(chunk["text"]) <= 24 for chunk in chunks))

    def test_blank_page_is_not_assigned_to_another_page(self):
        source = copy.deepcopy(SOURCE)
        source["pages"][0]["text"] = "  "
        self.assertEqual([chunk["page"] for chunk in make_chunks(source)], [2])

    def test_invalid_source_fails(self):
        for source in ([], {"document": []}, {"document": {}, "pages": []}):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    make_chunks(source)

    def test_duplicate_page_number_fails(self):
        source = copy.deepcopy(SOURCE)
        source["pages"][1]["page"] = 1
        with self.assertRaisesRegex(ValueError, "unique"):
            make_chunks(source)


class ResponseTests(unittest.TestCase):
    def test_json_and_complete_fence_parse(self):
        content = json.dumps({"qa_pairs": [PAIR]})
        self.assertEqual(parse_response(content), [PAIR])
        self.assertEqual(parse_response("```json\n" + content + "\n```"), [PAIR])

    def test_empty_truncated_and_wrong_schema_fail(self):
        for content in ("", '{"qa_pairs": [', "[]", '{"qa_pairs": {}}',
                        '{"qa_pairs": [], "extra": true}'):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    parse_response(content)

    def test_evidence_must_be_in_the_current_page_chunk(self):
        chunks = make_chunks(SOURCE)
        accepted, rejected = validate_pairs([PAIR], chunks[1], SOURCE["document"], set())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "evidence_not_in_source")

    def test_nonstring_and_empty_fields_are_rejected(self):
        for value in (" ", None, 3):
            with self.subTest(value=value):
                pair = {**PAIR, "answer": value}
                accepted, rejected = validate_pairs(
                    [pair], make_chunks(SOURCE)[0], SOURCE["document"], set())
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"], "empty_or_nonstring_field")

    def test_duplicates_use_unicode_whitespace_and_case_normalization(self):
        seen = set()
        chunk = make_chunks(SOURCE)[0]
        first, _ = validate_pairs([PAIR], chunk, SOURCE["document"], seen)
        duplicate = {**PAIR, "question": "  Ｈow  many visits were recorded? "}
        second, rejected = validate_pairs([duplicate], chunk, SOURCE["document"], seen)
        self.assertEqual(len(first), 1)
        self.assertFalse(second)
        self.assertEqual(rejected[0]["reason"], "duplicate_question")

    def test_generated_pair_gets_provenance_and_unreviewed_state(self):
        accepted, rejected = validate_pairs(
            [PAIR], make_chunks(SOURCE)[0], SOURCE["document"], set())
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["source"]["page"], 1)
        self.assertEqual(accepted[0]["source"]["document_sha256"], SOURCE["document"]["sha256"])
        self.assertEqual(accepted[0]["review_status"], "unreviewed")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.responses = self.folder / "responses"
        self.responses.mkdir()
        for index, pair in enumerate((PAIR, {"question": "How many volunteers?",
                                           "answer": "Twelve.", "evidence": PAGE_TWO}), 1):
            (self.responses / f"chunk_{index:04d}.json").write_text(
                json.dumps({"qa_pairs": [pair]}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_run_never_calls_api(self):
        with patch("autoqa.request_response", side_effect=AssertionError("API called")):
            report = generate(SOURCE, self.folder / "run.json", self.responses)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["accepted_pairs"], 2)
        self.assertEqual(inspect_output(report)["invalid_pairs"], 0)

    def test_missing_cached_chunk_is_visible_partial_result(self):
        (self.responses / "chunk_0002.json").unlink()
        report = generate(SOURCE, self.folder / "run.json", self.responses)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["failed_chunks"], 1)
        self.assertEqual(report["summary"]["accepted_pairs"], 1)

    def test_invalid_pair_makes_run_partial(self):
        path = self.responses / "chunk_0002.json"
        path.write_text(json.dumps({"qa_pairs": [{**PAIR, "evidence": "Invented quote"}]}))
        report = generate(SOURCE, self.folder / "run.json", self.responses)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["rejected_pairs"], 1)

    def test_output_is_not_silently_overwritten(self):
        output = self.folder / "run.json"
        output.write_text("existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            generate(SOURCE, output, self.responses)
        self.assertEqual(output.read_text(), "existing")

    def test_live_run_requires_key_and_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                generate(SOURCE, self.folder / "run.json", model="placeholder")

    def test_wrong_page_provenance_is_detected(self):
        report = generate(SOURCE, self.folder / "run.json", self.responses)
        report["qa_pairs"][0]["source"]["page"] = 2
        self.assertEqual(inspect_output(report)["invalid_pairs"], 1)

    def test_legacy_pairs_are_not_given_invented_sources(self):
        report = inspect_output([{"question": "Question?", "answer": "Answer."}])
        self.assertEqual(report["source_tracking"], "unavailable")
        self.assertEqual(report["clinical_review"], "not_verified")


class RequestTests(unittest.TestCase):
    def test_json_mode_request_and_completed_response(self):
        content = json.dumps({"qa_pairs": [PAIR]})
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        with patch("autoqa.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as opened:
            self.assertEqual(request_response(make_chunks(SOURCE)[0], "dummy-key", "dummy-model", 0.3, 500), content)
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["model"], "dummy-model")
        self.assertIn(PAGE_ONE, body["messages"][1]["content"])

    def test_incomplete_api_response_is_rejected(self):
        response = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
        with patch("autoqa.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                request_response(make_chunks(SOURCE)[0], "dummy-key", "dummy-model", 0.3, 1)


@unittest.skipUnless(all(shutil.which(tool) for tool in ("pdfinfo", "pdftoppm", "tesseract")),
                     "Poppler and Tesseract are required for the OCR integration test")
class OCRTests(unittest.TestCase):
    def test_two_pdf_pages_ocr_and_offline_generation(self):
        # A minimal two-page PDF made solely from invented test sentences.
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        ]
        for text, content_id in ((PAGE_ONE, 4), (PAGE_TWO, 6)):
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 7 0 R >> >> /Contents {content_id} 0 R >>".encode()
            )
            stream = f"BT /F1 16 Tf 54 700 Td ({text}) Tj ET".encode()
            objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
        data = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, item in enumerate(objects, 1):
            offsets.append(len(data))
            data.extend(f"{number} 0 obj\n".encode() + item + b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
        for offset in offsets:
            data.extend(f"{offset:010d} 00000 n \n".encode())
        data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "invented.pdf"
            pdf.write_bytes(data)
            source = extract_pdf(pdf, dpi=300)
            self.assertEqual(source["document"]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual([record["page"] for record in source["pages"]], [1, 2])
            # OCR spelling and spacing are not guaranteed; provenance follows
            # the saved OCR text and retains the PDF page for manual review.
            self.assertIn("pressure", source["pages"][0]["text"].lower())
            self.assertIn("volunteers", source["pages"][1]["text"].lower())
            responses = root / "responses"
            responses.mkdir()
            for index, pair in enumerate((PAIR, {"question": "How many volunteers?",
                                               "answer": "Twelve.", "evidence": PAGE_TWO}), 1):
                evidence = source["pages"][index - 1]["text"].strip()
                (responses / f"chunk_{index:04d}.json").write_text(
                    json.dumps({"qa_pairs": [{**pair, "evidence": evidence}]}))
            with patch("autoqa.request_response", side_effect=AssertionError("API called")):
                result = generate(source, root / "run.json", responses)
            self.assertEqual(result["status"], "complete")
            self.assertEqual([pair["source"]["page"] for pair in result["qa_pairs"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
