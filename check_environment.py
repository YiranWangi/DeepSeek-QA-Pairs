"""Check the command-line tools used by local PDF OCR."""

import shutil
import subprocess


def main():
    missing = []
    for name in ("pdfinfo", "pdftoppm", "tesseract"):
        path = shutil.which(name)
        if not path:
            missing.append(name)
            print(f"{name}: missing")
            continue
        flag = "--version" if name == "tesseract" else "-v"
        result = subprocess.run([path, flag], capture_output=True, text=True, timeout=10)
        print(f"{name}: {path}")
        print((result.stdout or result.stderr).splitlines()[0])
    if not missing:
        languages = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, check=True, timeout=10
        )
        print(languages.stdout.strip())
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
