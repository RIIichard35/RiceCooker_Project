"""One-off: extract plain text from a .docx for assistant context."""
import re
import sys
import zipfile
from pathlib import Path


def main() -> None:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\数据集\lunwen.docx")
    if not p.exists():
        print("FILE_MISSING:", p)
        sys.exit(1)
    with zipfile.ZipFile(p) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines[:120]:
        print(ln[:300])


if __name__ == "__main__":
    main()
