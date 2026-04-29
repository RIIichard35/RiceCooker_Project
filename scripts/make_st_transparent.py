from __future__ import annotations

from pathlib import Path
import io

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


FACES = ["back", "front", "left", "right", "top"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_transparent_png(src_path: Path, out_path: Path, session) -> None:
    with open(src_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data, session=session)
    rgba = Image.open(io.BytesIO(output_data)).convert("RGBA")
    arr = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(out_path), arr)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    standards_root = root / "assets" / "standards"
    st_root = root / "st"

    print("Loading rembg session...")
    session = new_session("isnet-general-use")

    done = 0
    for face in FACES:
        target_dir = st_root / face
        if not target_dir.exists():
            continue

        files = sorted([p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
        print(f"[{face}] converting {len(files)} files")
        for f in files:
            src_original = standards_root / face / f.name
            if not src_original.exists():
                print(f"  skip missing source: {src_original}")
                continue
            make_transparent_png(src_original, f, session)
            done += 1

    print(f"Done. Converted {done} files to transparent background.")


if __name__ == "__main__":
    main()
