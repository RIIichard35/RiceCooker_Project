from __future__ import annotations

from pathlib import Path
import io

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


FACES = ["right", "back", "front", "left", "top"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cutout_to_white_bg(image_path: Path, out_path: Path, session) -> None:
    with open(image_path, "rb") as f:
        input_data = f.read()

    output_data = remove(
        input_data,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
    )

    img_rgba = Image.open(io.BytesIO(output_data)).convert("RGBA")
    img_cv_rgba = cv2.cvtColor(np.array(img_rgba), cv2.COLOR_RGBA2BGRA)

    b, g, r, a = cv2.split(img_cv_rgba)
    img_rgb = cv2.merge((b, g, r))

    white_bg = np.full_like(img_rgb, 255)
    alpha_mask = a.astype(np.float32) / 255.0

    for c in range(3):
        white_bg[:, :, c] = (
            alpha_mask * img_rgb[:, :, c] + (1.0 - alpha_mask) * white_bg[:, :, c]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), white_bg)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_root = root / "assets" / "standards"
    dst_root = root / "st"

    print("Loading rembg model session...")
    session = new_session("isnet-general-use")

    total = 0
    for face in FACES:
        src_dir = src_root / face
        dst_dir = dst_root / face
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
        print(f"[{face}] found {len(images)} images")

        for img in images:
            out_path = dst_dir / img.name
            cutout_to_white_bg(img, out_path, session)
            total += 1

        print(f"[{face}] done -> {dst_dir}")

    print(f"All done. Total processed: {total}")


if __name__ == "__main__":
    main()
