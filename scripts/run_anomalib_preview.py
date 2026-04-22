import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from anomalib.engine import Engine
from anomalib.models import Padim


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def imread_color_safe(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image_safe(path: Path, image: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def to_u8_mask(mask_obj) -> np.ndarray:
    if mask_obj is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = mask_obj.detach().cpu().numpy() if hasattr(mask_obj, "detach") else np.asarray(mask_obj)
    arr = np.squeeze(arr)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomalib preview and draw anomaly boxes.")
    parser.add_argument("--input-dir", default="text1", help="Folder containing images to inspect.")
    parser.add_argument("--output-dir", default="text2/anomalib_preview", help="Annotated result folder.")
    parser.add_argument(
        "--ckpt",
        default="D:/RiceCooker_Project_ascii/output/anomalib_padim/Padim/ricecooker_stage1/v0/weights/lightning/model.ckpt",
        help="Trained anomalib checkpoint path.",
    )
    parser.add_argument("--ascii-workdir", default="D:/RiceCooker_Project_ascii/infer_batch", help="ASCII temp folder.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of random images to preview.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--min-area", type=int, default=200, help="Min anomaly box area in pixels.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ascii_workdir = Path(args.ascii_workdir).resolve()
    ckpt = Path(args.ckpt).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if ascii_workdir.exists():
        shutil.rmtree(ascii_workdir)
    ascii_workdir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not files:
        raise RuntimeError(f"No valid images found in: {input_dir}")

    random.seed(args.seed)
    picked = random.sample(files, min(args.sample_size, len(files)))

    # Copy images to ASCII path because anomalib currently fails on non-ASCII paths in this environment.
    mapped: list[tuple[Path, Path]] = []
    for idx, src in enumerate(picked):
        dst = ascii_workdir / f"{idx:03d}_{src.name}"
        shutil.copy2(src, dst)
        mapped.append((src, dst))

    engine = Engine(enable_progress_bar=False)
    model = Padim()

    rows: list[str] = []
    for src_path, ascii_path in mapped:
        preds = engine.predict(
            model=model,
            ckpt_path=str(ckpt),
            data_path=str(ascii_path),
            return_predictions=True,
        )
        pred = preds[0]
        pred_score = float(np.squeeze(pred.pred_score.detach().cpu().numpy()))
        pred_label = bool(np.squeeze(pred.pred_label.detach().cpu().numpy()))
        pred_mask = to_u8_mask(pred.pred_mask)

        image = imread_color_safe(src_path)
        if image is None:
            rows.append(f"{src_path.name}|READ_FAIL|score={pred_score:.4f}|boxes=0")
            continue

        if pred_mask.shape[:2] != image.shape[:2]:
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < args.min_area:
                continue
            box_count += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, f"anom {area}px", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        status = "ANOMALY" if (pred_label or box_count > 0) else "NORMAL"
        cv2.putText(
            image,
            f"{status} score={pred_score:.4f} boxes={box_count}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255) if status == "ANOMALY" else (0, 180, 0),
            2,
        )

        out_path = output_dir / f"{src_path.stem}_anomalib_preview{src_path.suffix.lower()}"
        write_image_safe(out_path, image)
        rows.append(f"{src_path.name}|{status}|score={pred_score:.4f}|boxes={box_count}")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"[Done] Saved {len(rows)} preview images to: {output_dir}")
    print(f"[Done] Report: {report_path}")


if __name__ == "__main__":
    main()
