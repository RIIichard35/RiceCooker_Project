"""
批量测试文件夹内合格品照片，流程与 gui/app1 一致（Stage1 + Anomalib）。
默认目录 D:\\数据集\\tt；输出 result_<原文件名>.png 到同目录。

视角从文件名推断：stem 以 front|back|left|right|top 结尾；否则使用 --default-view。

用法:
  python scripts/batch_test_tt_folder.py
  python scripts/batch_test_tt_folder.py --dir "D:\\数据集\\tt" --default-view back
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANOMALIB_HOLE_FAIL_THRESH,
    ANOMALIB_MODEL_DIR,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    STANDARDS_DIR,
    get_anomalib_threshold,
)
from src.stage1_vision import Stage1Detector
from src.stage2_anomalib import AnomalibDetector

VIEW_SUFFIXES = ("front", "back", "left", "right", "top")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_view(stem: str, default: str) -> str:
    s = stem.lower()
    for v in VIEW_SUFFIXES:
        if s.endswith(v):
            return v
    return default


def load_st_rgba(ref_path: str | None) -> np.ndarray | None:
    if not ref_path or not Path(ref_path).is_file():
        return None
    try:
        arr = np.fromfile(ref_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def load_bgr(path: Path) -> np.ndarray | None:
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def as_rgb3(img: np.ndarray) -> np.ndarray:
    """统一为 RGB H×W×3（st 参考图为 RGBA 时需转换）。"""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"unsupported image shape {img.shape}")


def render_detection_panel(
    anomalib_result: dict | None,
    ann_img: np.ndarray | None,
) -> np.ndarray | None:
    if anomalib_result is None:
        return ann_img if ann_img is not None else None
    _clean = anomalib_result.get("clean_cutout")
    if _clean is None or not _clean.any():
        return ann_img
    if _clean.ndim == 3 and _clean.shape[2] == 4:
        _a = _clean[:, :, 3:4].astype(np.float32) / 255.0
        _bgr_base = _clean[:, :, :3].astype(np.float32)
        _bg = np.full_like(_bgr_base, 128.0)
        final_ann = (_a * _bgr_base + (1 - _a) * _bg).astype(np.uint8)
    else:
        final_ann = _clean[:, :, :3].copy()
    _h, _w = final_ann.shape[:2]
    a_boxes = anomalib_result.get("anomaly_boxes", []) or []
    h_boxes = anomalib_result.get("hole_boxes", []) or []
    overlay_layer = final_ann.copy()
    all_defect_boxes = list(a_boxes)
    for hb in h_boxes:
        if not any(
            hb[0] < eb[2] and hb[2] > eb[0] and hb[1] < eb[3] and hb[3] > eb[1]
            for eb in all_defect_boxes
        ):
            all_defect_boxes.append(hb)
    for box in all_defect_boxes:
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(overlay_layer, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
    final_ann = cv2.addWeighted(overlay_layer, 0.38, final_ann, 0.62, 0)
    for idx, box in enumerate(all_defect_boxes):
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        is_hole = any(hb[0] == box[0] and hb[1] == box[1] for hb in h_boxes)
        label = "hole" if is_hole else f"a{idx+1}"
        color = (0, 0, 230) if is_hole else (30, 30, 220)
        cv2.rectangle(final_ann, (bx1, by1), (bx2, by2), color, 2)
        txt_y = max(by1 - 4, 18)
        cv2.putText(
            final_ann, label, (bx1 + 2, txt_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
    cv2.rectangle(final_ann, (2, 2), (_w - 3, _h - 3), (0, 215, 255), 3)
    a_status = anomalib_result.get("status", "?")
    a_score = float(anomalib_result.get("patchcore_score", anomalib_result.get("anomaly_score", 0.0)))
    h_score = float(anomalib_result.get("hole_score", 0.0))
    tpl_score = float(anomalib_result.get("template_score", 0.0))
    status_txt = f"{a_status} pc:{a_score:.3f} tpl:{tpl_score:.3f} hole:{h_score*100:.1f}%"
    cv2.putText(
        final_ann, status_txt, (8, max(20, _h - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0) if a_status == "PASS" else (0, 0, 220), 2,
    )
    return final_ann


def _dbg_agent(
    hypothesis_id: str,
    message: str,
    data: dict,
) -> None:
    # #region agent log
    try:
        log_path = PROJECT_ROOT / "debug-13fa83.log"
        rec = {
            "sessionId": "13fa83",
            "runId": "batch_tt",
            "hypothesisId": hypothesis_id,
            "location": "batch_test_tt_folder.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as df:
            df.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def hstack_same_height(images: list[np.ndarray], bg_gray: int = 40) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    H = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != H:
            sc = H / im.shape[0]
            W = max(1, int(im.shape[1] * sc))
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        resized.append(im)
    return np.hstack(resized)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=r"D:\数据集\tt")
    ap.add_argument("--default-view", type=str, default="front", choices=list(VIEW_SUFFIXES))
    ap.add_argument("--anomalib-threshold", type=float, default=None,
                    help="覆盖 ANOMALIB_THRESH_BY_VIEW；默认每视角用 config")
    ap.add_argument("--max-files", type=int, default=0, help="仅处理前 N 张，0=全部")
    args = ap.parse_args()

    tt = Path(args.dir)
    tt.mkdir(parents=True, exist_ok=True)
    std_root = Path(STANDARDS_DIR)

    files = sorted(
        p for p in tt.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith("result_")
    )
    if not files:
        print(f"[batch_test_tt] 目录无待测图片: {tt}")
        return

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    print(f"[batch_test_tt] 共 {len(files)} 张，输出前缀 result_", flush=True)

    _dbg_agent(
        "H0",
        "batch_start",
        {
            "n": len(files),
            "hole_fail_thresh": float(ANOMALIB_HOLE_FAIL_THRESH),
            "thr_override": args.anomalib_threshold,
        },
    )
    csv_rows: list[dict[str, object]] = []

    s1_cache: dict[str, Stage1Detector] = {}
    ano_cache: dict[str, AnomalibDetector] = {}

    for fp in files:
        view = infer_view(fp.stem, args.default_view)
        ref_dir = std_root / view
        if not ref_dir.is_dir() or not any(ref_dir.iterdir()):
            print(f"  [跳过] {fp.name} 视角={view} 但无标准库 {ref_dir}")
            continue

        img_bgr = load_bgr(fp)
        if img_bgr is None:
            print(f"  [跳过] 无法解码 {fp.name}")
            continue

        if view not in s1_cache:
            s1_cache[view] = Stage1Detector(str(ref_dir))
        det = s1_cache[view]
        if not det.reference_data:
            print(f"  [跳过] Stage1 未加载到参考图: {view}")
            continue

        result = det.inspect_with_localization(
            img_bgr,
            min_match_count=MIN_MATCH_COUNT,
            missing_regions_fail_count=MAX_MISSING_COUNT,
        )

        thr_eff_ann = float(
            args.anomalib_threshold
            if args.anomalib_threshold is not None
            else get_anomalib_threshold(view)
        )
        ar: dict | None = None
        if result.get("status") in ("PASS", "FAIL"):
            if view not in ano_cache:
                ano_cache[view] = AnomalibDetector(
                    view=view,
                    model_dir=ANOMALIB_MODEL_DIR,
                    threshold=args.anomalib_threshold,
                )
            ad = ano_cache[view]
            ad.threshold = thr_eff_ann
            if ad.is_ready:
                cutout = result.get("cutout_rgba")
                mask_crop = result.get("product_mask_crop")
                _pb = result.get("product_bbox")
                if cutout is not None and _pb is not None:
                    px1, py1, px2, py2 = _pb
                    if px2 > px1 and py2 > py1:
                        cutout = cutout[py1:py2, px1:px2]
                if cutout is not None and mask_crop is not None:
                    if mask_crop.shape[:2] != cutout.shape[:2]:
                        mask_crop = cv2.resize(
                            mask_crop,
                            (cutout.shape[1], cutout.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                if cutout is not None:
                    ar = ad.inspect(cutout, stage1_fg_mask=mask_crop)
                    result["anomalib_result"] = ar
                    if ar.get("status") == "FAIL":
                        result["status"] = "FAIL"
                        result["issues"] = result.get("issues", []) + ar.get("issues", [])

        # #region agent log
        ar_dict = result.get("anomalib_result") or {}
        pc = float(ar_dict.get("patchcore_score", 0.0))
        tpl = float(ar_dict.get("template_score", 0.0))
        hs = float(ar_dict.get("hole_score", 0.0))
        pc_fail = pc > thr_eff_ann
        hole_fail = hs > float(ANOMALIB_HOLE_FAIL_THRESH)
        _issues = result.get("issues") or []
        _dbg_agent(
            "H1",
            "per_file",
            {
                "file": fp.name,
                "view": view,
                "patchcore": pc,
                "template": tpl,
                "hole": hs,
                "thr_pc": thr_eff_ann,
                "hole_thresh": float(ANOMALIB_HOLE_FAIL_THRESH),
                "pc_fail": pc_fail,
                "hole_fail": hole_fail,
                "status": result.get("status"),
                "seg": result.get("segmentation_backend"),
                "issues_short": "; ".join(str(x) for x in _issues)[:240],
            },
        )
        # #endregion
        csv_rows.append(
            {
                "filename": fp.name,
                "view": view,
                "patchcore": f"{pc:.4f}",
                "template": f"{tpl:.4f}",
                "hole_score": f"{hs:.5f}",
                "thr_patchcore": f"{thr_eff_ann:.4f}",
                "thr_hole": f"{ANOMALIB_HOLE_FAIL_THRESH:.5f}",
                "pc_fail": int(pc_fail),
                "hole_fail": int(hole_fail),
                "status": str(result.get("status")),
                "seg": str(result.get("segmentation_backend", "")),
                "issues": " | ".join(str(x) for x in _issues)[:500],
            }
        )

        ref_rgba = load_st_rgba(result.get("ref_cutout_path"))
        cut_rgba = result.get("cutout_rgba")
        _pb = result.get("product_bbox")

        panel_orig = to_rgb(img_bgr)
        if ref_rgba is not None:
            panel_ref = ref_rgba
        elif cut_rgba is not None and _pb is not None:
            x1, y1, x2, y2 = _pb
            if x2 > x1 and y2 > y1:
                cr = cut_rgba[y1:y2, x1:x2]
                panel_ref = to_rgb(cr)
            else:
                panel_ref = to_rgb(cut_rgba)
        else:
            panel_ref = np.full((panel_orig.shape[0], panel_orig.shape[1], 3), 200, dtype=np.uint8)

        panel_det = render_detection_panel(result.get("anomalib_result"), result.get("annotated_image"))
        if panel_det is None:
            panel_det_rgb = np.full(panel_orig.shape, 220, dtype=np.uint8)
        else:
            panel_det_rgb = cv2.cvtColor(panel_det, cv2.COLOR_BGR2RGB)

        try:
            row = hstack_same_height(
                [
                    as_rgb3(panel_orig),
                    as_rgb3(panel_ref),
                    panel_det_rgb,
                ]
            )
        except Exception as e:
            print(f"  [错误] 拼图失败 {fp.name}: {e}")
            continue

        strip_h = 36
        banner = np.full((strip_h, row.shape[1], 3), 45, dtype=np.uint8)
        ar = result.get("anomalib_result") or {}
        a_st = ar.get("status", "—")
        line = (
            f"{fp.name} | view={view} | ST1={result.get('status')} | Anom={a_st} | "
            f"seg={result.get('segmentation_backend', '?')}"
        )
        cv2.putText(banner, line[:160], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

        out = np.vstack([banner, row])
        out_path = tt / f"result_{fp.stem}.png"
        ok, enc = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if ok:
            enc.tofile(str(out_path))
            print(f"  [OK] {out_path.name}  top_status={result.get('status')}")
        else:
            print(f"  [错误] 无法编码 {fp.name}")

    csv_path = tt / "batch_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
            w = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[batch_test_tt] 汇总 CSV: {csv_path}", flush=True)
    main()
