"""
Stage1 新版全量验证脚本。
用法：python scripts/test_stage1_new.py
"""
import os
import re
import sys
import cv2

# 确保 src 包可被找到
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage1_vision import Stage1Detector
from src.config import STANDARDS_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "text1")
OUT_DIR  = os.path.join(BASE_DIR, "text2", "stage1_new")
os.makedirs(OUT_DIR, exist_ok=True)

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    files = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith(VALID_EXTS))
    if not files:
        print(f"[错误] text1 文件夹中无图片: {SRC_DIR}")
        return

    detectors: dict[str, Stage1Detector] = {}
    rows: list[tuple] = []

    for fn in files:
        stem  = os.path.splitext(fn)[0]
        m     = re.search(r"(front|back|left|right|top)$", stem, re.I)
        angle = m.group(1).lower() if m else "front"

        ref_dir = os.path.join(STANDARDS_DIR, angle)
        if angle not in detectors:
            detectors[angle] = Stage1Detector(ref_dir)

        img_path = os.path.join(SRC_DIR, fn)
        res      = detectors[angle].inspect_with_localization(img_path)

        # 保存标注图
        ann = res.get("annotated_image")
        if ann is not None:
            out_ann = os.path.join(OUT_DIR, stem + "_result.png")
            cv2.imencode(".png", ann)[1].tofile(out_ann)

        # 保存白底抠图
        cut = res.get("cutout_image")
        if cut is not None:
            out_cut = os.path.join(OUT_DIR, stem + "_cutout.png")
            cv2.imencode(".png", cut)[1].tofile(out_cut)

        row = (
            fn,
            angle,
            res["status"],
            f"{res['film_coverage']*100:.1f}%",
            f"{res['similarity']:.2f}",
            len(res["missing_regions"]),
            len(res["extra_regions"]),
            "; ".join(res["issues"]) or "OK",
        )
        rows.append(row)
        print(" | ".join(map(str, row)))

    # 写报告
    report_path = os.path.join(OUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("文件名|视角|状态|覆膜率|相似度|缺件框数|多件框数|问题\n")
        for r in rows:
            f.write("|".join(map(str, r)) + "\n")

    passed = sum(1 for r in rows if r[2] == "PASS")
    failed = sum(1 for r in rows if r[2] == "FAIL")
    retake = sum(1 for r in rows if r[2] == "RETAKE")

    print(f"\n{'='*60}")
    print(f"共 {len(rows)} 张  |  PASS={passed}  FAIL={failed}  RETAKE={retake}")
    print(f"结果图输出: {OUT_DIR}")
    print(f"报告文件:   {report_path}")


if __name__ == "__main__":
    main()
