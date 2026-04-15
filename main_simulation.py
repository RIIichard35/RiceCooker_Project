import csv
import os
import time
from datetime import datetime

import cv2

from verify_stage1 import run_stage1_localized
from verify_stage2 import Stage2Detector


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
STAGE2_CONF_THRESHOLD = 0.25


def determine_ref_model(filename):
    name_lower = filename.lower()
    if "front" in name_lower:
        return "front"
    if "top" in name_lower:
        return "top"
    if "right" in name_lower:
        return "right"
    if "left" in name_lower:
        return "left"
    if "back" in name_lower:
        return "back"
    return None


def format_bbox(bbox):
    if not bbox:
        return ""
    return ",".join(str(int(v)) for v in bbox)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "assets", "images_test")
    model_path = os.path.join(base_dir, "assets", "models", "best20240919.pt")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "runs", f"batch_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    report_csv_path = os.path.join(output_dir, "inspection_report.csv")

    print("=" * 60)
    print(f"Starting cascade inspection batch: {current_time}")
    print("=" * 60)

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    image_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(VALID_EXTENSIONS))
    if not image_files:
        print(f"No test images found in: {test_dir}")
        return

    try:
        yolo_detector = Stage2Detector(model_path)
    except Exception as exc:
        print(f"Fatal model load error: {exc}")
        return

    report_data = []
    stats = {
        "total": len(image_files),
        "ok": 0,
        "ng_stage1": 0,
        "ng_stage2": 0,
        "unrecognized": 0,
    }
    start_time = time.time()

    for idx, filename in enumerate(image_files, 1):
        test_image_path = os.path.join(test_dir, filename)
        print(f"\n[{idx}/{stats['total']}] Inspecting {filename} ...")

        row_data = {
            "Filename": filename,
            "Model_Type": "Unknown",
            "Product_ROI": "",
            "Stage1_Result": "NotRun",
            "Stage1_Score": "",
            "Stage1_LocScore": "",
            "Stage2_Result": "NotRun",
            "Stage2_RawDetections": "",
            "Final_Decision": "NG",
            "Details": "",
        }

        ref_subfolder_name = determine_ref_model(filename)
        if not ref_subfolder_name:
            print("  - Skipped: unable to infer view from filename")
            row_data["Details"] = "Filename does not contain front/top/right/left/back"
            stats["unrecognized"] += 1
            report_data.append(row_data)
            continue

        row_data["Model_Type"] = ref_subfolder_name
        ref_folder = os.path.join(base_dir, "assets", "images_ref", ref_subfolder_name)

        stage1_result = run_stage1_localized(test_image_path, ref_folder, debug=False)
        final_img = stage1_result.get("annotated_image")
        row_data["Product_ROI"] = format_bbox(stage1_result.get("bbox"))
        row_data["Stage1_Score"] = stage1_result.get("score", "")
        row_data["Stage1_LocScore"] = f"{float(stage1_result.get('localization_score', 0.0)):.3f}"

        if not stage1_result.get("pass", False):
            issues = stage1_result.get("issues") or ["Stage 1 failed"]
            detail_text = " | ".join(issues)
            print(f"  - Stage 1 FAIL: {detail_text}")
            row_data["Stage1_Result"] = "Fail"
            row_data["Final_Decision"] = "NG"
            row_data["Details"] = detail_text
            stats["ng_stage1"] += 1
        else:
            print(
                f"  - Stage 1 PASS: ROI={row_data['Product_ROI']} "
                f"loc={row_data['Stage1_LocScore']} score={row_data['Stage1_Score']}"
            )
            row_data["Stage1_Result"] = "Pass"

            has_defect, msg2, img2, stage2_details = yolo_detector.detect(
                test_image_path,
                stage1_result["bbox"],
                conf_threshold=STAGE2_CONF_THRESHOLD,
                debug=False,
            )
            final_img = img2
            row_data["Stage2_RawDetections"] = stage2_details.get("raw_detection_count", 0)

            if has_defect:
                print(f"  - Stage 2/3 FAIL: {msg2}")
                row_data["Stage2_Result"] = "Fail"
                row_data["Final_Decision"] = "NG"
                row_data["Details"] = msg2
                stats["ng_stage2"] += 1
            else:
                print(f"  - Stage 2/3 PASS: {msg2}")
                row_data["Stage2_Result"] = "Pass"
                row_data["Final_Decision"] = "OK"
                row_data["Details"] = msg2
                stats["ok"] += 1

        if row_data["Final_Decision"] == "NG" and final_img is not None:
            save_path = os.path.join(output_dir, f"NG_{filename}")
            cv2.imwrite(save_path, final_img)

        report_data.append(row_data)

    print("\n" + "=" * 60)
    print("Generating inspection report...")
    with open(report_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "Filename",
            "Model_Type",
            "Product_ROI",
            "Stage1_Result",
            "Stage1_Score",
            "Stage1_LocScore",
            "Stage2_Result",
            "Stage2_RawDetections",
            "Final_Decision",
            "Details",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_data)

    total_time = time.time() - start_time
    fps = stats["total"] / total_time if total_time > 0 else 0.0

    print(f"Batch output directory: {output_dir}")
    print("Summary:")
    print(f"  - Total images: {stats['total']} (unrecognized: {stats['unrecognized']})")
    print(f"  - OK: {stats['ok']}")
    print(f"  - Stage 1 NG: {stats['ng_stage1']}")
    print(f"  - Stage 2/3 NG: {stats['ng_stage2']}")

    valid_total = stats["total"] - stats["unrecognized"]
    if valid_total > 0:
        yield_rate = (stats["ok"] / valid_total) * 100.0
        print(f"  - Yield: {yield_rate:.2f}%")

    print(f"  - Speed: {fps:.2f} img/s (total {total_time:.2f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
