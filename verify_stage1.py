import os

import cv2

from src.stage1_vision import Stage1Detector


def run_stage1(test_img_path, ref_folder_path, debug=False):
    """Backward-compatible Stage 1 wrapper using the legacy integrity check."""
    if not os.path.exists(ref_folder_path):
        return False, f"Reference folder not found: {ref_folder_path}", None

    if not os.path.exists(test_img_path):
        return False, f"Test image not found: {test_img_path}", None

    try:
        detector = Stage1Detector(ref_folder_path)
        is_passed, msg, result_img = detector.check_integrity(test_img_path)

        if debug and result_img is not None:
            h, w = result_img.shape[:2]
            resized_img = cv2.resize(result_img, (int(w * 0.6), int(h * 0.6)))
            cv2.imshow("Stage 1 Match Result", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return is_passed, msg, result_img
    except Exception as exc:
        return False, f"Stage 1 error: {exc}", None


def run_stage1_localized(test_img_path, ref_folder_path, debug=False):
    """Stage 1 wrapper that always returns product ROI localization details."""
    if not os.path.exists(ref_folder_path):
        return {
            "pass": False,
            "status": "FAIL",
            "issues": [f"Reference folder not found: {ref_folder_path}"],
            "bbox": None,
            "annotated_image": None,
        }

    if not os.path.exists(test_img_path):
        return {
            "pass": False,
            "status": "FAIL",
            "issues": [f"Test image not found: {test_img_path}"],
            "bbox": None,
            "annotated_image": None,
        }

    detector = Stage1Detector(ref_folder_path)
    result = detector.inspect_with_localization(test_img_path)

    if debug and result.get("annotated_image") is not None:
        cv2.imshow("Stage 1 ROI Localization", result["annotated_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result
