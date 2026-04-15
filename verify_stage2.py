import os

import cv2
import numpy as np
from ultralytics import YOLO

from src.stage3_rules import Stage3RuleEngine


def imread_bgr_safe(file_path):
    """Read local images safely on Windows paths that may contain Chinese characters."""
    img_array = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


class Stage2Detector:
    def __init__(self, model_path):
        """Load the defect model once for the whole batch lifecycle."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        print(f"[Stage2] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.rule_engine = Stage3RuleEngine()
        print("[Stage2] Model ready.")

    @staticmethod
    def _clip_roi_bbox(roi_bbox, width, height):
        x1, y1, x2, y2 = [int(v) for v in roi_bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    def _predict_roi(self, roi_bgr, conf_threshold):
        results = self.model.predict(source=roi_bgr, conf=conf_threshold, save=False, verbose=False)
        result_obj = results[0]

        detections = []
        for box in result_obj.boxes:
            cls_id = int(box.cls[0])
            cls_name = result_obj.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            detections.append({
                "cls": cls_name,
                "box": xyxy,
                "conf": conf,
                "status": "WAITING",
            })

        return detections

    def detect(self, img_input, roi_bbox, conf_threshold=0.25, debug=False):
        """
        Run defect inference only inside product ROI, then apply Stage 3 rules.

        :param img_input: image path or BGR ndarray
        :param roi_bbox: [x1, y1, x2, y2] in full-image coordinates
        :return: (has_defect, msg, result_image, details)
        """
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                return False, f"Image not found: {img_input}", None, {}
            full_img = imread_bgr_safe(img_input)
        else:
            full_img = img_input.copy() if img_input is not None else None

        if full_img is None:
            return False, "Image load failed", None, {}

        h_img, w_img = full_img.shape[:2]
        x1, y1, x2, y2 = self._clip_roi_bbox(roi_bbox, w_img, h_img)
        roi_bgr = full_img[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return False, "Invalid product ROI", full_img, {}

        detections_roi = self._predict_roi(roi_bgr, conf_threshold=conf_threshold)

        detections_full = []
        for det in detections_roi:
            dx1, dy1, dx2, dy2 = det["box"]
            detections_full.append({
                **det,
                "box": [dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1],
            })

        final_verdict, processed_detections, final_desc = self.rule_engine.apply_rules(detections_full)

        annotated = full_img.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(
            annotated,
            "Product ROI",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 215, 255),
            2,
        )

        for det in processed_detections:
            bx1, by1, bx2, by2 = [int(v) for v in det["box"]]
            color = det.get("color", (255, 0, 0))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(
                annotated,
                det.get("display_label", det["cls"]),
                (bx1, max(20, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        title = f"Stage2+3 {final_verdict} | ROI detections={len(detections_full)}"
        title_color = (0, 0, 255) if final_verdict == "FAIL" else (0, 180, 0)
        cv2.putText(annotated, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, title_color, 2)

        if debug:
            cv2.imshow("Stage 2 ROI Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        details = {
            "roi_bbox": [x1, y1, x2, y2],
            "raw_detection_count": len(detections_full),
            "processed_detections": processed_detections,
            "verdict": final_verdict,
            "description": final_desc,
        }
        has_defect = final_verdict == "FAIL"
        return has_defect, final_desc, annotated, details
