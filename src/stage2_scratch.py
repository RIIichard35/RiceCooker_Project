"""
Stage 2 划痕/裂纹检测模块 (src/stage2_scratch.py)
===================================================
基于 YOLOv8 ONNX 模型，在 Stage 1 定位到的产品 ROI 内
检测表面划痕（scratch）和裂纹（crack）。

核心流程
--------
1. 从 Stage 1 结果取 bbox + product_mask
2. 在 bbox 基础上加 padding，裁剪出产品区域
3. 将裁剪图 resize 到 640×640 送入 ONNX 模型
4. 解析输出 [1, 6, 8400]，执行 NMS
5. 将检测框坐标映射回原图
6. 用 product_mask 过滤掉中心点不在产品内的框（去除背景误报）
7. 返回结构化结果字典

模型信息（自动从 ONNX 元数据读取）
--------------------------------------
  输入  : images  [1, 3, 640, 640]
  输出  : output0 [1, 6, 8400]  → [cx, cy, w, h, scratch_score, crack_score]
  类别  : {0: 'scratch', 1: 'crack'}
  NMS   : 模型未内置，本模块手动实现
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

# ── 可选：onnxruntime（推荐），fallback 到 cv2.dnn ──────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────

def _xywh2xyxy(cx: float, cy: float, w: float, h: float) -> tuple:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _iou(box_a: list, box_b: list) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-6)


def _nms(boxes: list[dict], iou_thr: float = 0.45) -> list[dict]:
    """简单 NMS，按 score 降序排列后贪心合并。"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    kept = []
    suppressed = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if suppressed[i]:
            continue
        kept.append(b)
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and _iou(b["xyxy"], boxes[j]["xyxy"]) > iou_thr:
                suppressed[j] = True
    return kept


# ─────────────────────────────────────────────────────────────────────────
# 主检测器
# ─────────────────────────────────────────────────────────────────────────

class Stage2Detector:
    """
    YOLOv8 ONNX 划痕/裂纹检测器。

    参数
    ----
    model_path     : str   — ONNX 模型路径（相对或绝对）
    conf_threshold : float — 置信度阈值，低于此值的框丢弃（默认 0.25）
    iou_threshold  : float — NMS IoU 阈值（默认 0.45）
    roi_padding    : float — 在 Stage1 bbox 基础上额外扩展的比例（默认 0.05）
    """

    MODEL_INPUT_W = 640
    MODEL_INPUT_H = 640

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        roi_padding:    float = 0.05,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.roi_padding    = roi_padding
        self.class_names:   dict[int, str] = {0: "scratch", 1: "crack"}

        model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Stage2] 找不到模型文件: {model_path}")

        # 加载 ONNX
        if _ORT_AVAILABLE:
            self._sess = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._sess.get_inputs()[0].name
            # 从元数据读取类别名（如果有）
            meta = self._sess.get_modelmeta()
            raw = meta.custom_metadata_map.get("names")
            if raw:
                try:
                    self.class_names = {
                        int(k): v for k, v in json.loads(
                            raw.replace("'", '"')
                        ).items()
                    }
                except Exception:
                    pass
            print(f"[Stage2] 模型加载成功（onnxruntime），类别: {self.class_names}")
        else:
            self._net = cv2.dnn.readNetFromONNX(model_path)
            print(f"[Stage2] 模型加载成功（cv2.dnn），类别: {self.class_names}")

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def inspect(
        self,
        image_bgr:    np.ndarray,
        stage1_result: dict,
    ) -> dict:
        """
        在 Stage1 定位到的产品区域内执行划痕/裂纹检测。

        参数
        ----
        image_bgr     : 原始 BGR 图像（Stage1 输入的同一张图）
        stage1_result : Stage1Detector.inspect_with_localization() 的返回字典

        返回
        ----
        dict，含以下字段：
            status         : "PASS" | "FAIL"
            defects        : list[dict]  — 每个缺陷 {class_name, score, xyxy, area}
            defect_counts  : dict        — {class_name: count}
            annotated_image: np.ndarray  — BGR 标注图（原图尺寸）
            crop_image     : np.ndarray  — 产品裁剪图（用于展示）
            issues         : list[str]
        """
        result = self._empty_result()

        bbox         = stage1_result.get("bbox")
        product_mask = None  # Stage1 未直接返回 mask，用 bbox + padding 替代

        if bbox is None:
            result["issues"].append("Stage1 未提供产品定位框，无法执行细筛")
            return result

        h_img, w_img = image_bgr.shape[:2]

        # ── Step 1: 计算带 padding 的裁剪框 ─────────────────────────────
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * self.roi_padding)
        py = int(bh * self.roi_padding)
        cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
        cx2 = min(w_img, x2 + px); cy2 = min(h_img, y2 + py)
        result["crop_box"] = [cx1, cy1, cx2, cy2]

        # ── Step 2: 裁剪产品区域 ─────────────────────────────────────────
        crop = image_bgr[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            result["issues"].append("产品裁剪区域无效")
            return result
        result["crop_image"] = crop

        # ── Step 3: 预处理 → ONNX 推理 ──────────────────────────────────
        crop_h, crop_w = crop.shape[:2]
        blob = self._preprocess(crop)
        raw  = self._infer(blob)           # shape: (6, 8400)

        # ── Step 4: 解析输出 + NMS ───────────────────────────────────────
        candidates = self._parse_output(raw, crop_w, crop_h)
        kept       = _nms(candidates, self.iou_threshold)

        # ── Step 5: 坐标映射回原图 + product_mask 过滤 ──────────────────
        vis = image_bgr.copy()
        # 绘制产品区域框（Stage1 bbox）
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(vis, "Product ROI", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        for det in kept:
            # 裁剪坐标 → 原图坐标
            dx1, dy1, dx2, dy2 = det["xyxy"]
            ox1 = int(dx1 + cx1);  oy1 = int(dy1 + cy1)
            ox2 = int(dx2 + cx1);  oy2 = int(dy2 + cy1)
            ox1 = max(0, ox1);     oy1 = max(0, oy1)
            ox2 = min(w_img, ox2); oy2 = min(h_img, oy2)

            area = (ox2 - ox1) * (oy2 - oy1)
            if area <= 0:
                continue

            # 框中心必须在产品 bbox 内（防止 padding 区域误报）
            cx_det = (ox1 + ox2) // 2
            cy_det = (oy1 + oy2) // 2
            if not (x1 <= cx_det <= x2 and y1 <= cy_det <= y2):
                continue

            cls_name = det["class_name"]
            score    = det["score"]

            defect_entry = {
                "class_name": cls_name,
                "score":      round(score, 3),
                "xyxy":       [ox1, oy1, ox2, oy2],
                "area":       area,
            }
            result["defects"].append(defect_entry)
            result["defect_counts"][cls_name] = (
                result["defect_counts"].get(cls_name, 0) + 1
            )

            # 绘制缺陷框
            color = (0, 0, 220) if cls_name == "scratch" else (0, 100, 255)
            cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), color, 2)
            label = f"{cls_name} {score:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (ox1, oy1 - lh - 6), (ox1 + lw + 4, oy1), color, -1)
            cv2.putText(vis, label, (ox1 + 2, oy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Step 6: 综合判定 ─────────────────────────────────────────────
        n_defects = len(result["defects"])
        if n_defects == 0:
            result["status"] = "PASS"
            status_color = (0, 180, 0)
        else:
            result["status"] = "FAIL"
            result["issues"].append(
                f"发现 {n_defects} 处表面缺陷："
                + "、".join(
                    f"{v} 处{k}" for k, v in result["defect_counts"].items()
                )
            )
            status_color = (0, 0, 220)

        # 状态文字叠加
        status_text = (
            f"Stage2 {result['status']}  |  "
            f"scratch={result['defect_counts'].get('scratch', 0)}  "
            f"crack={result['defect_counts'].get('crack', 0)}"
        )
        cv2.putText(vis, status_text, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

        result["annotated_image"] = vis
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → float32 NCHW blob，归一化到 0~1。"""
        img = cv2.resize(crop_bgr, (self.MODEL_INPUT_W, self.MODEL_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """执行 ONNX 推理，返回 (6, 8400) 数组。"""
        if _ORT_AVAILABLE:
            out = self._sess.run(None, {self._input_name: blob})[0]  # (1,6,8400)
        else:
            self._net.setInput(blob)
            out = self._net.forward()
        return out[0]  # (6, 8400)

    def _parse_output(
        self,
        raw: np.ndarray,
        crop_w: int,
        crop_h: int,
    ) -> list[dict]:
        """
        解析 YOLOv8 输出 (6, 8400)。
        各行：cx cy w h score_cls0 score_cls1（均在 640 坐标系）
        """
        scale_x = crop_w / self.MODEL_INPUT_W
        scale_y = crop_h / self.MODEL_INPUT_H

        results = []
        num_classes = raw.shape[0] - 4  # = 2
        pred = raw.T  # (8400, 6)

        for row in pred:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            class_scores = row[4: 4 + num_classes]
            cls_id = int(np.argmax(class_scores))
            score  = float(class_scores[cls_id])

            if score < self.conf_threshold:
                continue

            # 坐标转换到裁剪图尺寸
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y

            results.append({
                "class_id":   cls_id,
                "class_name": self.class_names.get(cls_id, f"cls{cls_id}"),
                "score":      score,
                "xyxy":       [x1, y1, x2, y2],
            })

        return results

    @staticmethod
    def _empty_result() -> dict:
        return {
            "status":         "PASS",
            "defects":        [],
            "defect_counts":  {},
            "annotated_image": None,
            "crop_image":     None,
            "crop_box":       None,
            "issues":         [],
        }
