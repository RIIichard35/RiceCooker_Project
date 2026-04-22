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
import time
from pathlib import Path

import cv2
import numpy as np

# ── 推理后端自动探测（优先级：Hailo > onnxruntime > cv2.dnn） ──────────
# Hailo：树莓派 AI HAT+（Hailo-8L，13 TOPS）
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType,
    )
    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False

# onnxruntime：PC / Pi（无 Hailo 时回退）
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
    YOLOv8 划痕/裂纹检测器。

    推理后端自动选择（优先级从高到低）：
      1. Hailo-8L（AI HAT+，需要 .hef 文件 + hailo_platform 库）
      2. onnxruntime CPU（PC / Pi 通用）
      3. cv2.dnn（最低兼容回退）

    参数
    ----
    model_path     : str   — ONNX 模型路径；若同目录下存在同名 .hef 则优先用 Hailo
    hef_path       : str | None — 显式指定 .hef 路径（None 则自动推断）
    conf_threshold : float — 置信度阈值（默认 0.25）
    iou_threshold  : float — NMS IoU 阈值（默认 0.45）
    roi_padding    : float — 产品框扩展比例（默认 0.05）
    """

    MODEL_INPUT_W = 640
    MODEL_INPUT_H = 640

    def __init__(
        self,
        model_path: str,
        hef_path:       str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        roi_padding:    float = 0.05,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.roi_padding    = roi_padding
        self.class_names:   dict[int, str] = {0: "scratch", 1: "crack"}
        self._backend = "none"

        model_path = os.path.normpath(model_path)

        # ── 自动推断 .hef 路径（同目录，同名，不同扩展名）
        if hef_path is None:
            hef_path = str(Path(model_path).with_suffix(".hef"))

        # ── 后端 1：Hailo ────────────────────────────────────────────────
        if _HAILO_AVAILABLE and os.path.exists(hef_path):
            try:
                self._load_hailo(hef_path)
                print(f"[Stage2] Hailo-8L 推理就绪  ← {Path(hef_path).name}")
                self._backend = "hailo"
                return
            except Exception as e:
                print(f"[Stage2] Hailo 加载失败，退回 onnxruntime: {e}")

        # ── 后端 2 / 3：ONNX ─────────────────────────────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Stage2] 找不到模型文件: {model_path}")

        if _ORT_AVAILABLE:
            self._sess = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._sess.get_inputs()[0].name
            meta = self._sess.get_modelmeta()
            raw  = meta.custom_metadata_map.get("names")
            if raw:
                try:
                    self.class_names = {
                        int(k): v for k, v in json.loads(
                            raw.replace("'", '"')
                        ).items()
                    }
                except Exception:
                    pass
            print(f"[Stage2] onnxruntime CPU 推理就绪，类别: {self.class_names}")
            self._backend = "ort"
        else:
            self._net = cv2.dnn.readNetFromONNX(model_path)
            print(f"[Stage2] cv2.dnn 推理就绪（兼容模式），类别: {self.class_names}")
            self._backend = "cv2dnn"

    # ── Hailo 初始化 ──────────────────────────────────────────────────────

    def _load_hailo(self, hef_path: str) -> None:
        """加载 .hef 并建立 Hailo 推理管道（在 Pi 上调用）。"""
        self._hef      = HEF(hef_path)
        self._vdevice  = VDevice()          # 连接 Hailo-8L 硬件
        cfg_params     = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        net_groups     = self._vdevice.configure(self._hef, cfg_params)
        self._net_group        = net_groups[0]
        self._net_group_params = self._net_group.create_params()

        self._in_params  = InputVStreamParams.make(
            self._net_group, format_type=FormatType.FLOAT32
        )
        self._out_params = OutputVStreamParams.make(
            self._net_group, format_type=FormatType.FLOAT32
        )
        # 读取输入层名称
        self._hailo_input_name = list(
            self._hef.get_input_vstream_infos()
        )[0].name

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
        """
        执行推理，返回 (6, 8400) 数组。
        根据 self._backend 自动选择 Hailo / onnxruntime / cv2.dnn。
        """
        if self._backend == "hailo":
            return self._infer_hailo(blob)
        if self._backend == "ort":
            out = self._sess.run(None, {self._input_name: blob})[0]  # (1,6,8400)
            return out[0]
        # cv2.dnn
        self._net.setInput(blob)
        return self._net.forward()[0]   # (6, 8400)

    def _infer_hailo(self, blob: np.ndarray) -> np.ndarray:
        """
        通过 hailo_platform 在 Hailo-8L 上推理。
        blob: (1, 3, 640, 640) float32  →  返回 (6, 8400) float32
        """
        # Hailo 输入格式：NHWC float32
        nhwc = blob[0].transpose(1, 2, 0)[np.newaxis]  # (1, 640, 640, 3)
        input_data = {self._hailo_input_name: nhwc}

        with InferVStreams(
            self._net_group,
            self._in_params,
            self._out_params,
        ) as pipeline:
            with self._net_group.activate(self._net_group_params):
                raw_out = pipeline.infer(input_data)

        # 取第一个输出张量，整理成 (6, 8400)
        out_key = list(raw_out.keys())[0]
        out     = raw_out[out_key]          # 通常 (1, 8400, 6) 或 (1, 6, 8400)
        out     = np.squeeze(out)           # (8400, 6) 或 (6, 8400)
        if out.shape[0] == 8400:            # (8400, 6) → transpose → (6, 8400)
            out = out.T
        return out.astype(np.float32)

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
