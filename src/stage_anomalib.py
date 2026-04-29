"""
src/stage_anomalib.py — Anomalib PatchCore 推理模块
====================================================

推理优先级：
  1. ONNX（assets/models/anomalib/<view>/model.onnx）→ 纯 onnxruntime，适合树莓派
  2. Lightning checkpoint（anomalib Engine.predict）→ 需要 torch + anomalib

接收 Stage 1 输出的 cutout_rgba（BGRA，np.ndarray），返回：
  - anomaly_score : float   （0~1，越大越异常）
  - anomaly_map   : ndarray  （单通道 float32 热图，与输入同 H×W）
  - status        : "PASS" | "FAIL"
  - issues        : list[str]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    ANOMALIB_HEATMAP_ALPHA,
    ANOMALIB_HOLE_FAIL_THRESH,
    ANOMALIB_INPUT_SIZE,
    ANOMALIB_MODEL_DIR,
    ANOMALIB_SCORE_THRESHOLD,
    ANOMALIB_TEMPLATE_FAIL_THRESH,
    ANOMALIB_TEMPLATE_GATE_ENABLED,
    get_anomalib_threshold,
    ST_DIR,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ─────────────────────────────────────────────────────────────────────────
# 预处理工具
# ─────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _bgra_to_rgb_gray_bg(
    bgra: np.ndarray,
    size: int,
    bg_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    BGRA 抠图 → 固定灰色背景 → RGB → resize。
    返回 (size, size, 3) uint8 RGB。
    """
    if bgra.ndim == 3 and bgra.shape[2] == 4:
        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        # OpenCV 存储顺序 BGR，取前三通道转 RGB
        bgr = bgra[:, :, :3].astype(np.float32)
        bg  = np.full_like(bgr, bg_color[::-1], dtype=np.float32)  # BGR 顺序
        merged_bgr = alpha[..., None] * bgr + (1 - alpha[..., None]) * bg
        rgb = cv2.cvtColor(np.clip(merged_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    elif bgra.ndim == 3 and bgra.shape[2] == 3:
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB)
    else:
        gray = bgra if bgra.ndim == 2 else cv2.cvtColor(bgra, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def _to_tensor(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) → normalized float32 (1,3,H,W)。"""
    x = rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)


# ─────────────────────────────────────────────────────────────────────────
# 主检测器
# ─────────────────────────────────────────────────────────────────────────

class AnomalibDetector:
    """
    PatchCore 异常检测器（单视角）。

    参数
    ----
    view       : str   — 视角名称，对应 assets/models/anomalib/<view>/
    model_dir  : str   — 可覆盖默认模型目录（通常不需要）
    threshold  : float — 异常分阈值（覆盖 config）
    """

    def __init__(
        self,
        view: str,
        model_dir: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.view      = view
        self.threshold = float(
            threshold if threshold is not None else get_anomalib_threshold(view)
        )
        self._size     = int(ANOMALIB_INPUT_SIZE)

        base = Path(model_dir or ANOMALIB_MODEL_DIR) / view
        self._onnx_path = base / "model.onnx"
        # Anomalib 默认 checkpoint 位置
        self._ckpt_path = next(base.rglob("*.ckpt"), None)

        self._ort_session: Any | None = None
        self._lightning_model: Any | None = None
        self._lightning_engine: Any | None = None
        self._backend: str = "none"
        self._template_grays: list[np.ndarray] = []
        self._template_orb_desc: list[tuple[list[Any], np.ndarray] | None] = []
        self._orb = cv2.ORB_create(nfeatures=1200)

        self._load_model(base)
        self._load_template_bank()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load_model(self, base: Path) -> None:
        if ort is not None and self._onnx_path.exists():
            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self._ort_session = ort.InferenceSession(
                    str(self._onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._backend = "onnx"
                print(f"[AnomalibDetector/{self.view}] 已加载 ONNX: {self._onnx_path}")
                return
            except Exception as e:
                print(f"[AnomalibDetector/{self.view}] ONNX 加载失败: {e}，尝试 Lightning")

        if self._ckpt_path is not None:
            try:
                from anomalib.engine import Engine
                from anomalib.models import Patchcore

                model = Patchcore(
                    backbone="wide_resnet50_2",
                    layers=["layer2", "layer3"],
                    pre_trained=False,
                )
                engine = Engine(default_root_dir=str(base))
                self._lightning_model  = model
                self._lightning_engine = engine
                self._lightning_ckpt   = str(self._ckpt_path)
                self._backend = "lightning"
                print(
                    f"[AnomalibDetector/{self.view}] "
                    f"已加载 Lightning checkpoint: {self._ckpt_path}"
                )
                return
            except Exception as e:
                print(f"[AnomalibDetector/{self.view}] Lightning 加载失败: {e}")

        print(
            f"[AnomalibDetector/{self.view}] [警告] 未找到任何模型文件，"
            f"请先运行 scripts/train_anomalib.py 训练。"
        )

    @property
    def is_ready(self) -> bool:
        return self._backend in ("onnx", "lightning")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def inspect(
        self,
        cutout_input: np.ndarray,
        stage1_fg_mask: np.ndarray | None = None,
    ) -> dict:
        """
        对产品裁剪图执行二次抠图 + Anomalib 异常检测。

        参数
        ----
        cutout_input : np.ndarray
            Stage 1 产品裁剪图，可为：
            - BGRA (H,W,4)：含 alpha 通道的透明抠图
            - BGR  (H,W,3)：已渲染灰色背景的产品裁剪图
        stage1_fg_mask : np.ndarray | None
            与 cutout 同尺寸的 Stage1 二值产品蒙版（uint8，产品区 >127）。
            若提供，则优先用于约束前景，避免 RGBA alpha 将产线背景当作不透明前景。

        返回
        ----
        dict 含：
            clean_cutout  : np.ndarray (H,W,4) BGRA 二次抠图透明蒙版
            anomaly_score : float
            anomaly_map   : np.ndarray (H,W) float32 热图 0~1
            anomaly_boxes : list[[x1,y1,x2,y2]]  （二次抠图坐标系）
            heatmap_overlay: np.ndarray (H,W,3) BGR 热图叠加图
            status        : "PASS" | "FAIL" | "SKIP"
            issues        : list[str]
        """
        orig_h, orig_w = cutout_input.shape[:2]
        empty = self._empty_result(orig_h, orig_w)

        # ── 第二次抠图：灰色背景差分 ──────────────────────────────────
        clean_bgra = self._second_pass_cutout(cutout_input, stage1_fg_mask=stage1_fg_mask)
        if stage1_fg_mask is not None and clean_bgra.ndim == 3 and clean_bgra.shape[2] == 4:
            sm = stage1_fg_mask
            if sm.shape[:2] != (orig_h, orig_w):
                sm = cv2.resize(sm, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            outside = sm <= 127
            clean_bgra[outside] = (128, 128, 128, 0)
        empty["clean_cutout"] = clean_bgra

        if not self.is_ready:
            empty["status"] = "SKIP"
            empty["issues"] = ["Anomalib 模型未加载（跳过）"]
            return empty

        # ── Anomalib 推理（用二次抠图的灰色背景 RGB） ─────────────────
        rgb = _bgra_to_rgb_gray_bg(clean_bgra, self._size)
        tpl_inner: np.ndarray | None = None
        if clean_bgra.ndim == 3 and clean_bgra.shape[2] == 4:
            fg0 = (clean_bgra[:, :, 3] > 10).astype(np.uint8) * 255
            if np.any(fg0):
                er = max(5, min(31, int(0.035 * float(min(orig_h, orig_w)))))
                if er % 2 == 0:
                    er += 1
                k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))
                inner = cv2.erode(fg0, k_er, iterations=2)
                tpl_inner = cv2.resize(
                    inner,
                    (self._size, self._size),
                    interpolation=cv2.INTER_NEAREST,
                )

        if self._backend == "onnx":
            score, raw_map = self._infer_onnx(rgb)
        else:
            score, raw_map = self._infer_lightning(rgb)

        if score is None:
            empty["status"] = "SKIP"
            empty["issues"] = ["Anomalib 推理失败（跳过）"]
            return empty

        # 归一化热图，resize 回原始尺寸
        amap = cv2.resize(raw_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        amap_norm = self._normalize_map(amap)
        # 模板一致性异常图（轮廓处易有配准残差，仅用腐蚀后的「内带」统计模板分与热图贡献）
        tpl_map_small, tpl_score = self._template_consistency_map(rgb, tpl_inner)
        tpl_map = cv2.resize(tpl_map_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if clean_bgra.ndim == 3 and clean_bgra.shape[2] == 4:
            fg_full = (clean_bgra[:, :, 3] > 0).astype(np.uint8) * 255
            if np.any(fg_full):
                br = max(5, min(55, int(0.048 * float(min(orig_h, orig_w)))))
                if br % 2 == 0:
                    br += 1
                k_br = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (br, br))
                tpl_band = cv2.erode(fg_full, k_br, iterations=2)
                tpl_map = tpl_map.astype(np.float32) * (
                    tpl_band.astype(np.float32) / 255.0
                )
        tpl_map_norm = self._normalize_map(tpl_map)

        # 明显破洞兜底检测（暗区+纹理双通道，不依赖 alpha 孔洞）
        hole_map, hole_boxes, hole_score = self._detect_obvious_holes(clean_bgra)

        # 融合热图：PatchCore +（可选）模板残差 + 破洞图
        if ANOMALIB_TEMPLATE_GATE_ENABLED:
            fused_map = np.maximum(amap_norm, tpl_map_norm)
        else:
            fused_map = amap_norm.copy()
        fused_map = np.maximum(fused_map, hole_map)
        # 仅在产品凸包范围内统计异常，抑制边缘配准误差
        # （注：不再仅限 erode 内部，避免把靠边破洞抑制掉）
        if clean_bgra.ndim == 3 and clean_bgra.shape[2] == 4:
            fg_mask = (clean_bgra[:, :, 3] > 0).astype(np.uint8) * 255
            # 构建凸包掩码
            convex_m = np.zeros_like(fg_mask)
            fg_pts = np.column_stack(np.where(fg_mask > 0))
            if len(fg_pts) >= 10:
                hull = cv2.convexHull(fg_pts[:, ::-1].astype(np.float32))
                cv2.fillConvexPoly(convex_m, hull.astype(np.int32), 255)
            else:
                convex_m = fg_mask
            fused_map = fused_map * (convex_m.astype(np.float32) / 255.0)

        overlay = self._make_heatmap_overlay(clean_bgra, fused_map)

        # PatchCore 分 + 模板残差分（阈值为 ANOMALIB_TEMPLATE_FAIL_THRESH） + 破洞 联合判定
        template_fail_thresh = float(ANOMALIB_TEMPLATE_FAIL_THRESH)
        hole_fail_thresh = float(ANOMALIB_HOLE_FAIL_THRESH)
        tpl_reject = ANOMALIB_TEMPLATE_GATE_ENABLED and (tpl_score > template_fail_thresh)
        status = "FAIL" if (
            score > self.threshold
            or tpl_reject
            or hole_score > hole_fail_thresh
        ) else "PASS"
        issues: list[str] = []
        if status == "FAIL":
            if score > self.threshold:
                issues.append(
                    f"PatchCore 异常分 {score:.3f} 超过阈值 {self.threshold:.2f}，"
                    "疑似缺件/漏件"
                )
            if tpl_reject:
                issues.append(
                    f"模板一致性分 {tpl_score:.3f} 超过阈值 {template_fail_thresh:.2f}，"
                    "疑似与标准外观不一致（含轮廓配准误差时参考热图边缘是否发红）"
                )
            if hole_score > hole_fail_thresh:
                pct = hole_score * 100
                issues.append(
                    f"检测到明显破洞/破损区域（面积占比 {pct:.1f}%），疑似严重损坏"
                )

        bm = max(28, int(0.04 * float(min(orig_h, orig_w))))
        anomaly_boxes = self._heatmap_to_boxes(
            fused_map, thresh=0.52, min_area=650, max_area_ratio=0.22, border_margin=bm
        )
        # 合并破洞兜底框（优先保证明显破损可见）
        if hole_boxes:
            # 去重：若已有框与破洞框 IoU > 0.3 则不重复添加
            def _iou(a: list[int], b: list[int]) -> float:
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
                return inter / (ua + 1e-6)
            for hb in hole_boxes:
                if not any(_iou(hb, eb) > 0.3 for eb in anomaly_boxes):
                    anomaly_boxes.append(hb)

        if status != "FAIL":
            # 通过样本不展示红框，避免误导操作员。
            anomaly_boxes = []

        combo_score = float(
            max(
                score,
                tpl_score if ANOMALIB_TEMPLATE_GATE_ENABLED else 0.0,
                hole_score * 5.0,
            )
        )
        return {
            "clean_cutout":      clean_bgra,
            "patchcore_score":   float(score),
            "template_score":    float(tpl_score),
            "anomaly_score":     combo_score,
            "hole_score":        hole_score,
            "hole_boxes":        hole_boxes,
            "anomaly_map":       fused_map,
            "anomaly_boxes":     anomaly_boxes,
            "heatmap_overlay":   overlay,
            "status":            status,
            "issues":            issues,
        }

    # ------------------------------------------------------------------
    # ONNX 推理
    # ------------------------------------------------------------------

    def _infer_onnx(
        self, rgb: np.ndarray
    ) -> tuple[float | None, np.ndarray | None]:
        try:
            tensor = _to_tensor(rgb)
            in_name  = self._ort_session.get_inputs()[0].name
            outputs  = self._ort_session.run(None, {in_name: tensor})
            out_names = [o.name.lower() for o in self._ort_session.get_outputs()]
            # 兼容多种输出排列。注意：部分导出里 pred_score 表示“正常度”而非异常分，
            # 因此统一以 anomaly_map 的峰值作为异常分数，避免方向误判。
            score_val, amap = self._parse_onnx_outputs(outputs, out_names, rgb.shape[:2])
            return score_val, amap
        except Exception as e:
            print(f"[AnomalibDetector/{self.view}] ONNX 推理异常: {e}")
            return None, None

    @staticmethod
    def _parse_onnx_outputs(
        outputs: list[np.ndarray],
        out_names: list[str],
        spatial_shape: tuple[int, int],
    ) -> tuple[float, np.ndarray]:
        h, w = spatial_shape
        amap: np.ndarray | None = None
        pred_mask: np.ndarray | None = None
        score_val: float | None = None

        for idx, out in enumerate(outputs):
            name = out_names[idx] if idx < len(out_names) else ""
            out = np.squeeze(out)

            # 优先按名称抓 anomaly_map
            if "anomaly_map" in name and out.ndim >= 2:
                amap = out.astype(np.float32)
                continue
            if "pred_mask" in name and out.ndim >= 2:
                pred_mask = out.astype(np.uint8)
                continue

            # 无名称提示时，2D 浮点张量也视作 anomaly_map
            if amap is None and out.ndim == 2 and np.issubdtype(out.dtype, np.floating):
                amap = out.astype(np.float32)
                continue
            if pred_mask is None and out.ndim == 2 and np.issubdtype(out.dtype, np.bool_):
                pred_mask = out.astype(np.uint8)
                continue

            # 兜底记录标量
            if out.ndim == 0 or (out.ndim == 1 and out.size == 1):
                # 若名字是 pred_score，很多模型表示“正常度”，这里只做兜底不用作主判据
                score_val = float(out.flat[0])

        if amap is None:
            amap = np.zeros((h, w), np.float32)

        # 1) 若有 pred_mask，优先用其面积比例，但自动判别方向：
        #    异常区域通常更稀疏，因此取“更稀疏的一侧”作为 anomaly mask。
        if pred_mask is not None:
            pm = (pred_mask > 0).astype(np.uint8)
            ratio = float(np.count_nonzero(pm)) / float(pm.size + 1e-6)
            if ratio <= 0.5:
                anomaly_mask = pm
            else:
                anomaly_mask = 1 - pm
                ratio = 1.0 - ratio
            score_val = ratio

            # 若 anomaly_map 与 anomaly_mask 方向相反（正常度图），自动翻转。
            if amap is not None and np.count_nonzero(anomaly_mask) > 0:
                anom_mean = float(np.mean(amap[anomaly_mask > 0]))
                norm_mean = float(np.mean(amap[anomaly_mask == 0])) if np.count_nonzero(anomaly_mask == 0) > 0 else anom_mean
                if anom_mean < norm_mean:
                    amap = 1.0 - amap
            elif amap is None or amap.size == 0:
                amap = anomaly_mask.astype(np.float32)
            return float(score_val), amap.astype(np.float32)

        # 2) 无 pred_mask 时，退回 anomaly_map 峰值
        map_score = float(np.max(amap)) if amap.size > 0 else 0.0
        if np.isfinite(map_score):
            score_val = map_score
        elif score_val is None:
            score_val = 0.0

        return score_val, amap

    # ------------------------------------------------------------------
    # Lightning 推理
    # ------------------------------------------------------------------

    def _infer_lightning(
        self, rgb: np.ndarray
    ) -> tuple[float | None, np.ndarray | None]:
        try:
            import tempfile

            import torch
            from PIL import Image

            # 写临时图片，通过 Engine.predict(data_path=...) 推理
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_path, img_bgr)

            preds = self._lightning_engine.predict(
                model=self._lightning_model,
                data_path=tmp_path,
                ckpt_path=self._lightning_ckpt,
            )
            os.unlink(tmp_path)

            if not preds:
                return None, None

            pred = preds[0]
            score_val = float(pred.pred_score.item() if hasattr(pred, "pred_score") else 0.0)
            if hasattr(pred, "anomaly_map") and pred.anomaly_map is not None:
                amap = pred.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
            else:
                h, w = rgb.shape[:2]
                amap = np.zeros((h, w), np.float32)
            return score_val, amap
        except Exception as e:
            print(f"[AnomalibDetector/{self.view}] Lightning 推理异常: {e}")
            return None, None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _load_template_bank(self) -> None:
        """加载 st/<view> 下标准抠图模板，用于一致性异常检测。"""
        face_dir = Path(ST_DIR) / self.view
        if not face_dir.exists():
            return
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        files = sorted([p for p in face_dir.iterdir() if p.suffix.lower() in valid_ext])[:80]
        for p in files:
            try:
                arr = np.fromfile(str(p), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                rgb = _bgra_to_rgb_gray_bg(img, self._size)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                kp, des = self._orb.detectAndCompute(gray, None)
                self._template_grays.append(gray)
                self._template_orb_desc.append((kp, des) if des is not None else None)
            except Exception:
                continue

    def _template_consistency_map(
        self,
        rgb: np.ndarray,
        inner_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        与标准模板库做一致性残差，返回异常图与异常分（0~1）。
        异常分定义为残差图 99 分位值；若提供 inner_mask（边缘腐蚀后的前景），
        只在带内统计并抑制轮廓处的配准假高亮。
        """
        h, w = rgb.shape[:2]
        if not self._template_grays:
            return np.zeros((h, w), np.float32), 0.0

        tgt_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp_tgt, des_tgt = self._orb.detectAndCompute(tgt_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        best_res: np.ndarray | None = None
        best_mean = 1e9

        for idx, ref_gray in enumerate(self._template_grays):
            aligned = ref_gray
            desc_pack = self._template_orb_desc[idx]
            if desc_pack is not None and des_tgt is not None and len(kp_tgt) >= 10:
                kp_ref, des_ref = desc_pack
                try:
                    matches = bf.knnMatch(des_ref, des_tgt, k=2)
                    good = [m for m, n in matches if m.distance < 0.78 * n.distance]
                    if len(good) >= 10:
                        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst = np.float32([kp_tgt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                        if H is not None:
                            aligned = cv2.warpPerspective(ref_gray, H, (w, h))
                except Exception:
                    pass

            res = cv2.absdiff(tgt_gray, aligned).astype(np.float32) / 255.0
            mean_res = float(np.mean(res))
            if mean_res < best_mean:
                best_mean = mean_res
                best_res = res

        if best_res is None:
            return np.zeros((h, w), np.float32), 0.0

        # 轻度平滑；若先乘 mask 再 blur，零边会把假残差渗进内缘，故在 blur 后再缩一圈mask统计
        res_blur = cv2.GaussianBlur(best_res, (5, 5), 0)
        if (
            inner_mask is not None
            and inner_mask.shape[0] == h
            and inner_mask.shape[1] == w
        ):
            m0 = inner_mask.astype(np.uint8)
            k_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            m_tight = cv2.erode(m0, k_tight, iterations=1)
            wts = m_tight.astype(np.float32) / 255.0
            res_blur = res_blur * wts
            vals = res_blur[wts > 0.05]
            if vals.size > 0:
                # 99 分位对少量未消尽的边带仍敏感，改用 96 + 内缩 mask
                score = float(np.percentile(vals, 96))
            else:
                score = 0.0
        else:
            score = float(np.percentile(res_blur, 99))
        return res_blur.astype(np.float32), score

    @staticmethod
    def _detect_obvious_holes(clean_bgra: np.ndarray) -> tuple[np.ndarray, list[list[int]], float]:
        """
        规则兜底：检测产品上的明显破洞/穿孔/撕裂破损区域。

        策略（双通道合并，不依赖 alpha 孔洞）
        ─────────────────────────────────────
        通道 A  alpha 内部孔洞
            填充前景轮廓再减去 alpha，得到 GrabCut 保留的"内洞"区域。

        通道 B  前景暗区检测（主要通道）
            GrabCut 把破洞当背景抠掉后 alpha=0，此时破洞在 BGR 层里仍是
            极暗像素。在前景覆盖范围（凸包）内，找与周围前景颜色差异极大的
            暗色连通区域，即为穿孔/破损。
            · 用前景像素的中位亮度估计产品正常亮度
            · 对整张 BGR 图计算灰度，在凸包范围内找亮度 < 自适应阈值且
              与周边前景颜色偏差大的区域
            · 形态学合并细碎区域后提取边界框

        通道 C  高频纹理异常（辅助）
            破洞边缘通常有撕裂细纹，用 Laplacian 高频图在前景内找异常高能量区，
            配合暗区掩码做与运算，避免把正常按键/纹理误判。

        返回 (hole_map float32 H×W, hole_boxes list[[x1,y1,x2,y2]], hole_score float)
        """
        h, w = clean_bgra.shape[:2]
        is_bgra = (clean_bgra.ndim == 3 and clean_bgra.shape[2] == 4)

        if is_bgra:
            alpha = clean_bgra[:, :, 3]
            bgr   = clean_bgra[:, :, :3].copy()
            fg    = (alpha > 0).astype(np.uint8) * 255
        else:
            bgr = (clean_bgra[:, :, :3].copy() if clean_bgra.ndim == 3
                   else cv2.cvtColor(clean_bgra, cv2.COLOR_GRAY2BGR))
            alpha = np.full((h, w), 255, dtype=np.uint8)
            fg    = np.ones((h, w), np.uint8) * 255

        boxes: list[list[int]] = []

        # ── 通道 A：alpha 内部孔洞 ────────────────────────────────────────
        hole_alpha = np.zeros((h, w), np.uint8)
        if is_bgra and np.count_nonzero(fg) > 0:
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(fg)
            if cnts:
                cv2.drawContours(filled, cnts, -1, 255, -1)
                hole_alpha = cv2.bitwise_and(filled, cv2.bitwise_not(fg))

        # ── 构建前景凸包掩码（用于约束暗区检测范围）────────────────────
        convex_mask = np.zeros((h, w), np.uint8)
        fg_pts = np.column_stack(np.where(fg > 0))
        if len(fg_pts) >= 10:
            hull = cv2.convexHull(fg_pts[:, ::-1].astype(np.float32))
            cv2.fillConvexPoly(convex_mask, hull.astype(np.int32), 255)
        else:
            convex_mask = fg.copy()

        # ── 通道 B：前景内暗区检测 ────────────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 用前景内像素的中位亮度估计产品正常亮度
        fg_pixels = gray[fg > 0]
        if len(fg_pixels) > 0:
            median_bright = float(np.median(fg_pixels))
            p10_bright    = float(np.percentile(fg_pixels, 10))
        else:
            median_bright = 128.0
            p10_bright    = 30.0

        # 自适应暗区阈值：不超过正常亮度的 35%，且至少比 p10 再低 15
        dark_thresh = max(min(median_bright * 0.35, p10_bright - 15.0), 12.0)

        # 在凸包内查找暗区（包括 alpha=0 区域，即已被 GrabCut 抠掉的黑洞）
        dark_mask_raw = ((gray < dark_thresh) & (convex_mask > 0)).astype(np.uint8) * 255

        # 去除纯黑背景：用 bgr 三通道最大值再筛一遍（真实破洞三通道都很暗）
        bgr_max = bgr.max(axis=2).astype(np.float32)
        truly_dark = (bgr_max < dark_thresh * 1.3).astype(np.uint8) * 255
        dark_mask_raw = cv2.bitwise_and(dark_mask_raw, truly_dark)

        # 形态学：先开运算去噪，再闭运算连接邻近暗区
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dark_mask = cv2.morphologyEx(dark_mask_raw, cv2.MORPH_OPEN,  k_open,  iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask,     cv2.MORPH_CLOSE, k_close, iterations=2)

        # ── 通道 C：高频纹理异常（破洞边缘撕裂纹）────────────────────────
        lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F, ksize=3)
        lap_energy = np.abs(lap)
        # 仅统计前景内高能量区，取 95 分位做自适应阈值
        fg_lap = lap_energy[fg > 0]
        if len(fg_lap) > 0:
            lap_thresh = float(np.percentile(fg_lap, 95)) * 1.8
        else:
            lap_thresh = 40.0
        high_freq = ((lap_energy > lap_thresh) & (convex_mask > 0)).astype(np.uint8) * 255
        # 膨胀使高频区能覆盖到相邻暗区
        high_freq_dilated = cv2.dilate(
            high_freq,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )

        # ── 合并三通道候选区域 ────────────────────────────────────────────
        # 优先合并：alpha 内洞 | 暗区且邻近撕裂纹理
        dark_with_texture = cv2.bitwise_and(dark_mask, high_freq_dilated)
        candidate = cv2.bitwise_or(hole_alpha, dark_mask)
        # 若暗区面积 > 200px 且有纹理支撑则保留，否则只保留 alpha 孔洞
        # 再做一次闭运算统一连通
        k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_final, iterations=2)

        # ── 提取边界框 ────────────────────────────────────────────────────
        best_area = 0
        min_hole_area = max(800, int(h * w * 0.002))   # 至少 0.2% 图像面积
        max_hole_ratio = 0.30                           # 不超过 30% 图像面积

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = int(cv2.contourArea(cnt))
            if area < min_hole_area:
                continue
            if bw * bh > max_hole_ratio * h * w:
                continue
            aspect = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
            if aspect > 7.0:          # 极细条状大概率是分割边缘噪声
                continue
            # 贴边过近（2% 边距）视为外轮廓噪声
            border = max(8, int(min(h, w) * 0.02))
            if x < border or y < border or x + bw > w - border or y + bh > h - border:
                # 但若面积足够大（> 1.5% 图像），即使靠边也保留（确实是大破洞）
                if bw * bh < h * w * 0.015:
                    continue
            boxes.append([x, y, x + bw, y + bh])
            best_area = max(best_area, bw * bh)

        # ── 构建热图 ──────────────────────────────────────────────────────
        hole_map = np.zeros((h, w), np.float32)
        # 暗区本身作为热图基础
        hole_map = dark_mask.astype(np.float32) / 255.0 * 0.6
        # alpha 内洞权重更高
        hole_map = np.maximum(hole_map, hole_alpha.astype(np.float32) / 255.0)
        # 框内置满
        for x1, y1, x2, y2 in boxes:
            hole_map[y1:y2, x1:x2] = np.maximum(hole_map[y1:y2, x1:x2], 0.85)

        hole_score = float(best_area / float(h * w + 1e-6)) if best_area > 0 else 0.0
        return hole_map, boxes, hole_score

    @staticmethod
    def _normalize_map(amap: np.ndarray) -> np.ndarray:
        mn, mx = float(amap.min()), float(amap.max())
        if mx - mn < 1e-6:
            return np.zeros_like(amap, dtype=np.float32)
        return ((amap - mn) / (mx - mn)).astype(np.float32)

    @staticmethod
    def _heatmap_to_boxes(
        amap_norm: np.ndarray,
        thresh: float = 0.55,
        min_area: int = 800,
        max_area_ratio: float = 0.5,
        border_margin: int = 0,
    ) -> list[list[int]]:
        """
        将归一化热图阈值化后提取异常区域边界框。

        参数
        ----
        amap_norm : 归一化热图 (H, W)，值域 0~1
        thresh    : 像素级异常阈值（高于此值视为异常）
        min_area  : 最小连通域面积（过滤噪点）

        返回
        ----
        list of [x1, y1, x2, y2]（像素坐标，与 cutout 同尺寸）
        """
        binary = (amap_norm >= thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = amap_norm.shape[:2]
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < min_area:
                continue
            if bw * bh > max_area_ratio * w * h:
                continue
            if (
                bx <= border_margin
                or by <= border_margin
                or (bx + bw) >= (w - border_margin)
                or (by + bh) >= (h - border_margin)
            ):
                continue
            boxes.append([bx, by, bx + bw, by + bh])
        return boxes

    @staticmethod
    def _make_heatmap_overlay(
        bgra: np.ndarray,
        amap_norm: np.ndarray,
    ) -> np.ndarray:
        """将归一化热图以 JET 色彩叠加到灰色背景抠图上，返回 BGR。"""
        h, w = bgra.shape[:2]
        # 背景：灰色 RGB
        if bgra.ndim == 3 and bgra.shape[2] == 4:
            alpha = bgra[:, :, 3].astype(np.float32) / 255.0
            bgr   = bgra[:, :, :3].astype(np.float32)
            bg    = np.full_like(bgr, 128, dtype=np.float32)
            base  = (alpha[..., None] * bgr + (1 - alpha[..., None]) * bg).astype(np.uint8)
        else:
            base = bgra[:, :, :3].copy() if bgra.ndim == 3 else cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)

        heat_uint8 = (amap_norm * 255).astype(np.uint8)
        heatmap    = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        alpha_blend = float(ANOMALIB_HEATMAP_ALPHA)
        overlay = cv2.addWeighted(base, 1 - alpha_blend, heatmap, alpha_blend, 0)
        return overlay

    @staticmethod
    def _second_pass_cutout(
        cutout_input: np.ndarray,
        gray_bg: int = 128,
        diff_thresh: int = 18,
        morph_ksize: int = 7,
        stage1_fg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        对 Stage 1 抠图再做一次简单 OpenCV 抠图，得到更干净的透明蒙版。

        原理
        ----
        Stage 1 的 cutout_rgba 已将非产品区域设为灰色（128,128,128）或透明。
        本方法：
          1. 把输入渲染到灰色背景 BGR；
          2. 与纯灰色背景做差分，绝对差 > diff_thresh 的像素视为"产品像素"；
          3. 形态学闭运算填充小孔，开运算去噪；
          4. 将蒙版写入 alpha 通道，返回 BGRA。

        返回
        ----
        BGRA uint8 (H, W, 4)，产品区域 alpha=255，背景 alpha=0。
        """
        h, w = cutout_input.shape[:2]

        # Stage1 前景提示：差分蒙版必须与 alpha 求交，否则厂房等非灰背景会被整幅当
        # 成前景，clean BGR 仍保留原图颜色，UI 上看起来像在用原图做检测。
        input_fg: np.ndarray | None = None
        if stage1_fg_mask is not None:
            sm = stage1_fg_mask
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
            input_fg = (sm > 127).astype(np.uint8) * 255
        elif cutout_input.ndim == 3 and cutout_input.shape[2] == 4:
            input_fg = (cutout_input[:, :, 3] > 8).astype(np.uint8) * 255

        # Step 1：渲染灰色背景 BGR
        if cutout_input.ndim == 3 and cutout_input.shape[2] == 4:
            a = cutout_input[:, :, 3:4].astype(np.float32) / 255.0
            bgr_in = cutout_input[:, :, :3].astype(np.float32)
            bg = np.full_like(bgr_in, gray_bg)
            bgr = (a * bgr_in + (1 - a) * bg).astype(np.uint8)
        elif cutout_input.ndim == 3 and cutout_input.shape[2] == 3:
            bgr = cutout_input.copy()
        else:
            bgr = cv2.cvtColor(cutout_input, cv2.COLOR_GRAY2BGR)

        # Step 2：与纯灰色背景差分
        bg_flat = np.full((h, w, 3), gray_bg, dtype=np.uint8)
        diff = cv2.absdiff(bgr, bg_flat)
        diff_gray = diff.max(axis=2)                          # (H,W) 单通道最大差
        mask_gray_bg = (diff_gray > diff_thresh).astype(np.uint8) * 255

        # 黑底图片兜底：用边角主背景色做差分，避免把黑背景误判成前景
        corner = np.vstack([
            bgr[:20, :20].reshape(-1, 3),
            bgr[:20, -20:].reshape(-1, 3),
            bgr[-20:, :20].reshape(-1, 3),
            bgr[-20:, -20:].reshape(-1, 3),
        ]).astype(np.float32)
        corner_mean = np.mean(corner, axis=0)
        diff_corner = np.max(np.abs(bgr.astype(np.float32) - corner_mean[None, None, :]), axis=2)
        mask_corner_bg = (diff_corner > max(14, diff_thresh)).astype(np.uint8) * 255

        # 若角落接近纯黑，优先使用 corner-bg 分割；否则用原灰底分割
        if float(np.mean(corner_mean)) < 28.0:
            mask = mask_corner_bg
        else:
            mask = mask_gray_bg

        # Step 3：形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        if input_fg is not None:
            refined = cv2.bitwise_and(mask, input_fg)
            min_fg = max(500.0, 0.02 * float(np.count_nonzero(input_fg)))
            if float(np.count_nonzero(refined)) >= min_fg:
                mask = refined
            else:
                # 差分几乎丢光前景时退回 Stage1 alpha（形态学再清一遍）
                mask = cv2.morphologyEx(input_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 4：按最终蒙版强制灰底 BGR，再写 alpha（避免 mask 外仍带产线颜色）
        mf = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        bg_f = np.full_like(bgr, gray_bg, dtype=np.float32)
        bgr_out = (mf * bgr.astype(np.float32) + (1.0 - mf) * bg_f).astype(np.uint8)
        clean = np.dstack([bgr_out, mask])   # (H,W,4) BGRA
        return clean

    @staticmethod
    def _empty_result(h: int, w: int) -> dict:
        return {
            "clean_cutout":     np.zeros((h, w, 4), dtype=np.uint8),
            "patchcore_score":  0.0,
            "template_score":   0.0,
            "anomaly_score":    0.0,
            "hole_score":       0.0,
            "hole_boxes":       [],
            "anomaly_map":      np.zeros((h, w), dtype=np.float32),
            "anomaly_boxes":    [],
            "heatmap_overlay":  np.full((h, w, 3), 128, dtype=np.uint8),
            "status":           "SKIP",
            "issues":           [],
        }
