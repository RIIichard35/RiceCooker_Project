"""
Stage1 初筛检测器（整图检测版）

新版检测流程（论文方法章节主线）：
  A. 图像质量门控（清晰度、分辨率、过曝、欠曝）
  B. 最佳参考图选择（ORB 多模板匹配）
  C. 整图产品检测（前景分离 + 边缘轮廓 + 面积/长宽比约束）
     → 无需 ROI 标定也能稳定运行；calibrated_bbox 作为可选辅助 hint
  D. GrabCut 精细抠图 → 产品 mask
  D2. 白色标签保留与补偿（高亮低饱和邻接区域回填）
  E. 塑料膜覆盖率检测（超阈值 → RETAKE）
  E2. 姿态偏斜评估（最小外接矩形法；极端偏斜 → RETAKE / warning）
  F. Homography 对齐（测试图 ROI 对准参考图）
  G. 差异图计算（产品+标签联合 mask 内，标签区域加权）
  H. 缺件/多件框选（严格限制在产品 mask 内）
  I. 综合规则判定 → PASS / FAIL / RETAKE

所有路径使用相对路径，兼容 Windows 和树莓派。
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from .config import (
    DIFF_THRESHOLD,
    FILM_COVERAGE_THRESHOLD,
    FILM_S_MAX,
    FILM_V_MIN,
    ST_DIR,
    GRABCUT_FILL_HOLES,
    GRABCUT_ITERATIONS,
    GRABCUT_KEEP_RATIO,
    GRABCUT_USE_CONVEX_HULL,
    GRABCUT_USE_MASK_INIT,
    GRABCUT_VERT_BRIDGE_PX,
    LABEL_ADAPTIVE_ENABLED,
    LABEL_BORDER_DIST_PX,
    LABEL_MIN_AREA_PX,
    LABEL_RETAIN_S_MAX,
    LABEL_RETAIN_V_MIN,
    LABEL_V_PERCENTILE,
    LABEL_WEIGHT,
    MAX_MISSING_COUNT,
    MIN_DEFECT_BOX_AREA,
    MIN_LOCALIZATION_SCORE,
    MIN_MATCH_COUNT,
    MIN_RESOLUTION,
    MIN_SHARPNESS,
    MODNET_ALPHA_THRESHOLD,
    MODNET_ALLOW_FALLBACK,
    MODNET_INPUT_SIZE,
    MODNET_MAX_AREA_RATIO,
    MODNET_MAX_COMPONENTS,
    MODNET_MIN_AREA_RATIO,
    MODNET_MIN_BOTTOM_COVERAGE,
    MODNET_MODEL_PATH,
    OVEREXPOSE_RATIO_THRESHOLD,
    POSE_SKEW_RETAKE_DEG,
    POSE_SKEW_WARN_DEG,
    REMBG_ALT_PAD_RATIO,
    REMBG_ALPHA_MATTING,
    REMBG_ALLOW_FALLBACK,
    REMBG_ALPHA_THRESHOLD,
    REMBG_FULL_RETRY_ENABLED,
    REMBG_PREFLIGHT_FULL_FRAME,
    REMBG_MAX_INPUT_SIDE,
    REMBG_MAX_AREA_RATIO,
    REMBG_MAX_COMPONENTS,
    REMBG_MIN_AREA_RATIO,
    REMBG_MIN_BOTTOM_COVERAGE,
    REMBG_MIN_HEIGHT_COVERAGE,
    REMBG_MODEL_NAME,
    REMBG_MODELS_DIR,
    REMBG_ROTATE_DEGS,
    REMBG_ROI_PAD_RATIO,
    SEGMENT_BACKEND,
    UNDEREXPOSE_RATIO_THRESHOLD,
)

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from rembg import new_session, remove
except Exception:
    new_session = None
    remove = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def imread_safe(file_path: str) -> np.ndarray | None:
    """支持中文路径的图像读取（灰度）。"""
    try:
        arr = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"[读取失败-灰度] {file_path}: {e}")
        return None


def _imread_color_safe(file_path: str) -> np.ndarray | None:
    """支持中文路径的图像读取（彩色 BGR）。"""
    try:
        arr = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[读取失败-彩色] {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# 主检测器
# ---------------------------------------------------------------------------

class Stage1Detector:
    """
    Stage1 初筛检测器（整图检测版）。

    参数
    ----
    ref_folder_path : str
        标准参考图文件夹（相对或绝对路径均可）。
        文件夹内直接放 PNG/JPG/BMP 图片（同一视角）。
    """

    def __init__(self, ref_folder_path: str) -> None:
        self.reference_data: list[dict] = []
        self.orb = cv2.ORB_create(nfeatures=1500)
        self._rembg_session: Any | None = None
        self._last_rembg_alpha: np.ndarray | None = None
        self._modnet_session: Any | None = None
        self._modnet_in_name: str | None = None
        self._modnet_out_name: str | None = None
        self._ref_folder_path: str = ""

        ref_folder_path = os.path.normpath(ref_folder_path)
        if not os.path.isdir(ref_folder_path):
            print(f"[Stage1] [错误] 参考图文件夹不存在: {ref_folder_path}")
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        print(f"[Stage1] 正在加载标准图库: {ref_folder_path} ...")
        count = 0
        for fname in sorted(os.listdir(ref_folder_path)):
            if not fname.lower().endswith(valid_exts):
                continue
            fpath = os.path.join(ref_folder_path, fname)
            gray = imread_safe(fpath)
            bgr  = _imread_color_safe(fpath)
            if gray is None or bgr is None:
                print(f"  [警告] 无法读取: {fname}")
                continue
            kp, des = self.orb.detectAndCompute(gray, None)
            if des is None:
                continue
            self.reference_data.append(
                {"name": fname, "gray": gray, "bgr": bgr, "kp": kp, "des": des}
            )
            count += 1
        self._ref_folder_path = ref_folder_path
        print(f"[Stage1] 初始化完成，成功加载 {count} 张标准图。")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def inspect_with_localization(
        self,
        target_input,
        min_match_count: int | None = None,
        min_localization_score: float | None = None,
        min_similarity_fail: float = 0.70,
        missing_regions_fail_count: int | None = None,
        calibrated_bbox: list | None = None,
    ) -> dict:
        """
        完整初筛流程（整图检测主路径）。

        参数
        ----
        target_input : str | np.ndarray
            图片路径或已加载的 BGR/灰度图。
        calibrated_bbox : [x1, y1, x2, y2] | None
            可选的标定 ROI 提示（若提供则优先使用）。
            新版主路径为整图自动检测，此参数仅作辅助 hint。

        返回
        ----
        dict，含以下关键字段：
            status            : "PASS" | "FAIL" | "RETAKE"
            issues            : list[str]
            warnings          : list[str]
            film_coverage     : float
            similarity        : float
            missing_regions   : list[[x1,y1,x2,y2]]
            extra_regions     : list[[x1,y1,x2,y2]]
            annotated_image   : np.ndarray
            cutout_image      : np.ndarray  (透明背景抠图，BGRA，兼容旧界面字段名)
            cutout_rgba       : np.ndarray  (透明背景抠图，BGRA)
            quality           : dict
            bbox              : list
            localization_score: float
            best_ref_name     : str | None
            score             : int
            pose_angle        : float       (偏斜角度，度)
            pose_warning      : str | None  (姿态告警信息)
            label_mask        : np.ndarray | None
            label_area_ratio  : float       (标签区域占产品mask比例)
        """
        threshold      = int(min_match_count or MIN_MATCH_COUNT)
        loc_thresh     = float(min_localization_score or MIN_LOCALIZATION_SCORE)
        missing_thresh = int(missing_regions_fail_count or MAX_MISSING_COUNT)

        result = self._empty_result(threshold)

        if not self.reference_data:
            result["issues"].append("标准图未加载成功，请检查标准图库路径")
            return result

        # 读取目标图
        target_gray, target_bgr = self._load_target(target_input)
        if target_gray is None:
            result["issues"].append("待测图读取失败，请检查路径或图片格式")
            return result

        h_img, w_img = target_gray.shape[:2]
        # 默认可视化底图（早退场景仍可能使用原图）
        vis = target_bgr.copy()

        # ── A. 图像质量门控 ────────────────────────────────────────────
        quality = self._evaluate_quality(target_bgr)
        result["quality"] = quality

        if quality["sharpness"] < MIN_SHARPNESS:
            result["warnings"].append(
                f"图像清晰度不足 (sharpness={quality['sharpness']:.1f})，建议重拍"
            )
        if min(h_img, w_img) < MIN_RESOLUTION:
            result["issues"].append(
                f"分辨率不足 ({w_img}×{h_img})，最低要求 {MIN_RESOLUTION}px"
            )
            result["status"] = "RETAKE"
            result["annotated_image"] = self._put_status(vis, "RETAKE", result["issues"][-1])
            return result

        if quality.get("overexpose_ratio", 0) > OVEREXPOSE_RATIO_THRESHOLD:
            msg = (
                f"图像过曝 ({quality['overexpose_ratio']*100:.1f}% 像素亮度>240)，"
                "请降低相机曝光或调整补光"
            )
            result["issues"].append(msg)
            result["status"] = "RETAKE"
            result["annotated_image"] = self._put_status(vis, "RETAKE", msg)
            return result
        if quality.get("underexpose_ratio", 0) > UNDEREXPOSE_RATIO_THRESHOLD:
            msg = (
                f"图像欠曝 ({quality['underexpose_ratio']*100:.1f}% 像素亮度<25)，"
                "请改善照明条件"
            )
            result["issues"].append(msg)
            result["status"] = "RETAKE"
            result["annotated_image"] = self._put_status(vis, "RETAKE", msg)
            return result

        # ── B. 最佳参考图选择（ORB） ───────────────────────────────────
        best_ref, match_score = self._select_best_reference(target_gray)
        result["score"] = match_score
        if best_ref is None:
            result["issues"].append("无法匹配任何参考图，请确认检测视角是否正确")
            result["status"] = "RETAKE"
            result["annotated_image"] = self._put_status(vis, "RETAKE", result["issues"][-1])
            return result
        result["best_ref_name"] = best_ref["name"]

        # 预处理透明抠图路径（st/ 目录，按视角子文件夹存放，与标准图同名）
        _ref_stem = os.path.splitext(best_ref["name"])[0]
        _st_face_dir = os.path.join(ST_DIR, os.path.basename(self._ref_folder_path))
        _st_candidates = [
            os.path.join(_st_face_dir, best_ref["name"]),
            os.path.join(_st_face_dir, _ref_stem + ".png"),
        ]
        for _c in _st_candidates:
            if os.path.isfile(_c):
                result["ref_cutout_path"] = _c
                break

        # ── C. 产品定位（整图检测主路径） ─────────────────────────────
        # 优先使用外部传入的 calibrated_bbox；否则走整图自动检测
        if calibrated_bbox is not None:
            bbox = self._clamp(*calibrated_bbox, w_img, h_img)
            result["localization_score"] = 1.0
            result["localization_method"] = "标定辅助"
        else:
            coarse = self._multiscale_locate(target_gray, best_ref["gray"])
            result["localization_score"] = coarse["score"]
            if coarse["score"] < loc_thresh or coarse["bbox"] is None:
                # 整图兜底：前景分离定位
                fallback = self._foreground_locate(target_bgr)
                if fallback is not None:
                    bbox = fallback
                    result["localization_score"] = 0.5
                    result["localization_method"] = "前景分离"
                    result["warnings"].append(
                        f"模板匹配置信度偏低 (score={coarse['score']:.3f})，"
                        "已使用前景分离作为定位兜底"
                    )
                else:
                    result["issues"].append(
                        f"产品定位失败 (score={coarse['score']:.3f})，"
                        "请检查产品摆放及光照条件"
                    )
                    result["status"] = "RETAKE"
                    result["annotated_image"] = self._put_status(
                        vis, "RETAKE", result["issues"][-1]
                    )
                    return result
            else:
                bbox = coarse["bbox"]
                result["localization_method"] = "多尺度模板"

        result["bbox"] = bbox
        x1, y1, x2, y2 = bbox

        # ── D. 主体抠图（rembg 主路径 + grabcut 回退） ────────────────
        product_mask, seg_backend, seg_reason = self._segment_product_mask(target_bgr, bbox)
        result["segmentation_backend"] = seg_backend
        result["segmentation_fallback_reason"] = seg_reason

        # 用 GrabCut mask 重新计算紧致 bbox（仅自动定位时更新）
        # 先用 _extract_main_component 过滤噪点，只保留与初始 bbox 重叠最多的主体
        if calibrated_bbox is None:
            clean_mask = self._extract_main_component(product_mask, bbox)
            if np.any(clean_mask):
                product_mask = clean_mask  # 用去噪后的 mask 替换，消除外部噪点
            ys, xs = np.where(product_mask > 0)
            if len(xs) > 0:
                mx, my = int(xs.min()), int(ys.min())
                mx2, my2 = int(xs.max()), int(ys.max())
                pad = 6   # 固定小边距，不随尺寸放大
                mx  = max(0, mx - pad)
                my  = max(0, my - pad)
                mx2 = min(w_img, mx2 + pad)
                my2 = min(h_img, my2 + pad)
                bbox = [mx, my, mx2, my2]
                x1, y1, x2, y2 = bbox
                result["bbox"] = bbox

        # ── D2. 白色标签保留与补偿 ────────────────────────────────────
        product_mask, label_mask = self._retain_labels(target_bgr, product_mask)
        label_area = int(np.count_nonzero(label_mask))
        prod_area_total = max(int(np.count_nonzero(product_mask)), 1)
        result["label_mask"] = label_mask
        result["label_area_ratio"] = label_area / prod_area_total

        # 更新抠图（包含标签区域）
        transparent_cutout = self._make_rgba_cutout(
            target_bgr,
            product_mask,
            alpha_hint=self._last_rembg_alpha,
            label_mask=label_mask,
        )
        result["cutout_image"] = transparent_cutout
        result["cutout_rgba"] = transparent_cutout
        # 关键改动：后续所有标注都画在抠图底图上，而不是原图
        vis = self._cutout_to_bgr_for_annotation(transparent_cutout)

        # ── E. 塑料膜覆盖率检测 ────────────────────────────────────────
        film_ratio, film_mask = self._detect_film(target_bgr, product_mask)
        result["film_coverage"] = float(film_ratio)

        if film_ratio > FILM_COVERAGE_THRESHOLD:
            result["issues"].append(
                f"塑料膜覆盖过大 ({film_ratio*100:.1f}% > {FILM_COVERAGE_THRESHOLD*100:.0f}%)，"
                "覆膜遮挡产品，无法进行细筛"
            )
            result["status"] = "RETAKE"
            result["annotated_image"] = self._draw_film_warning(vis, film_mask, film_ratio)
            return result
        elif film_ratio > 0.10:
            result["warnings"].append(
                f"存在塑料膜覆盖 ({film_ratio*100:.1f}%)，细筛时注意干扰"
            )

        # ── E2. 姿态偏斜评估 ──────────────────────────────────────────
        pose_angle, pose_msg = self._check_pose_skew(product_mask)
        result["pose_angle"] = pose_angle
        if pose_msg:
            result["pose_warning"] = pose_msg
            if pose_angle >= POSE_SKEW_RETAKE_DEG:
                result["issues"].append(pose_msg)
                result["status"] = "RETAKE"
                result["annotated_image"] = self._put_status(vis, "RETAKE", pose_msg)
                return result
            else:
                result["warnings"].append(pose_msg)

        # -- F. Final judgment: quality gates passed = PASS; Anomalib handles defect screening
        # ORB diff / missing-region judgment moved to Anomalib stage.
        result["pass"]   = True
        result["status"] = "PASS"
        result["similarity"] = 1.0


        # ── J. annotated_image：裁剪出产品区域，黄框围整个产品 ──────────
        # 直接裁剪到 bbox，产品充满画面，不再显示整幅灰色画布。
        # GUI 层叠加 Anomalib 异常框（坐标已对齐到裁剪图）。
        cutout_bgr_full = self._cutout_to_bgr_for_annotation(transparent_cutout)
        product_crop = cutout_bgr_full[y1:y2, x1:x2].copy()
        if product_crop.size == 0:
            product_crop = cutout_bgr_full.copy()
        ch, cw = product_crop.shape[:2]
        # 黄框围住整个裁剪图（即围住整个产品）
        cv2.rectangle(product_crop, (2, 2), (cw - 3, ch - 3), (0, 215, 255), 3)
        cv2.putText(
            product_crop, "Anomalib init",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2,
        )
        if pose_angle > 1.0:
            cv2.putText(
                product_crop, f"Skew:{pose_angle:.1f}deg",
                (cw - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 100, 255) if pose_angle >= POSE_SKEW_WARN_DEG else (180, 180, 0), 2,
            )
        result["annotated_image"] = product_crop
        result["product_bbox"] = [x1, y1, x2, y2]
        # 与 cutout_rgba 同坐标系的抠图蒙版（供 Anomalib 限制前景，避免 RGBA alpha 泄漏导致产线背景参与初筛）
        result["product_mask_crop"] = product_mask[y1:y2, x1:x2].copy()
        return result

    def check_integrity(self, target_img_path: str):
        """向后兼容接口，返回 (bool, message, debug_img)。"""
        res = self.inspect_with_localization(target_img_path)
        passed = res["status"] == "PASS"
        msg    = res["issues"][0] if res["issues"] else res["status"]
        return passed, msg, res.get("annotated_image")

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _empty_result(self, threshold: int) -> dict:
        return {
            "pass":                False,
            "status":              "FAIL",
            "score":               0,
            "threshold":           threshold,
            "localization_score":  0.0,
            "localization_method": "未定位",
            "best_ref_name":       None,
            "bbox":                None,
            "polygon":             None,
            "similarity":          0.0,
            "film_coverage":       0.0,
            "quality":             {},
            "issues":              [],
            "warnings":            [],
            "missing_regions":     [],
            "extra_regions":       [],
            "annotated_image":     None,
            "cutout_image":        None,
            "cutout_rgba":         None,
            "ref_cutout_path":     None,
            "pose_angle":          0.0,
            "pose_warning":        None,
            "label_mask":          None,
            "label_area_ratio":    0.0,
            "segmentation_backend": "unknown",
            "segmentation_fallback_reason": None,
        }

    def _segment_product_mask(self, bgr: np.ndarray, bbox: list) -> tuple[np.ndarray, str, str | None]:
        """
        统一分割入口：
          - modnet: MODNet 主路径，失败可回退
          - auto: MODNet 优先，失败回退 rembg/grabcut
          - rembg: 强制 rembg（可配置是否回退）
          - grabcut: 强制 grabcut
        """
        self._last_rembg_alpha = None
        backend = str(SEGMENT_BACKEND).strip().lower()
        if backend == "grabcut":
            return self._grabcut_mask(bgr, bbox), "grabcut", None

        if backend in ("modnet", "auto"):
            modnet_mask, reason = self._modnet_mask(bgr, bbox)
            if modnet_mask is not None:
                return modnet_mask, "modnet", None
            if backend == "modnet" and not MODNET_ALLOW_FALLBACK:
                return self._grabcut_mask(bgr, bbox), "grabcut", f"modnet-disabled-fallback:{reason}"

        if backend in ("modnet", "auto", "rembg"):
            rembg_mask, reason = self._rembg_mask(bgr, bbox)
            if rembg_mask is not None:
                return rembg_mask, "rembg", None
            if backend == "rembg" and not REMBG_ALLOW_FALLBACK:
                return self._grabcut_mask(bgr, bbox), "grabcut", f"rembg-disabled-fallback:{reason}"
            if REMBG_ALLOW_FALLBACK:
                return self._grabcut_mask(bgr, bbox), "grabcut-fallback", reason or "rembg-failed"
            return self._grabcut_mask(bgr, bbox), "grabcut", reason

        # 未知配置时兜底
        return self._grabcut_mask(bgr, bbox), "grabcut", f"invalid-backend:{backend}"

    def _modnet_mask(self, bgr: np.ndarray, bbox: list) -> tuple[np.ndarray | None, str | None]:
        """
        MODNet 主抠图：输出 alpha matte 后转二值 mask，并做主体筛选。
        """
        if ort is None:
            return None, "modnet-onnxruntime-missing"
        if not os.path.isfile(MODNET_MODEL_PATH):
            return None, "modnet-model-missing"

        sess, in_name, out_name, err = self._get_modnet_session()
        if sess is None:
            return None, err

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            return None, "bbox-too-small"

        pad_x = int(bw * REMBG_ROI_PAD_RATIO)
        pad_y = int(bh * REMBG_ROI_PAD_RATIO)
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)
        roi = bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, "roi-empty"

        matte, reason = self._run_modnet_once(roi, sess, in_name, out_name)
        if matte is None:
            return None, reason

        mask_roi = np.where(matte >= int(MODNET_ALPHA_THRESHOLD), 255, 0).astype(np.uint8)
        mask_roi = self._postprocess_mask(mask_roi)
        bbox_in_roi = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]
        mask_roi = self._extract_main_component(mask_roi, bbox_in_roi)
        if not np.any(mask_roi):
            return None, "modnet-main-component-empty"

        valid, reason = self._is_valid_modnet_mask(mask_roi, bw=max(bw, 1), bh=max(bh, 1))
        if not valid:
            return None, reason

        full_mask = np.zeros((h, w), np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = mask_roi
        return full_mask, None

    def _get_modnet_session(self) -> tuple[Any | None, str | None, str | None, str | None]:
        if self._modnet_session is not None:
            return self._modnet_session, self._modnet_in_name, self._modnet_out_name, None
        try:
            sess = ort.InferenceSession(MODNET_MODEL_PATH, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            self._modnet_session = sess
            self._modnet_in_name = in_name
            self._modnet_out_name = out_name
            return sess, in_name, out_name, None
        except Exception as exc:
            return None, None, None, f"modnet-session-init:{type(exc).__name__}"

    def _run_modnet_once(
        self,
        roi_bgr: np.ndarray,
        session: Any,
        in_name: str,
        out_name: str,
    ) -> tuple[np.ndarray | None, str | None]:
        try:
            inp, oh, ow = self._preprocess_modnet_input(roi_bgr)
            out = session.run([out_name], {in_name: inp})[0]
            matte = self._decode_modnet_output(out, oh, ow)
            return matte, None
        except Exception as exc:
            return None, f"modnet-infer:{type(exc).__name__}"

    def _preprocess_modnet_input(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
        oh, ow = roi_bgr.shape[:2]
        size = int(max(256, MODNET_INPUT_SIZE))
        resized = cv2.resize(roi_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization is robust for most MODNet ONNX exports.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        chw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
        return chw, oh, ow

    @staticmethod
    def _decode_modnet_output(output: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        matte = output
        if matte.ndim == 4:
            matte = matte[0, 0]
        elif matte.ndim == 3:
            matte = matte[0]
        matte = np.clip(matte, 0.0, 1.0)
        matte = (matte * 255.0).astype(np.uint8)
        return cv2.resize(matte, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def _rembg_mask(self, bgr: np.ndarray, bbox: list) -> tuple[np.ndarray | None, str | None]:
        """
        使用 rembg 对 bbox 邻域做多候选前景分割，返回整图最佳 mask。
        """
        if remove is None or new_session is None:
            return None, "rembg-not-installed"

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            return None, "bbox-too-small"

        try:
            if self._rembg_session is None:
                os.environ.setdefault("U2NET_HOME", REMBG_MODELS_DIR)
                self._rembg_session = new_session(REMBG_MODEL_NAME)
        except Exception as exc:
            return None, f"session-init:{type(exc).__name__}"

        # 与 scripts/make_st_transparent.py 一致：对整幅图做 rembg，再取与 bbox 相交的主连通域
        if REMBG_PREFLIGHT_FULL_FRAME:
            for angle in REMBG_ROTATE_DEGS:
                full_m, full_a, reason = self._run_rembg_once(bgr, float(angle))
                if full_m is None:
                    continue
                crop_mask = self._extract_main_component(full_m, bbox)
                if not np.any(crop_mask):
                    continue
                mcrop = crop_mask[y1:y2, x1:x2]
                valid, _ = self._is_valid_rembg_mask(mcrop, bw=max(bw, 1), bh=max(bh, 1))
                if valid:
                    self._last_rembg_alpha = full_a
                    return crop_mask, None

        # 候选扩边去重：默认只跑一次，异常场景可在 config 中调大 ALT_PAD。
        pad_ratios = list(dict.fromkeys([float(REMBG_ROI_PAD_RATIO), float(REMBG_ALT_PAD_RATIO)]))
        best_full = None
        best_alpha = None
        best_score = -1.0
        reject_reasons: list[str] = []

        for pad_ratio in pad_ratios:
            pad_x = int(bw * pad_ratio)
            pad_y = int(bh * pad_ratio)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(w, x2 + pad_x)
            ry2 = min(h, y2 + pad_y)
            roi = bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                reject_reasons.append(f"roi-empty@pad={pad_ratio:.2f}")
                continue

            for angle in REMBG_ROTATE_DEGS:
                mask_roi, alpha_roi, reason = self._run_rembg_once(roi, float(angle))
                if mask_roi is None:
                    reject_reasons.append(f"{reason}@rot={angle}")
                    continue

                bbox_in_roi = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]
                mask_roi = self._extract_main_component(mask_roi, bbox_in_roi)
                if not np.any(mask_roi):
                    reject_reasons.append(f"main-component-empty@pad={pad_ratio:.2f},rot={angle}")
                    continue

                valid, reason = self._is_valid_rembg_mask(mask_roi, bw=max(bw, 1), bh=max(bh, 1))
                score = self._score_mask(mask_roi, bw=max(bw, 1), bh=max(bh, 1))
                if not valid:
                    reject_reasons.append(f"{reason}@pad={pad_ratio:.2f},rot={angle}")
                    continue

                full_mask = np.zeros((h, w), np.uint8)
                full_mask[ry1:ry2, rx1:rx2] = mask_roi
                if score > best_score:
                    best_score = score
                    best_full = full_mask
                    full_alpha = np.zeros((h, w), np.uint8)
                    if alpha_roi is not None:
                        full_alpha[ry1:ry2, rx1:rx2] = alpha_roi
                    best_alpha = full_alpha

        # 全图 rembg 兜底：定位框明显偏移/姿态异常时，局部 ROI 可能不可靠
        if best_full is None and REMBG_FULL_RETRY_ENABLED:
            for angle in REMBG_ROTATE_DEGS:
                full_roi_mask, full_alpha, reason = self._run_rembg_once(bgr, float(angle))
                if full_roi_mask is None:
                    reject_reasons.append(f"full:{reason}@rot={angle}")
                    continue
                crop_mask = self._extract_main_component(full_roi_mask, bbox)
                if not np.any(crop_mask):
                    reject_reasons.append(f"full:main-component-empty@rot={angle}")
                    continue
                valid, reason = self._is_valid_rembg_mask(crop_mask[y1:y2, x1:x2], bw=max(bw, 1), bh=max(bh, 1))
                score = self._score_mask(crop_mask[y1:y2, x1:x2], bw=max(bw, 1), bh=max(bh, 1))
                if not valid:
                    reject_reasons.append(f"full:{reason}@rot={angle}")
                    continue
                if score > best_score:
                    best_score = score
                    best_full = crop_mask
                    best_alpha = full_alpha

        if best_full is not None:
            self._last_rembg_alpha = best_alpha
            return best_full, None
        if reject_reasons:
            return None, ";".join(reject_reasons[:4])
        return None, "rembg-no-candidate"

    def _run_rembg_once(
        self,
        roi: np.ndarray,
        angle_deg: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        """
        单次 rembg 推理，可选先旋转再逆旋转，提升姿态异常鲁棒性。
        """
        rot = roi
        rot_m = None
        h, w = roi.shape[:2]
        if abs(angle_deg) > 0.1:
            center = (w / 2.0, h / 2.0)
            rot_m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rot = cv2.warpAffine(
                roi, rot_m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

        try:
            infer_img, scale = self._resize_for_rembg(rot)
            ok, enc = cv2.imencode(".png", infer_img)
            if not ok:
                return None, None, "encode-failed"

            if REMBG_ALPHA_MATTING:
                out_bytes = remove(
                    enc.tobytes(),
                    session=self._rembg_session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                )
            else:
                out_bytes = remove(enc.tobytes(), session=self._rembg_session)

            out_arr = np.frombuffer(out_bytes, dtype=np.uint8)
            rgba = cv2.imdecode(out_arr, cv2.IMREAD_UNCHANGED)
            if rgba is None:
                return None, None, "decode-failed"
            if rgba.ndim != 3 or rgba.shape[2] < 4:
                return None, None, "alpha-missing"
        except Exception as exc:
            return None, None, f"rembg-exception:{type(exc).__name__}"

        alpha = rgba[:, :, 3]
        # 恢复原始 ROI 尺寸（推理时缩放过则还原，保证 mask 与 ROI 对齐）
        if abs(scale - 1.0) > 1e-3:
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.where(alpha >= int(REMBG_ALPHA_THRESHOLD), 255, 0).astype(np.uint8)

        # 逆旋转到原始 ROI 坐标
        if rot_m is not None:
            inv = cv2.invertAffineTransform(rot_m)
            mask = cv2.warpAffine(
                mask, inv, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            alpha = cv2.warpAffine(
                alpha, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        return self._postprocess_mask(mask), alpha, None

    @staticmethod
    def _resize_for_rembg(img: np.ndarray) -> tuple[np.ndarray, float]:
        max_side = int(REMBG_MAX_INPUT_SIDE or 0)
        if max_side <= 0:
            return img, 1.0

        h, w = img.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return img, 1.0

        scale = max_side / float(side)
        resized = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """统一 mask 后处理，服务于 rembg 输出。"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 纵向桥接，优先修复产品下半部分的断裂
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, v_kernel, iterations=1)
        return mask

    def _is_valid_modnet_mask(self, mask: np.ndarray, bw: int, bh: int) -> tuple[bool, str | None]:
        area = int(np.count_nonzero(mask))
        bbox_area = max(int(bw * bh), 1)
        ratio = area / float(bbox_area)
        if ratio < MODNET_MIN_AREA_RATIO:
            return False, f"modnet-area-low:{ratio:.3f}"
        if ratio > MODNET_MAX_AREA_RATIO:
            return False, f"modnet-area-high:{ratio:.3f}"

        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False, "modnet-empty"
        bottom_cov = int(ys.max()) / float(max(mask.shape[0] - 1, 1))
        if bottom_cov < MODNET_MIN_BOTTOM_COVERAGE:
            return False, f"modnet-bottom-low:{bottom_cov:.3f}"

        comp = self._count_components(mask, min_area=250)
        if comp > MODNET_MAX_COMPONENTS:
            return False, f"modnet-too-fragmented:{comp}"
        return True, None

    def _extract_main_component(self, mask: np.ndarray, bbox_hint: list) -> np.ndarray:
        """
        只保留产品主体连通域（强约束版）。

        规则：
        1) 优先保留与 bbox 中心核心区重叠最多的连通域
        2) 若核心区无重叠，则保留与 bbox 区域重叠最多的连通域
        3) 再次无重叠，回退到面积最大连通域
        """
        n, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return self._fill_holes(mask)

        x1, y1, x2, y2 = bbox_hint
        h, w = mask.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        core_w = max(10, int((x2 - x1) * 0.42))
        core_h = max(10, int((y2 - y1) * 0.42))
        core_x1 = max(0, cx - core_w // 2)
        core_y1 = max(0, cy - core_h // 2)
        core_x2 = min(w, cx + core_w // 2)
        core_y2 = min(h, cy + core_h // 2)

        candidates: list[tuple[int, int, int, int]] = []
        # (label, core_overlap_px, bbox_overlap_px, area_px)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 250:
                continue
            comp = (labeled == i)
            core_overlap = int(np.count_nonzero(comp[core_y1:core_y2, core_x1:core_x2]))
            bbox_overlap = int(np.count_nonzero(comp[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]))
            candidates.append((i, core_overlap, bbox_overlap, area))

        if not candidates:
            return np.zeros_like(mask)

        # 1) 中心核心区重叠优先
        candidates_core = [c for c in candidates if c[1] > 0]
        if candidates_core:
            best_idx = max(candidates_core, key=lambda t: (t[1], t[2], t[3]))[0]
        else:
            # 2) bbox 重叠优先
            candidates_bbox = [c for c in candidates if c[2] > 0]
            if candidates_bbox:
                best_idx = max(candidates_bbox, key=lambda t: (t[2], t[3]))[0]
            else:
                # 3) 回退面积最大
                best_idx = max(candidates, key=lambda t: t[3])[0]

        out = np.zeros_like(mask)
        out[labeled == best_idx] = 255
        out = self._fill_holes(out)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        return out

    def _is_valid_rembg_mask(self, mask: np.ndarray, bw: int, bh: int) -> tuple[bool, str | None]:
        """对 rembg mask 进行工程质量门控。"""
        area = int(np.count_nonzero(mask))
        bbox_area = max(int(bw * bh), 1)
        ratio = area / float(bbox_area)
        if ratio < REMBG_MIN_AREA_RATIO:
            return False, f"rembg-area-low:{ratio:.3f}"
        if ratio > REMBG_MAX_AREA_RATIO:
            return False, f"rembg-area-high:{ratio:.3f}"

        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False, "rembg-empty"
        h_cov = (int(ys.max()) - int(ys.min()) + 1) / float(max(mask.shape[0], 1))
        if h_cov < REMBG_MIN_HEIGHT_COVERAGE:
            return False, f"rembg-height-low:{h_cov:.3f}"
        bottom_cov = int(ys.max()) / float(max(mask.shape[0] - 1, 1))
        if bottom_cov < REMBG_MIN_BOTTOM_COVERAGE:
            return False, f"rembg-bottom-low:{bottom_cov:.3f}"

        comp = self._count_components(mask, min_area=250)
        if comp > REMBG_MAX_COMPONENTS:
            return False, f"rembg-too-fragmented:{comp}"
        return True, None

    def _score_mask(self, mask: np.ndarray, bw: int, bh: int) -> float:
        """
        给候选 mask 打分（越高越好），用于多候选选择最优。
        """
        area = int(np.count_nonzero(mask))
        bbox_area = max(int(bw * bh), 1)
        ratio = area / float(bbox_area)
        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return -1.0
        h_cov = (int(ys.max()) - int(ys.min()) + 1) / float(max(mask.shape[0], 1))
        comp = self._count_components(mask, min_area=250)
        bottom_cov = int(ys.max()) / float(max(mask.shape[0] - 1, 1))
        # 更偏好：覆盖更完整 + 底部保留更好 + 连通性更好 + 面积不过小
        return (
            (1.35 * h_cov)
            + (1.05 * bottom_cov)
            + (0.85 * min(ratio, 1.1))
            - (0.15 * max(comp - 1, 0))
        )

    @staticmethod
    def _count_components(mask: np.ndarray, min_area: int = 100) -> int:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                count += 1
        return count

    def _load_target(self, target_input) -> tuple[np.ndarray | None, np.ndarray | None]:
        if isinstance(target_input, str):
            gray = imread_safe(os.path.normpath(target_input))
            if gray is None:
                return None, None
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif isinstance(target_input, np.ndarray):
            if len(target_input.shape) == 2:
                gray = target_input.copy()
                bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                bgr  = target_input.copy()
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            return None, None
        return gray, bgr

    def _evaluate_quality(self, bgr: np.ndarray) -> dict:
        gray             = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharpness        = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges            = cv2.Canny(gray, 80, 180)
        texture_ratio    = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)
        hsv              = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v          = cv2.split(hsv)
        highlight        = (v > 220) & (s < 45)
        reflection_ratio = float(np.count_nonzero(highlight)) / float(highlight.size + 1e-6)
        total            = float(gray.size + 1e-6)
        overexpose_ratio  = float(np.count_nonzero(gray > 240)) / total
        underexpose_ratio = float(np.count_nonzero(gray < 25))  / total
        return {
            "sharpness":         sharpness,
            "texture_ratio":     texture_ratio,
            "reflection_ratio":  reflection_ratio,
            "overexpose_ratio":  overexpose_ratio,
            "underexpose_ratio": underexpose_ratio,
        }

    def _select_best_reference(self, target_gray: np.ndarray) -> tuple[dict | None, int]:
        kp_tgt, des_tgt = self.orb.detectAndCompute(target_gray, None)
        if des_tgt is None:
            return None, 0
        bf         = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        best_ref   = None
        best_count = 0
        for ref in self.reference_data:
            try:
                matches = bf.knnMatch(ref["des"], des_tgt, k=2)
            except cv2.error:
                continue
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) > best_count:
                best_count = len(good)
                best_ref   = ref
        return best_ref, best_count

    def _multiscale_locate(self, target_gray: np.ndarray, ref_gray: np.ndarray) -> dict:
        h_t, w_t   = target_gray.shape[:2]
        h_r0, w_r0 = ref_gray.shape[:2]
        target_edge = cv2.Canny(cv2.GaussianBlur(target_gray, (5, 5), 0), 50, 150)
        best = {"score": -1.0, "bbox": None}
        for scale in [0.5, 0.65, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]:
            w_r, h_r = int(w_r0 * scale), int(h_r0 * scale)
            if w_r < 60 or h_r < 60 or w_r >= w_t or h_r >= h_t:
                continue
            ref_resized = cv2.resize(ref_gray, (w_r, h_r), interpolation=cv2.INTER_AREA)
            ref_edge    = cv2.Canny(cv2.GaussianBlur(ref_resized, (5, 5), 0), 50, 150)
            res         = cv2.matchTemplate(target_edge, ref_edge, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best["score"]:
                x1, y1 = max_loc
                x2, y2 = x1 + w_r, y1 + h_r
                best   = {
                    "score": float(max_val),
                    "bbox":  self._clamp(x1, y1, x2, y2, w_t, h_t),
                }
        return best

    def _foreground_locate(self, bgr: np.ndarray) -> list | None:
        """
        前景分离兜底定位：边缘+形态学，找最大前景轮廓。
        用于模板匹配置信度过低时的保底产品定位。
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 自适应阈值 + 形态学
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 过滤：面积 > 图像 5%，长宽比 < 5
        valid = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            ratio = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
            if area > 0.05 * w * h and ratio < 5:
                valid.append(cnt)

        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        pad = max(10, int(min(bw, bh) * 0.03))
        return self._clamp(bx - pad, by - pad, bx + bw + pad, by + bh + pad, w, h)

    def _grabcut_mask(self, bgr: np.ndarray, bbox: list) -> np.ndarray:
        """
        增强版 GrabCut 抠图。

        改进点（解决产品下半部分或标签被当背景剔除的问题）：
        1. Mask 初始化：内部标为可能前景，外边缘标为确定背景，
           比单纯 RECT 初始化更稳定
        2. 后处理保留“多连通域”：只要面积 ≥ 最大域 × GRABCUT_KEEP_RATIO 就保留，
           避免产品被阴影/反光拆成几块后只留最大块
        3. 纵向桥接闭运算：用高瘦核修复上下断裂
        4. 孔洞填充：防止标签或内嵌结构被当成背景
        5. 可选凸包兜底：对接近凸形状的产品能彻底避免漏
        """
        h, w           = bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh         = x2 - x1, y2 - y1

        if bw < 20 or bh < 20:
            mask = np.zeros((h, w), np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask

        roi_bgr  = bgr[y1:y2, x1:x2].copy()
        rh, rw   = roi_bgr.shape[:2]
        max_side = 512
        scale    = min(max_side / max(rw, rh), 1.0)
        roi_small = (
            cv2.resize(roi_bgr, (int(rw * scale), int(rh * scale)))
            if scale < 1.0 else roi_bgr
        )
        sh, sw = roi_small.shape[:2]

        bgd_mdl = np.zeros((1, 65), np.float64)
        fgd_mdl = np.zeros((1, 65), np.float64)

        fg_small = None
        if GRABCUT_USE_MASK_INIT:
            gc_mask = self._build_grabcut_init_mask(roi_small)
            try:
                cv2.grabCut(
                    roi_small, gc_mask, None, bgd_mdl, fgd_mdl,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK,
                )
                fg_small = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except Exception:
                fg_small = None

        if fg_small is None:
            # RECT 兜底（原逻辑）
            margin  = max(5, int(min(sw, sh) * 0.08))
            rect    = (margin, margin, sw - 2 * margin, sh - 2 * margin)
            gc_mask = np.zeros((sh, sw), np.uint8)
            try:
                cv2.grabCut(
                    roi_small, gc_mask, rect, bgd_mdl, fgd_mdl,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT,
                )
                fg_small = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except Exception:
                fg_small = np.ones((sh, sw), np.uint8) * 255

        fg_roi = (
            cv2.resize(fg_small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            if scale < 1.0 else fg_small
        )

        # ── 后处理 ────────────────────────────────────────────
        # 1) 椭圆核形态学（比方核更贴合真实物体边缘）
        k_size = max(5, int(min(rw, rh) * 0.015))
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        ell_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_CLOSE, ell_kernel, iterations=2)
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN,  ell_kernel, iterations=1)

        # 2) 纵向桥接（修复上下断裂）
        vk = max(5, int(min(rw, rh) * (GRABCUT_VERT_BRIDGE_PX / 512.0) * 2))
        vk = vk + 1 if vk % 2 == 0 else vk
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vk))
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_CLOSE, vert_kernel, iterations=1)

        # 3) 保留主要连通域（非单一最大域）
        clean_roi = self._keep_top_components(fg_roi, min_ratio=GRABCUT_KEEP_RATIO)
        if not np.any(clean_roi):
            clean_roi = fg_roi  # 全部被过滤时回退

        # 4) 孔洞填充
        if GRABCUT_FILL_HOLES:
            clean_roi = self._fill_holes(clean_roi)

        # 5) 凸包兜底（可选）
        if GRABCUT_USE_CONVEX_HULL:
            clean_roi = self._convex_hull_mask(clean_roi)

        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y1:y2, x1:x2] = clean_roi
        return full_mask

    # ------------------------------------------------------------------
    # GrabCut 相关辅助
    # ------------------------------------------------------------------

    def _build_grabcut_init_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        基于启发式线索构建 GrabCut 初始 mask。

        规则：
          - 最外层 5% 边缘 → 确定背景 (GC_BGD)
          - 外层 5%~15%      → 可能背景 (GC_PR_BGD)
          - 中心 50% 区域    → 可能前景 (GC_PR_FGD)
          - 与四角平均色差异大的像素 → 更可能前景
        """
        sh, sw = roi_bgr.shape[:2]
        gc = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)

        # 中心 80% 区域 → 可能前景（扩大覆盖，减少边缘被误判为背景）
        mx1 = int(sw * 0.10);  my1 = int(sh * 0.10)
        mx2 = int(sw * 0.90);  my2 = int(sh * 0.90)
        gc[my1:my2, mx1:mx2] = cv2.GC_PR_FGD

        # 最外层 2% → 确定背景（收窄背景带，防止产品边缘被强制标为背景）
        bd = max(2, int(min(sw, sh) * 0.02))
        gc[:bd, :] = cv2.GC_BGD
        gc[-bd:, :] = cv2.GC_BGD
        gc[:, :bd] = cv2.GC_BGD
        gc[:, -bd:] = cv2.GC_BGD

        # 颜色提示：与四角平均色差异大的像素更可能是前景
        try:
            corners = np.concatenate([
                roi_bgr[:bd, :bd].reshape(-1, 3),
                roi_bgr[:bd, -bd:].reshape(-1, 3),
                roi_bgr[-bd:, :bd].reshape(-1, 3),
                roi_bgr[-bd:, -bd:].reshape(-1, 3),
            ], axis=0).astype(np.float32)
            corner_mean = np.mean(corners, axis=0)
            diff = np.linalg.norm(
                roi_bgr.astype(np.float32) - corner_mean, axis=2
            )
            # 扩展到整个初始前景区，颜色差距显著的标为可能前景
            strong_fg = diff > 45
            gc[strong_fg & (gc != cv2.GC_BGD)] = cv2.GC_PR_FGD
        except Exception:
            pass

        return gc

    def _keep_top_components(self, mask: np.ndarray, min_ratio: float = 0.15) -> np.ndarray:
        """
        保留面积 ≥ 最大连通域 × min_ratio 的所有连通域。

        避免只保留最大块导致产品上下/左右被拆分后丢失。
        同时用图像总像素 * 0.002 作为绝对面积下限，过滤真正的噪点。
        """
        n, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask
        max_area = int(areas.max())
        ratio_thresh = max(int(max_area * min_ratio), 200)
        # 绝对面积下限：至少占图像 0.2%，过滤远离主体的小碎块
        h, w = mask.shape[:2]
        abs_thresh = int(h * w * 0.002)
        threshold = max(ratio_thresh, abs_thresh)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= threshold:
                out[labeled == i] = 255
        return out

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """
        通过从外部泛洪填充的方式修补 mask 内部的孔洞。
        """
        h, w = mask.shape[:2]
        flood = mask.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(mask, holes)

    @staticmethod
    def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
        """对 mask 的所有轮廓点取凸包（适合大致凸形产品兜底）。"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        out = np.zeros_like(mask)
        cv2.drawContours(out, [hull], -1, 255, -1)
        return out

    def _retain_labels(
        self, bgr: np.ndarray, product_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        白色标签保留与补偿（增强版）。

        改进点：
        1. 自适应阈值：用产品附近区域 V 通道的分位数作为亮度阈值，
           偏灰白的标签（V≈170~200）也能被纳入
        2. 同时检测“产品 mask 内部的标签”和“紧邻产品的外部标签”
        3. 用 Lab 色彩空间作第二判据：Lab 的 b 通道 ≈ 0 的低饱和区
        4. 扩大距离阈值，应对贴在产品边缘外突出的长条形标签

        返回
        ----
        (enhanced_mask, label_mask)
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # ── 阈值计算（自适应） ─────────────────────────────────────
        v_thresh = LABEL_RETAIN_V_MIN
        s_thresh = LABEL_RETAIN_S_MAX
        if LABEL_ADAPTIVE_ENABLED and np.count_nonzero(product_mask) > 1000:
            # 在产品 mask 邻域（含 mask 内 + 外扩 2×LABEL_BORDER_DIST_PX）计算 V 通道分位数
            kernel_size = max(15, LABEL_BORDER_DIST_PX)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (kernel_size, kernel_size))
            roi_mask = cv2.dilate(product_mask, dilate_k, iterations=1)
            v_pixels = v[roi_mask > 0]
            if v_pixels.size > 0:
                adaptive_v = int(np.percentile(v_pixels, LABEL_V_PERCENTILE))
                # 自适应阈值不能比静态下限更激进
                v_thresh = max(LABEL_RETAIN_V_MIN, min(adaptive_v, 230))

        # ── 候选像素：高亮 + 低饱和 ────────────────────────────────
        label_candidate = (
            (v.astype(np.int32) >= v_thresh) &
            (s.astype(np.int32) <= s_thresh)
        ).astype(np.uint8) * 255

        # 用 Lab 色彩空间再做一次过滤（白色标签的 a、b 应接近 128）
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            _, a_ch, b_ch = cv2.split(lab)
            whiteness = (
                (np.abs(a_ch.astype(np.int32) - 128) <= 18) &
                (np.abs(b_ch.astype(np.int32) - 128) <= 22)
            ).astype(np.uint8) * 255
            label_candidate = cv2.bitwise_and(label_candidate, whiteness)
        except Exception:
            pass

        # ── 产品 mask 内部的标签（贴在产品表面） ──────────────────
        inner_label = cv2.bitwise_and(label_candidate, product_mask)

        # ── 产品 mask 外部、但紧邻产品的候选 ──────────────────────
        outside = cv2.bitwise_and(label_candidate, cv2.bitwise_not(product_mask))
        inverted_mask = cv2.bitwise_not(product_mask)
        dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        near_border = (dist <= float(LABEL_BORDER_DIST_PX)).astype(np.uint8) * 255
        adjacent = cv2.bitwise_and(outside, near_border)

        # ── 连通域筛选 ────────────────────────────────────────────
        outer_mask = self._filter_label_components(adjacent, LABEL_MIN_AREA_PX)
        inner_clean = self._filter_label_components(
            inner_label, max(LABEL_MIN_AREA_PX // 2, 150)
        )

        # 形态学平滑
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        inner_clean = cv2.morphologyEx(inner_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 限制外部标签扩展不超出主体凸包，防止撑大 bbox
        convex_bound = self._convex_hull_mask(product_mask)
        # 适度膨胀凸包（允许贴边标签有少量余量）
        hull_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        convex_bound = cv2.dilate(convex_bound, hull_kernel, iterations=1)
        outer_mask = cv2.bitwise_and(outer_mask, convex_bound)

        full_label_mask = cv2.bitwise_or(inner_clean, outer_mask)
        enhanced_mask   = cv2.bitwise_or(product_mask, outer_mask)

        return enhanced_mask, full_label_mask

    @staticmethod
    def _filter_label_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """按面积过滤 mask 中的连通域。"""
        if not np.any(mask):
            return mask
        n, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labeled == i] = 255
        return out

    def _check_pose_skew(self, product_mask: np.ndarray) -> tuple[float, str | None]:
        """
        姿态偏斜评估（最小外接矩形法）。

        对产品 mask 的最大轮廓拟合最小外接矩形，
        计算偏斜角度并给出分级告警。

        返回
        ----
        (angle_deg, warning_msg or None)
        """
        contours, _ = cv2.findContours(
            product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1000:
            return 0.0, None

        rect  = cv2.minAreaRect(largest)
        angle = float(abs(rect[2]))  # OpenCV 返回 -90 ~ 0°

        # 归一化到 0~45° 范围
        if angle > 45.0:
            angle = 90.0 - angle

        if angle >= POSE_SKEW_RETAKE_DEG:
            msg = (
                f"产品摆放偏斜过大（偏斜角≈{angle:.1f}°，阈值={POSE_SKEW_RETAKE_DEG:.0f}°），"
                "请重新摆正产品后重拍"
            )
            return angle, msg
        elif angle >= POSE_SKEW_WARN_DEG:
            msg = (
                f"产品轻微偏斜（偏斜角≈{angle:.1f}°），不影响检测，建议摆正"
            )
            return angle, msg

        return angle, None

    def _detect_film(self, bgr: np.ndarray, product_mask: np.ndarray) -> tuple[float, np.ndarray]:
        hsv          = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v      = cv2.split(hsv)
        film_pixels  = ((v > FILM_V_MIN) & (s < FILM_S_MAX)).astype(np.uint8) * 255
        film_product = cv2.bitwise_and(film_pixels, product_mask)
        prod_area    = max(int(np.count_nonzero(product_mask)), 1)
        film_area    = int(np.count_nonzero(film_product))
        return film_area / prod_area, film_product

    def _align_to_reference(
        self, ref_gray: np.ndarray, test_gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        kp1, des1 = self.orb.detectAndCompute(ref_gray,  None)
        kp2, des2 = self.orb.detectAndCompute(test_gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return test_gray, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw  = bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return test_gray, None
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return test_gray, None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _    = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return test_gray, None
        h, w    = ref_gray.shape[:2]
        aligned = cv2.warpPerspective(test_gray, H, (w, h))
        return aligned, H

    def _find_diff_boxes_weighted(
        self,
        ref_gray:   np.ndarray,
        test_gray:  np.ndarray,
        ref_mask:   np.ndarray,
        test_mask:  np.ndarray,
        label_mask: np.ndarray | None = None,
    ) -> tuple[list, list]:
        """
        差异图计算（支持标签区域加权）。
        标签区域的差异乘以 LABEL_WEIGHT 放大，
        确保标签缺失/损坏能更灵敏地触发检出。
        """
        h, w     = ref_gray.shape[:2]
        area_min = max(MIN_DEFECT_BOX_AREA, int(0.0012 * w * h))

        valid_mask = cv2.bitwise_and(ref_mask, test_mask)
        abs_diff   = cv2.absdiff(ref_gray, test_gray)
        abs_diff   = cv2.bitwise_and(abs_diff, valid_mask)

        # 标签区域加权
        if label_mask is not None and np.any(label_mask > 0):
            label_in_valid = cv2.bitwise_and(label_mask, valid_mask)
            label_float    = label_in_valid.astype(np.float32) / 255.0
            abs_diff_f     = abs_diff.astype(np.float32)
            abs_diff_f     = abs_diff_f * (1.0 + (LABEL_WEIGHT - 1.0) * label_float)
            abs_diff       = np.clip(abs_diff_f, 0, 255).astype(np.uint8)

        _, diff_bin = cv2.threshold(
            abs_diff, int(DIFF_THRESHOLD * 255), 255, cv2.THRESH_BINARY
        )

        k_open  = np.ones((11, 11), np.uint8)
        k_close = np.ones((15, 15), np.uint8)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN,  k_open,  iterations=1)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k_close, iterations=2)

        def _mask_to_boxes(mask: np.ndarray) -> list:
            conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in conts:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                if area < area_min:
                    continue
                ratio = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
                if ratio > 8:
                    continue
                if area > int(0.35 * w * h):
                    continue
                boxes.append([bx, by, bx + bw, by + bh])
            return boxes

        return _mask_to_boxes(diff_bin), []

    def _make_rgba_cutout(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        alpha_hint: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if alpha_hint is not None and alpha_hint.shape[:2] == mask.shape[:2]:
            alpha = alpha_hint.copy()
            alpha[mask == 0] = 0
        else:
            alpha = np.zeros(mask.shape[:2], dtype=np.uint8)
            alpha[mask > 0] = 255

        if label_mask is not None and label_mask.shape[:2] == mask.shape[:2]:
            alpha[label_mask > 0] = 255

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    @staticmethod
    def _cutout_to_bgr_for_annotation(cutout_bgra: np.ndarray) -> np.ndarray:
        """
        将 BGRA 抠图转换为用于画框的 BGR 底图。
        透明区域统一铺灰底，避免回退到原图可视化。
        """
        if cutout_bgra.ndim == 3 and cutout_bgra.shape[2] == 4:
            # uint32 按通道混合，避免同时持有多幅 float32 H×W×3（低内存环境易 OOM）
            a = cutout_bgra[:, :, 3].astype(np.uint32, copy=False)
            out = np.empty((*cutout_bgra.shape[:2], 3), dtype=np.uint8)
            k128 = 128 * (255 - a)
            for c in range(3):
                ch = cutout_bgra[:, :, c].astype(np.uint32, copy=False)
                out[:, :, c] = np.minimum((ch * a + k128) // 255, 255).astype(np.uint8)
            return out
        if cutout_bgra.ndim == 2:
            return cv2.cvtColor(cutout_bgra, cv2.COLOR_GRAY2BGR)
        return cutout_bgra.copy()

    def _draw_film_warning(self, vis: np.ndarray, film_mask: np.ndarray, ratio: float) -> np.ndarray:
        out     = vis.copy()
        overlay = np.zeros_like(out)
        overlay[film_mask > 0] = (0, 165, 255)
        out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)
        cv2.putText(
            out,
            f"RETAKE: 塑料膜覆盖 {ratio*100:.1f}%（超过阈值 {FILM_COVERAGE_THRESHOLD*100:.0f}%）",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2,
        )
        return out

    def _put_status(self, vis: np.ndarray, status: str, msg: str) -> np.ndarray:
        out   = vis.copy()
        color = (0, 0, 255) if status != "PASS" else (0, 180, 0)
        cv2.putText(out, f"{status}: {msg}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return out

    def _clamp(self, x1, y1, x2, y2, w, h) -> list:
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        return [x1, y1, x2, y2]

    def _simple_similarity(self, ref: np.ndarray, test: np.ndarray) -> float:
        if ref.shape != test.shape:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
        return max(0.0, 1.0 - float(np.mean(cv2.absdiff(ref, test))) / 255.0)
