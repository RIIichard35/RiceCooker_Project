"""
Stage1 初筛检测器（视觉比对版）

完整检测流程：
  A. 图像质量门控（清晰度、分辨率、过曝、欠曝）
  B. 最佳参考图选择（ORB 多模板匹配）
  C. 产品定位（标定 ROI 优先；否则多尺度边缘模板匹配）
  D. GrabCut 精细抠图（→ 产品 mask，消除背景干扰）
  E. 塑料膜覆盖率检测（超阈值 → RETAKE）
  F. Homography 对齐（测试图 ROI 对准参考图）
  G. 差异图计算（仅在产品 mask 内）
  H. 缺件/多件框选（严格限制在产品 mask 内）
  I. 综合规则判定 → PASS / FAIL / RETAKE

所有路径使用相对路径，兼容 Windows 和树莓派。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .config import (
    DIFF_THRESHOLD,
    FILM_COVERAGE_THRESHOLD,
    FILM_S_MAX,
    FILM_V_MIN,
    MAX_MISSING_COUNT,
    MIN_DEFECT_BOX_AREA,
    MIN_LOCALIZATION_SCORE,
    MIN_MATCH_COUNT,
    MIN_RESOLUTION,
    MIN_SHARPNESS,
    OVEREXPOSE_RATIO_THRESHOLD,
    UNDEREXPOSE_RATIO_THRESHOLD,
)


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
    Stage1 初筛检测器。

    参数
    ----
    ref_folder_path : str
        标准参考图文件夹（相对或绝对路径均可）。
        文件夹内直接放 PNG/JPG/BMP 图片（同一视角）。
    """

    def __init__(self, ref_folder_path: str) -> None:
        self.reference_data: list[dict] = []
        self.orb = cv2.ORB_create(nfeatures=1500)

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
        完整初筛流程。

        参数
        ----
        target_input : str | np.ndarray
            图片路径或已加载的 BGR/灰度图。
        calibrated_bbox : [x1, y1, x2, y2] | None
            外部传入的标定 ROI（像素坐标）。传入时跳过模板匹配。

        返回
        ----
        dict，含以下关键字段：
            status          : "PASS" | "FAIL" | "RETAKE"
            issues          : list[str]
            warnings        : list[str]
            film_coverage   : float
            similarity      : float
            missing_regions : list[[x1,y1,x2,y2]]
            extra_regions   : list[[x1,y1,x2,y2]]
            annotated_image : np.ndarray
            cutout_image    : np.ndarray
            quality         : dict
            bbox            : list
            localization_score : float
            best_ref_name   : str | None
            score           : int
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

        # 过曝 / 欠曝检测
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

        # ── C. 产品定位 ────────────────────────────────────────────────
        if calibrated_bbox is not None:
            bbox = self._clamp(*calibrated_bbox, w_img, h_img)
            result["localization_score"] = 1.0
        else:
            coarse = self._multiscale_locate(target_gray, best_ref["gray"])
            result["localization_score"] = coarse["score"]
            if coarse["score"] < loc_thresh or coarse["bbox"] is None:
                result["issues"].append(
                    f"产品定位失败 (score={coarse['score']:.3f})，"
                    "建议使用 ROI 标定功能固定产品区域"
                )
                result["status"] = "RETAKE"
                result["annotated_image"] = self._put_status(vis, "RETAKE", result["issues"][-1])
                return result
            bbox = coarse["bbox"]

        result["bbox"] = bbox
        x1, y1, x2, y2 = bbox
        _bbox_from_calib = calibrated_bbox is not None   # 记录是否来自标定

        # ── D. GrabCut 精细抠图 → 产品 mask ───────────────────────────
        product_mask = self._grabcut_mask(target_bgr, bbox)
        result["cutout_image"] = self._make_white_bg(target_gray, product_mask)

        # 用 GrabCut mask 轮廓重新计算紧致 bbox
        # 注意：若 bbox 来自手动标定，则不用 GrabCut 结果覆盖，保持标定坐标
        if not _bbox_from_calib:
            cts_pm, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cts_pm:
                mc_pm = max(cts_pm, key=cv2.contourArea)
                mx, my, mw, mh = cv2.boundingRect(mc_pm)
                pad = max(4, int(min(mw, mh) * 0.02))
                mx = max(0, mx - pad);  my = max(0, my - pad)
                mw = min(w_img - mx, mw + 2 * pad)
                mh = min(h_img - my, mh + 2 * pad)
                bbox = [mx, my, mx + mw, my + mh]
                x1, y1, x2, y2 = bbox
                result["bbox"] = bbox

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

        # ── F. ROI 提取 + Homography 对齐 ─────────────────────────────
        ref_gray = best_ref["gray"]
        h_ref, w_ref = ref_gray.shape[:2]
        roi_gray = target_gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            result["issues"].append("产品 ROI 区域无效，请重新摆放")
            result["status"] = "RETAKE"
            result["annotated_image"] = vis
            return result

        roi_resized = cv2.resize(roi_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
        aligned_roi, _ = self._align_to_reference(ref_gray, roi_resized)
        result["similarity"] = float(self._simple_similarity(ref_gray, aligned_roi))

        # ── G. 差异图（仅在产品 mask 内） ─────────────────────────────
        roi_pm = product_mask[y1:y2, x1:x2]
        if roi_pm.size == 0:
            roi_pm = np.ones((y2 - y1, x2 - x1), np.uint8) * 255
        ref_product_mask = cv2.resize(
            roi_pm, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST
        )
        missing_boxes_roi, extra_boxes_roi = self._find_diff_boxes(
            ref_gray, aligned_roi, ref_product_mask, ref_product_mask
        )

        # ── H. 框坐标映射回整图，严格限制在产品 mask 内 ───────────────
        def map_to_full(bx1, by1, bx2, by2):
            rx1 = x1 + int(bx1 * (x2 - x1) / max(w_ref, 1))
            ry1 = y1 + int(by1 * (y2 - y1) / max(h_ref, 1))
            rx2 = x1 + int(bx2 * (x2 - x1) / max(w_ref, 1))
            ry2 = y1 + int(by2 * (y2 - y1) / max(h_ref, 1))
            rx1, ry1, rx2, ry2 = self._clamp(rx1, ry1, rx2, ry2, w_img, h_img)
            if (rx2 - rx1) * (ry2 - ry1) < 50:
                return None
            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            if product_mask[cy, cx] == 0:
                return None
            return [rx1, ry1, rx2, ry2]

        for idx, box in enumerate(missing_boxes_roi):
            mapped = map_to_full(*box)
            if mapped is None:
                continue
            result["missing_regions"].append(mapped)
            rx1, ry1, rx2, ry2 = mapped
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 220), 2)
            cv2.putText(
                vis, f"异常{idx+1}", (rx1, max(18, ry1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 220), 2,
            )

        for idx, box in enumerate(extra_boxes_roi):
            mapped = map_to_full(*box)
            if mapped is None:
                continue
            result["extra_regions"].append(mapped)
            rx1, ry1, rx2, ry2 = mapped
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
            cv2.putText(
                vis, f"多余{idx+1}", (rx1, max(18, ry1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 140, 255), 2,
            )

        # ── I. 综合判定 ────────────────────────────────────────────────
        if len(result["missing_regions"]) >= missing_thresh:
            result["issues"].append(
                f"发现 {len(result['missing_regions'])} 处异常区域（阈值={missing_thresh}）"
            )
        if len(result["extra_regions"]) > 0:
            result["issues"].append(
                f"发现 {len(result['extra_regions'])} 处疑似多余区域"
            )
        if result["similarity"] < min_similarity_fail:
            result["warnings"].append(
                f"与标准件相似度偏低 ({result['similarity']:.2f} < {min_similarity_fail})"
            )

        result["pass"]   = len(result["issues"]) == 0
        result["status"] = "PASS" if result["pass"] else "FAIL"
        status_color     = (0, 180, 0) if result["pass"] else (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(
            vis, "Product ROI", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2,
        )
        status_text = (
            f"Stage1 {result['status']} | "
            f"film={result['film_coverage']*100:.1f}% | "
            f"sim={result['similarity']:.2f} | "
            f"miss={len(result['missing_regions'])} | "
            f"extra={len(result['extra_regions'])}"
        )
        cv2.putText(
            vis, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2,
        )

        result["annotated_image"] = vis
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
            "pass":               False,
            "status":             "FAIL",
            "score":              0,
            "threshold":          threshold,
            "localization_score": 0.0,
            "best_ref_name":      None,
            "bbox":               None,
            "polygon":            None,
            "similarity":         0.0,
            "film_coverage":      0.0,
            "quality":            {},
            "issues":             [],
            "warnings":           [],
            "missing_regions":    [],
            "extra_regions":      [],
            "annotated_image":    None,
            "cutout_image":       None,
        }

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

    def _grabcut_mask(self, bgr: np.ndarray, bbox: list) -> np.ndarray:
        h, w            = bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh          = x2 - x1, y2 - y1

        if bw < 20 or bh < 20:
            mask = np.zeros((h, w), np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask

        roi_bgr  = bgr[y1:y2, x1:x2].copy()
        rh, rw   = roi_bgr.shape[:2]
        max_side = 512
        scale    = min(max_side / max(rw, rh), 1.0)
        roi_small = cv2.resize(roi_bgr, (int(rw * scale), int(rh * scale))) if scale < 1.0 else roi_bgr
        sh, sw   = roi_small.shape[:2]

        margin  = max(5, int(min(sw, sh) * 0.10))
        rect    = (margin, margin, sw - 2 * margin, sh - 2 * margin)
        gc_mask = np.zeros((sh, sw), np.uint8)
        bgd_mdl = np.zeros((1, 65), np.float64)
        fgd_mdl = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi_small, gc_mask, rect, bgd_mdl, fgd_mdl, 3, cv2.GC_INIT_WITH_RECT)
            fg_small = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except Exception:
            fg_small = np.ones((sh, sw), np.uint8) * 255

        fg_roi = cv2.resize(fg_small, (rw, rh), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else fg_small

        kernel = np.ones((7, 7), np.uint8)
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_roi   = np.zeros_like(fg_roi)
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_roi, [max_cnt], -1, 255, -1)
        else:
            clean_roi[:] = 255

        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y1:y2, x1:x2] = clean_roi
        return full_mask

    def _detect_film(self, bgr: np.ndarray, product_mask: np.ndarray) -> tuple[float, np.ndarray]:
        hsv          = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v      = cv2.split(hsv)
        film_pixels  = ((v > FILM_V_MIN) & (s < FILM_S_MAX)).astype(np.uint8) * 255
        film_product = cv2.bitwise_and(film_pixels, product_mask)
        prod_area    = max(int(np.count_nonzero(product_mask)), 1)
        film_area    = int(np.count_nonzero(film_product))
        return film_area / prod_area, film_product

    def _align_to_reference(self, ref_gray: np.ndarray, test_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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

    def _find_diff_boxes(
        self,
        ref_gray:  np.ndarray,
        test_gray: np.ndarray,
        ref_mask:  np.ndarray,
        test_mask: np.ndarray,
    ) -> tuple[list, list]:
        h, w     = ref_gray.shape[:2]
        area_min = max(MIN_DEFECT_BOX_AREA, int(0.0012 * w * h))

        valid_mask = cv2.bitwise_and(ref_mask, test_mask)
        abs_diff   = cv2.absdiff(ref_gray, test_gray)
        abs_diff   = cv2.bitwise_and(abs_diff, valid_mask)

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

    def _make_white_bg(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        canvas = np.full_like(gray, 255)
        canvas[mask > 0] = gray[mask > 0]
        return canvas

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
