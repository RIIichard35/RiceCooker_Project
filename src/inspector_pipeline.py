"""
src/inspector_pipeline.py — 方案 D 检测调度器
================================================
适用于树莓派 + AI HAT+（Hailo-8L）的流水线检测。

方案 D 流程
-----------
1. 接收三连拍帧列表（来自 trigger.py 的 capture_burst()）
2. 对三张图各跑一次 Stage 1（结构初筛）
3. 有任意一张 FAIL / RETAKE → 立刻返回 FAIL，记录出问题的帧序号和原因
4. 三张全部 PASS → 选最清晰一张（Laplacian 方差最大）
5. 对最清晰张跑 Stage 2（Hailo 划痕检测）
6. 返回完整结果字典

Pi 适配说明
-----------
- 顺序处理（不使用多线程），避免 Pi 内存/线程竞争
- 所有路径使用相对路径
- 每个阶段均打印耗时，便于 Pi 端调试
- 可选：通过 enable_stage2=False 跳过细筛（仅运行结构检测）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── 路径（兼容直接运行和作为模块导入） ──────────────────────────────────
_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CALIB_DIR,
    HEF_PATH,
    INSPECTION_DB_PATH,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    MODELS_DIR,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage2_scratch import Stage2Detector
from src.stage3_sql import Stage3SQLRecorder


# ─────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────

def _laplacian_var(img_bgr: np.ndarray) -> float:
    """Laplacian 方差，数值越大越清晰。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _log(msg: str) -> None:
    """简单时间戳日志，适合 Pi 端 stdout 监控。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# 主管道
# ─────────────────────────────────────────────────────────────────────────

class InspectorPipeline:
    """
    方案 D 检测调度器。

    参数
    ----
    view           : str   — 检测视角，对应 assets/standards/<view>/ 目录
                             （"front" / "back" / "left" / "right" / "top"）
    calibrated_bbox: list | None — [x1, y1, x2, y2] 像素坐标的固定 ROI；
                             None 表示自动定位（慢）
    enable_stage2  : bool  — 是否启用 Stage 2 细筛（默认 True）
    conf_threshold : float — Stage 2 划痕检测置信度阈值（默认 0.25）
    missing_thresh : int   — Stage 1 异常框超过此数判为 FAIL（默认同 config）
    """

    def __init__(
        self,
        view:            str,
        calibrated_bbox: list | None = None,
        enable_stage2:   bool  = True,
        conf_threshold:  float = 0.25,
        missing_thresh:  int   = MAX_MISSING_COUNT,
        enable_sql_record: bool = True,
        db_path: str = INSPECTION_DB_PATH,
    ) -> None:
        self.view            = view
        self.calibrated_bbox = calibrated_bbox
        self.enable_stage2   = enable_stage2
        self.conf_threshold  = conf_threshold
        self.missing_thresh  = missing_thresh
        self.enable_sql_record = enable_sql_record
        self._recorder: Stage3SQLRecorder | None = None

        if self.enable_sql_record:
            self._recorder = Stage3SQLRecorder(db_path)
            _log(f"Stage3 SQL 记录已启用: {db_path}")

        # ── 加载 Stage 1 ─────────────────────────────────────────────────
        std_dir = Path(STANDARDS_DIR) / view
        if not std_dir.exists():
            raise FileNotFoundError(f"[Pipeline] 找不到标准图库: {std_dir}")
        t0 = time.time()
        self._det1 = Stage1Detector(str(std_dir))
        _log(f"Stage1 标准图库加载完成（{view}），耗时 {time.time()-t0:.2f}s")

        # ── 加载 Stage 2（如果启用） ──────────────────────────────────────
        self._det2: Stage2Detector | None = None
        if enable_stage2:
            onnx_path = Path(MODELS_DIR) / "best20240919.onnx"
            hef_path  = Path(HEF_PATH)
            if not onnx_path.exists() and not hef_path.exists():
                _log(f"[警告] 找不到 Stage2 模型（{onnx_path.name} 或 .hef），细筛已禁用")
                self.enable_stage2 = False
            else:
                t0 = time.time()
                self._det2 = Stage2Detector(
                    model_path=str(onnx_path),
                    hef_path=str(hef_path) if hef_path.exists() else None,
                    conf_threshold=conf_threshold,
                )
                _log(f"Stage2 模型加载完成，耗时 {time.time()-t0:.2f}s")

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def run(
        self,
        frames: list[np.ndarray],
        product_id: str | None = None,
        shift: str = "A",
    ) -> dict:
        """
        对三连拍帧执行方案 D 检测。

        参数
        ----
        frames : list[np.ndarray]
            三张（或更多）BGR 图像，来自 ProductTrigger.capture_burst()。
            至少需要 1 张；建议 3 张。

        返回
        ----
        dict，含以下字段：
            final_status   : "PASS" | "FAIL"
            fail_reason    : str | None        — FAIL 时的人类可读原因
            fail_frame_idx : int | None        — 哪一帧导致 FAIL（0-based）
            sharpest_idx   : int | None        — 最清晰帧序号（Stage2 使用）
            stage1_results : list[dict]        — 每张的 Stage1 结果
            stage2_result  : dict | None       — 最清晰帧的 Stage2 结果
            timing         : dict              — 各阶段耗时（秒）
        """
        if not frames:
            return self._empty("未提供任何帧")

        n = len(frames)
        result: dict = {
            "product_id":     product_id,
            "final_status":   "FAIL",
            "fail_reason":    None,
            "fail_frame_idx": None,
            "sharpest_idx":   None,
            "stage1_results": [],
            "stage2_result":  None,
            "timing":         {},
        }

        if result["product_id"] is None and self._recorder is not None:
            result["product_id"] = self._recorder.generate_product_id(shift=shift)

        # ── Step 1: 逐张跑 Stage 1 ───────────────────────────────────────
        _log(f"开始 Stage1 × {n} 张")
        t_s1_start = time.time()

        for i, frame in enumerate(frames):
            t0 = time.time()
            r1 = self._det1.inspect_with_localization(
                frame,
                min_match_count=MIN_MATCH_COUNT,
                missing_regions_fail_count=self.missing_thresh,
                calibrated_bbox=self.calibrated_bbox,
            )
            elapsed = time.time() - t0
            result["stage1_results"].append(r1)
            _log(f"  帧[{i}] Stage1={r1['status']}  耗时={elapsed:.2f}s")

            if r1["status"] != "PASS":
                # 任意一张不通过 → 立刻终止
                reason = ("; ".join(r1["issues"]) or
                          f"帧{i} 状态={r1['status']}")
                result["fail_reason"]    = f"帧[{i}] Stage1 未通过: {reason}"
                result["fail_frame_idx"] = i
                result["timing"]["stage1_total"] = time.time() - t_s1_start
                _log(f"Stage1 FAIL at 帧[{i}]: {reason}")
                result["timing"]["total"] = time.time() - t_s1_start
                self._persist_if_needed(frames, result)
                return result

        result["timing"]["stage1_total"] = time.time() - t_s1_start
        _log(f"Stage1 全部通过，共耗时 {result['timing']['stage1_total']:.2f}s")

        # ── Step 2: 选最清晰帧 ───────────────────────────────────────────
        sharpness = [_laplacian_var(f) for f in frames]
        best_idx  = int(np.argmax(sharpness))
        result["sharpest_idx"] = best_idx
        _log(f"清晰度得分: {[f'{s:.1f}' for s in sharpness]}，选帧[{best_idx}]")

        # ── Step 3: Stage 2（仅对最清晰帧） ─────────────────────────────
        if not self.enable_stage2 or self._det2 is None:
            result["final_status"] = "PASS"
            result["fail_reason"]  = None
            _log("Stage2 已禁用，综合结论: PASS")
            result["timing"]["total"] = result["timing"]["stage1_total"]
            self._persist_if_needed(frames, result)
            return result

        _log(f"开始 Stage2（帧[{best_idx}]）")
        t0 = time.time()
        r2 = self._det2.inspect(frames[best_idx], result["stage1_results"][best_idx])
        result["timing"]["stage2"] = time.time() - t0
        result["stage2_result"]    = r2
        _log(f"Stage2={r2['status']}  耗时={result['timing']['stage2']:.2f}s")

        if r2["status"] == "PASS":
            result["final_status"] = "PASS"
        else:
            result["final_status"] = "FAIL"
            result["fail_reason"]  = "; ".join(r2["issues"]) or "Stage2 检测到表面缺陷"

        result["timing"]["total"] = (
            result["timing"].get("stage1_total", 0.0) +
            result["timing"].get("stage2", 0.0)
        )
        self._persist_if_needed(frames, result)
        _log(f"综合结论: {result['final_status']}")
        return result

    # ── 静态工具 ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty(reason: str) -> dict:
        return {
            "product_id":     None,
            "final_status":   "FAIL",
            "fail_reason":    reason,
            "fail_frame_idx": None,
            "sharpest_idx":   None,
            "stage1_results": [],
            "stage2_result":  None,
            "timing":         {},
        }

    def _persist_if_needed(self, frames: list[np.ndarray], result: dict) -> None:
        if self._recorder is None:
            return
        product_id = result.get("product_id")
        if not product_id:
            return

        sharpness_scores = [_laplacian_var(f) for f in frames] if frames else []
        frame_rows: list[dict] = []
        stage1_results = result.get("stage1_results", [])
        for idx, s1 in enumerate(stage1_results):
            frame_rows.append(
                {
                    "frame_id": f"B{product_id}-F{idx+1}",
                    "frame_idx": idx,
                    "sharpness": sharpness_scores[idx] if idx < len(sharpness_scores) else None,
                    "stage1_status": s1.get("status"),
                    "stage1_issues": s1.get("issues", []),
                }
            )

        try:
            self._recorder.save_inspection(
                product_id=product_id,
                view=self.view,
                frame_rows=frame_rows,
                stage2_result=result.get("stage2_result"),
                final_status=result.get("final_status", "FAIL"),
                fail_reason=result.get("fail_reason"),
                sharpest_idx=result.get("sharpest_idx"),
                timing=result.get("timing"),
            )
            _log(f"Stage3 SQL 已写入: product_id={product_id}")
        except Exception as e:
            _log(f"[警告] Stage3 SQL 写入失败: {e}")

    @staticmethod
    def load_calibration(view: str) -> list | None:
        """
        从 assets/calibration/<view>.json 读取 ROI 相对坐标，
        返回 [x1r, y1r, x2r, y2r]（0~1），未标定则返回 None。
        """
        import json
        p = Path(CALIB_DIR) / f"{view}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text("utf-8")).get("roi_rel")
        except Exception:
            return None

    @staticmethod
    def roi_rel_to_abs(roi_rel: list, w: int, h: int) -> list:
        """相对坐标 → 像素坐标。"""
        return [
            int(roi_rel[0] * w), int(roi_rel[1] * h),
            int(roi_rel[2] * w), int(roi_rel[3] * h),
        ]


# ─────────────────────────────────────────────────────────────────────────
# 简单命令行测试入口（Pi 上可直接跑）
# python src/inspector_pipeline.py --view back --images img1.jpg img2.jpg img3.jpg
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="方案D检测管道测试")
    ap.add_argument("--view",   required=True, help="检测视角（front/back/left/right/top）")
    ap.add_argument("--images", nargs="+", required=True, help="三张测试图片路径")
    ap.add_argument("--no-stage2", action="store_true", help="跳过 Stage2")
    args = ap.parse_args()

    # 读取标定
    roi_rel = InspectorPipeline.load_calibration(args.view)
    calib_bbox = None
    if roi_rel:
        _log(f"ROI 已标定: {roi_rel}")
    else:
        _log("ROI 未标定，将自动定位（较慢）")

    # 加载图片
    frames = []
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            _log(f"[警告] 无法读取图片: {p}")
            continue
        # 若有标定，转换为像素坐标
        if roi_rel and calib_bbox is None:
            h, w = img.shape[:2]
            calib_bbox = InspectorPipeline.roi_rel_to_abs(roi_rel, w, h)
        frames.append(img)

    if not frames:
        print("未加载到任何图片，退出。")
        sys.exit(1)

    # 构建管道
    pipeline = InspectorPipeline(
        view=args.view,
        calibrated_bbox=calib_bbox,
        enable_stage2=not args.no_stage2,
    )

    # 执行检测
    t_total = time.time()
    result  = pipeline.run(frames)
    _log(f"总耗时: {time.time()-t_total:.2f}s")

    print("\n===== 结果摘要 =====")
    print(f"  综合结论   : {result['final_status']}")
    print(f"  失败原因   : {result['fail_reason']}")
    print(f"  最清晰帧   : {result['sharpest_idx']}")
    print(f"  各阶段耗时 : {result['timing']}")
    if result["stage2_result"]:
        s2 = result["stage2_result"]
        print(f"  Stage2     : {s2['status']}  缺陷={len(s2['defects'])}")
