"""
src/inspector_pipeline.py — 检测调度器
================================================
适用于树莓派 + AI HAT+（Hailo-8L）的流水线检测。

初筛流程（单图版）
------------------
1. 接收单张图片（来自上传或相机抓拍）
2. 运行 Stage 1（质量门控 + ORB选参考图 + 产品定位 + 抠图 + 塑料膜/姿态检测）
   - 质量不合格（过曝/欠曝/分辨率不足/覆膜过多/严重偏斜）→ RETAKE/FAIL，立刻终止
3. 对该图 product_bbox 裁剪图运行 Anomalib PatchCore 缺件检测
   - anomaly_score > 阈值 → FAIL，记录异常区域坐标（红框）
   - anomaly_score ≤ 阈值 → PASS
4. 可选：对同一张图运行 Stage 2（Hailo 划痕检测）
5. 返回完整结果字典

Pi 适配说明
-----------
- 顺序处理（不使用多线程），避免 Pi 内存/线程竞争
- 所有路径使用相对路径
- 每个阶段均打印耗时，便于 Pi 端调试
- 可选：通过 enable_stage2=False 跳过细筛（仅运行 Stage1 + Anomalib）
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
    ANOMALIB_MODEL_DIR,
    CALIB_DIR,
    HEF_PATH,
    INSPECTION_DB_PATH,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    MODELS_DIR,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage2_anomalib import AnomalibDetector
from src.stage3_yolov8 import Stage2Detector
from src.stage4_sql import Stage3SQLRecorder


# ─────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """简单时间戳日志，适合 Pi 端 stdout 监控。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# 主管道
# ─────────────────────────────────────────────────────────────────────────

class InspectorPipeline:
    """
    检测调度器。

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
        enable_anomalib: bool  = True,
        conf_threshold:  float = 0.25,
        missing_thresh:  int   = MAX_MISSING_COUNT,
        enable_sql_record: bool = True,
        db_path: str = INSPECTION_DB_PATH,
    ) -> None:
        self.view            = view
        self.calibrated_bbox = calibrated_bbox
        self.enable_stage2   = enable_stage2
        self.enable_anomalib = enable_anomalib
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

        # ── 加载 Anomalib 检测器 ──────────────────────────────────────────
        self._det_anomalib: AnomalibDetector | None = None
        if enable_anomalib:
            t0 = time.time()
            try:
                self._det_anomalib = AnomalibDetector(view=view, model_dir=ANOMALIB_MODEL_DIR)
                if self._det_anomalib.is_ready:
                    _log(f"Anomalib 模型加载完成（{view}），耗时 {time.time()-t0:.2f}s")
                else:
                    _log(f"[警告] Anomalib 模型未就绪（{view}），跳过异常检测。"
                         "请运行 python scripts/train_anomalib.py 训练。")
            except Exception as e:
                _log(f"[警告] Anomalib 初始化失败: {e}，跳过异常检测")

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
        frame: np.ndarray,
        product_id: str | None = None,
        shift: str = "A",
    ) -> dict:
        """
        对单张图片执行检测流程。

        参数
        ----
        frame : np.ndarray
            单张 BGR 图像，来自上传文件或相机抓拍。

        返回
        ----
        dict，含以下字段：
            final_status   : "PASS" | "FAIL"
            fail_reason    : str | None        — FAIL 时的人类可读原因
            fail_frame_idx : int | None        — 固定为 0（单图流程）
            inspected_idx  : int | None        — 固定为 0（单图流程）
            stage1_result  : dict | None       — 该图的 Stage1 结果
            stage2_result  : dict | None       — 该图的 Stage2 结果
            timing         : dict              — 各阶段耗时（秒）
        """
        if frame is None:
            return self._empty("未提供图片")
        result: dict = {
            "product_id":     product_id,
            "final_status":   "FAIL",
            "fail_reason":    None,
            "fail_frame_idx": None,
            "inspected_idx":  None,
            "stage1_result":  None,
            "anomalib_result": None,
            "stage2_result":  None,
            "timing":         {},
        }

        if result["product_id"] is None and self._recorder is not None:
            result["product_id"] = self._recorder.generate_product_id(shift=shift)

        # ── Step 1: 单张 Stage 1 ─────────────────────────────────────────
        _log("开始 Stage1（单张）")
        t_s1_start = time.time()
        t0 = time.time()
        r1 = self._det1.inspect_with_localization(
            frame,
            min_match_count=MIN_MATCH_COUNT,
            missing_regions_fail_count=self.missing_thresh,
            calibrated_bbox=self.calibrated_bbox,
        )
        elapsed = time.time() - t0
        result["stage1_result"] = r1
        result["inspected_idx"] = 0
        _log(f"Stage1={r1['status']}  耗时={elapsed:.2f}s")

        if r1["status"] != "PASS":
            reason = ("; ".join(r1["issues"]) or
                      f"状态={r1['status']}")
            result["fail_reason"] = f"Stage1 未通过: {reason}"
            result["fail_frame_idx"] = 0
            result["timing"]["stage1_total"] = time.time() - t_s1_start
            _log(f"Stage1 FAIL: {reason}")
            result["timing"]["total"] = time.time() - t_s1_start
            self._persist_if_needed(frame, result)
            return result

        result["timing"]["stage1_total"] = time.time() - t_s1_start
        _log(f"Stage1 通过，耗时 {result['timing']['stage1_total']:.2f}s")

        # ── Step 2: Anomalib 异常检测（单图，裁剪到 product_bbox） ─────────
        if self._det_anomalib is not None and self._det_anomalib.is_ready:
            cutout = r1.get("cutout_rgba")
            mask_crop = r1.get("product_mask_crop")
            _pb = r1.get("product_bbox")
            if cutout is not None and _pb is not None:
                _px1, _py1, _px2, _py2 = _pb
                if _px2 > _px1 and _py2 > _py1:
                    cutout = cutout[_py1:_py2, _px1:_px2]
                    # product_mask_crop 已与 product_bbox 对齐裁切
            if cutout is not None and mask_crop is not None:
                if mask_crop.shape[:2] != cutout.shape[:2]:
                    mask_crop = cv2.resize(
                        mask_crop,
                        (cutout.shape[1], cutout.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            if cutout is not None:
                _log(f"开始 Anomalib 检测（尺寸={cutout.shape[:2]}）")
                t0 = time.time()
                ra = self._det_anomalib.inspect(cutout, stage1_fg_mask=mask_crop)
                result["timing"]["anomalib"] = time.time() - t0
                result["anomalib_result"] = ra
                _log(
                    f"Anomalib={ra['status']}  "
                    f"score={ra['anomaly_score']:.3f}  "
                    f"耗时={result['timing']['anomalib']:.2f}s"
                )
                if ra["status"] == "FAIL":
                    result["final_status"] = "FAIL"
                    result["fail_reason"]  = "; ".join(ra["issues"]) or "Anomalib 检测到异常"
                    result["timing"]["total"] = (
                        result["timing"].get("stage1_total", 0.0)
                        + result["timing"].get("anomalib", 0.0)
                    )
                    self._persist_if_needed(frame, result)
                    return result
            else:
                _log("[警告] Stage1 未返回 cutout_rgba，跳过 Anomalib")

        # ── Step 3: Stage 2（单图） ──────────────────────────────────────
        if not self.enable_stage2 or self._det2 is None:
            result["final_status"] = "PASS"
            result["fail_reason"]  = None
            _log("Stage2 已禁用，综合结论: PASS")
            result["timing"]["total"] = result["timing"]["stage1_total"]
            self._persist_if_needed(frame, result)
            return result

        _log("开始 Stage2")
        t0 = time.time()
        r2 = self._det2.inspect(frame, r1)
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
        self._persist_if_needed(frame, result)
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
            "inspected_idx":  None,
            "stage1_result":  None,
            "anomalib_result": None,
            "stage2_result":  None,
            "timing":         {},
        }

    def _persist_if_needed(self, frame: np.ndarray, result: dict) -> None:
        if self._recorder is None:
            return
        product_id = result.get("product_id")
        if not product_id:
            return

        frame_rows: list[dict] = []
        s1 = result.get("stage1_result")
        if s1 is not None:
            idx = 0
            frame_rows.append(
                {
                    "frame_id": f"B{product_id}-F{idx+1}",
                    "frame_idx": idx,
                    "sharpness": None,
                    "stage1_status": s1.get("status", "SKIP"),
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
                inspected_idx=result.get("inspected_idx"),
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
# python src/inspector_pipeline.py --view back --image img.jpg
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="检测管道测试")
    ap.add_argument("--view",   required=True, help="检测视角（front/back/left/right/top）")
    ap.add_argument("--image", required=True, help="单张测试图片路径")
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
    frame = cv2.imread(args.image)
    if frame is None:
        _log(f"[警告] 无法读取图片: {args.image}")
        print("未加载到图片，退出。")
        sys.exit(1)
    # 若有标定，转换为像素坐标
    if roi_rel and calib_bbox is None:
        h, w = frame.shape[:2]
        calib_bbox = InspectorPipeline.roi_rel_to_abs(roi_rel, w, h)

    # 构建管道
    pipeline = InspectorPipeline(
        view=args.view,
        calibrated_bbox=calib_bbox,
        enable_stage2=not args.no_stage2,
    )

    # 执行检测
    t_total = time.time()
    result  = pipeline.run(frame)
    _log(f"总耗时: {time.time()-t_total:.2f}s")

    print("\n===== 结果摘要 =====")
    print(f"  综合结论   : {result['final_status']}")
    print(f"  失败原因   : {result['fail_reason']}")
    print(f"  检测帧序号 : {result['inspected_idx']}")
    print(f"  各阶段耗时 : {result['timing']}")
    if result["stage2_result"]:
        s2 = result["stage2_result"]
        print(f"  Stage2     : {s2['status']}  缺陷={len(s2['defects'])}")
