"""
产品到位触发模块 (src/trigger.py)
====================================
统一封装"何时拍照"的逻辑，与硬件无关。

PC Demo 模式
    使用 MockCamera，读取本地图片列表模拟相机帧流。
    无需任何额外依赖，直接运行。

树莓派迁移
    将 MockCamera 替换为 PiCamera，其余代码 **零修改**。
    迁移步骤见 PiCamera 类的注释。

核心流程
    ProductTrigger.check_trigger(frame)
        → 计算标定 ROI 内深色像素占比（fill_ratio）
        → 连续 stable_frames 帧占比 > fill_threshold 且波动 < 0.06
        → 返回 True（触发拍照）

使用示例（PC Demo）
    cam     = MockCamera(image_folder="path/to/frames")
    trigger = ProductTrigger(cam, roi_rel=[0.1, 0.05, 0.9, 0.95])
    cam.start()
    while True:
        frame = cam.capture_frame()
        fill  = trigger.compute_fill_ratio(frame)
        if trigger.check_trigger(fill):
            hires = cam.capture_hires()
            # → 送 Stage1Detector 检测
            trigger.reset()

使用示例（树莓派，迁移后）
    cam = PiCamera(resolution=(1280, 960))   # 改这一行
    # 以下完全相同
    trigger = ProductTrigger(cam, roi_rel=...)
    ...
"""

from __future__ import annotations

import abc
import time
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# 相机抽象基类（统一接口）
# ─────────────────────────────────────────────────────────────────────────

class BaseCamera(abc.ABC):
    """
    相机统一接口。
    PC Demo 和树莓派共用同一套接口，上层代码无感切换。
    """

    @abc.abstractmethod
    def start(self) -> None:
        """初始化并启动相机。"""

    @abc.abstractmethod
    def stop(self) -> None:
        """释放相机资源。"""

    @abc.abstractmethod
    def capture_frame(self) -> np.ndarray | None:
        """
        低延迟预览帧（可降低分辨率以提高速度）。
        返回 BGR ndarray，失败返回 None。
        """

    @abc.abstractmethod
    def capture_hires(self) -> np.ndarray | None:
        """
        高分辨率抓拍，用于送入 Stage1Detector 检测。
        返回 BGR ndarray，失败返回 None。
        """


# ─────────────────────────────────────────────────────────────────────────
# PC Demo 相机
# ─────────────────────────────────────────────────────────────────────────

class MockCamera(BaseCamera):
    """
    PC 模拟相机：循环读取本地图片，模拟流水线帧流。

    参数
    ----
    image_folder : str | Path
        存放模拟帧图片的文件夹（PNG / JPG / BMP）。
    fps : int
        模拟帧率（默认 5 fps），控制 capture_frame() 的节奏。
    loop : bool
        是否在播放完所有帧后循环（默认 True）。
    """

    def __init__(
        self,
        image_folder: str | Path,
        fps: int = 5,
        loop: bool = True,
    ) -> None:
        self.folder = Path(image_folder)
        self.fps    = max(1, fps)
        self.loop   = loop
        self._files:   list[Path] = []
        self._idx:     int        = 0
        self._running: bool       = False

    # ── 接口实现 ─────────────────────────────────────────────────────────

    def start(self) -> None:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self._files = sorted(
            f for f in self.folder.iterdir() if f.suffix.lower() in exts
        )
        if not self._files:
            raise FileNotFoundError(f"MockCamera: 文件夹内无图片 → {self.folder}")
        self._idx     = 0
        self._running = True
        print(f"[MockCamera] 已加载 {len(self._files)} 帧，模拟 {self.fps} fps")

    def stop(self) -> None:
        self._running = False
        print("[MockCamera] 已停止")

    def capture_frame(self) -> np.ndarray | None:
        """读取当前帧，自动按 fps 限速，并推进帧指针。"""
        if not self._running or not self._files:
            return None
        if self._idx >= len(self._files):
            if self.loop:
                self._idx = 0
            else:
                return None
        img = self._read(self._files[self._idx])
        self._idx += 1
        time.sleep(1.0 / self.fps)
        return img

    def capture_hires(self) -> np.ndarray | None:
        """Demo 模式：高清抓拍等同于重新读取当前帧（不推进指针）。"""
        if not self._running or not self._files:
            return None
        idx = max(0, self._idx - 1)
        return self._read(self._files[idx % len(self._files)])

    # ── 额外工具方法 ──────────────────────────────────────────────────────

    def total_frames(self) -> int:
        return len(self._files)

    def current_index(self) -> int:
        return self._idx

    def peek_frame(self, idx: int) -> np.ndarray | None:
        """读取指定序号的帧（不影响内部指针，用于批量预览）。"""
        if not self._files or idx >= len(self._files):
            return None
        return self._read(self._files[idx])

    @staticmethod
    def _read(path: Path) -> np.ndarray | None:
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img


# ─────────────────────────────────────────────────────────────────────────
# 树莓派相机（接口预留）
# ─────────────────────────────────────────────────────────────────────────

class PiCamera(BaseCamera):
    """
    树莓派 picamera2 接口。

    迁移步骤
    --------
    1. 在树莓派上安装依赖：
           pip install picamera2 opencv-python-headless
    2. 将下方注释全部取消。
    3. 调用方代码中把 MockCamera(...) 改为 PiCamera(...)，其他不变。

    推荐配置
    --------
    - 分辨率：1280×960（平衡精度与速度）
    - 关闭自动曝光（AeEnable=False），锁定曝光时间
    - 预览流用低分辨率（320×240），触发后切高分辨率

    抓拍说明
    --------
    - 当前流程为单张抓拍，触发后调用一次 capture_hires()
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1280, 960),
        preview_size: tuple[int, int] = (320, 240),
        exposure_us: int = 2000,   # 曝光时间（微秒），按实际光照调整
        analogue_gain: float = 1.5,
    ) -> None:
        self.resolution    = resolution
        self.preview_size  = preview_size
        self.exposure_us   = exposure_us
        self.analogue_gain = analogue_gain
        self._cam = None

    def start(self) -> None:
        # ── 迁移到树莓派时取消以下注释 ────────────────────────────────
        # from picamera2 import Picamera2
        # self._cam = Picamera2()
        # config = self._cam.create_still_configuration(
        #     main={"size": self.resolution, "format": "BGR888"},
        #     lores={"size": self.preview_size, "format": "BGR888"},
        # )
        # self._cam.configure(config)
        # self._cam.set_controls({
        #     "AeEnable":      False,
        #     "AwbEnable":     False,
        #     "ExposureTime":  self.exposure_us,
        #     "AnalogueGain":  self.analogue_gain,
        # })
        # self._cam.start()
        # print(f"[PiCamera] 启动完成  分辨率={self.resolution}")
        # ──────────────────────────────────────────────────────────────
        raise NotImplementedError(
            "PiCamera 仅在树莓派上可用，请在 PC 上使用 MockCamera"
        )

    def stop(self) -> None:
        # if self._cam:
        #     self._cam.stop()
        #     self._cam.close()
        pass

    def capture_frame(self) -> np.ndarray | None:
        # return self._cam.capture_array("lores")   # 低分辨率预览帧
        raise NotImplementedError

    def capture_hires(self) -> np.ndarray | None:
        # return self._cam.capture_array("main")    # 高分辨率抓拍
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────
# 产品到位触发器
# ─────────────────────────────────────────────────────────────────────────

class ProductTrigger:
    """
    产品完整度检测 + 触发逻辑 + 单张抓拍。

    触发流程
    --------
    1. 持续读取预览帧，计算标定 ROI 内深色像素占比（fill_ratio）
    2. 连续 stable_frames 帧满足：
           fill_ratio > fill_threshold  AND  波动幅度 < stable_tol
       → 触发
    3. 触发后立即抓拍单张高清图送检测

    参数
    ----
    camera            : BaseCamera — 相机实例（MockCamera 或 PiCamera）
    roi_rel           : list       — [x1r, y1r, x2r, y2r]（相对坐标 0~1）
    fill_threshold    : float      — 触发所需最低占比（默认 0.95）
    stable_frames     : int        — 判断稳定所需连续帧数（默认 4）
    dark_thresh       : int        — 产品前景灰度上限（默认 110，深色电饭煲）
    stable_tol        : float      — 帧间允许波动幅度（默认 0.06）
    """

    def __init__(
        self,
        camera: BaseCamera,
        roi_rel: list,
        fill_threshold:    float = 0.95,
        stable_frames:     int   = 4,
        dark_thresh:       int   = 110,
        stable_tol:        float = 0.06,
    ) -> None:
        self.camera            = camera
        self.roi_rel           = roi_rel
        self.fill_threshold    = fill_threshold
        self.stable_frames     = stable_frames
        self.dark_thresh       = dark_thresh
        self.stable_tol        = stable_tol
        self._history: list[float] = []

    # ── 占比检测 ──────────────────────────────────────────────────────────

    def compute_fill_ratio(self, frame: np.ndarray) -> float:
        """计算帧中标定 ROI 内的深色像素占比（0.0 ~ 1.0）。"""
        roi = self._crop_roi(frame)
        if roi is None or roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        fg   = gray < self.dark_thresh
        return float(fg.sum()) / float(fg.size)

    def check_trigger(self, fill_ratio: float) -> bool:
        """
        将 fill_ratio 送入历史窗口，判断是否满足触发条件。
        触发后必须调用 reset()，否则会持续触发。
        """
        self._history.append(fill_ratio)
        if len(self._history) > self.stable_frames * 2:
            self._history.pop(0)
        if len(self._history) < self.stable_frames:
            return False
        recent = self._history[-self.stable_frames:]
        stable = (max(recent) - min(recent)) < self.stable_tol
        above  = min(recent) > self.fill_threshold
        return stable and above

    def reset(self) -> None:
        """触发完成后调用，清空历史窗口，准备检测下一个产品。"""
        self._history.clear()

    def get_history(self) -> list[float]:
        """返回当前历史窗口的占比列表（用于可视化）。"""
        return list(self._history)

    # ── 单张抓拍 ───────────────────────────────────────────────────────────

    def capture_single(self) -> np.ndarray | None:
        """触发后抓拍单张高清图。"""
        img = self.camera.capture_hires()
        if img is not None:
            print("[Capture] 已抓拍 1 张高清图")
        return img

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1 = int(self.roi_rel[0] * w);  y1 = int(self.roi_rel[1] * h)
        x2 = int(self.roi_rel[2] * w);  y2 = int(self.roi_rel[3] * h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # ── 便捷阻塞接口（供树莓派脚本直接调用） ─────────────────────────────

    def run_until_trigger(
        self, timeout: float = 30.0
    ) -> np.ndarray | None:
        """
        阻塞运行，直到触发或超时。
        触发后自动抓拍单张高清图并返回。
        超时返回 None。

        典型用法（树莓派主循环）
        -----------------------
        cam     = PiCamera()
        cam.start()
        trigger = ProductTrigger(cam, roi_rel=..., fill_threshold=0.95)
        while True:
            shot = trigger.run_until_trigger(timeout=60)
            if shot is None:
                continue
            result = detector.inspect_with_localization(shot, ...)
            # → 上报结果 / 控制流水线
        """
        t0 = time.time()
        self.reset()
        while time.time() - t0 < timeout:
            frame = self.camera.capture_frame()
            if frame is None:
                continue
            fill = self.compute_fill_ratio(frame)
            if self.check_trigger(fill):
                shot = self.capture_single()
                self.reset()
                return shot
        print("[ProductTrigger] 超时未检测到完整产品")
        return None
