"""
scripts/train_anomalib.py — 每视角训练 PatchCore 并导出 ONNX
=============================================================

用法（在项目根目录执行）：
    python scripts/train_anomalib.py
    python scripts/train_anomalib.py --views front back        # 只训练指定视角
    python scripts/train_anomalib.py --views top --augment     # 启用数据增强

训练数据：st/<view>/*.png（预处理透明背景抠图）
输出：
    assets/models/anomalib/<view>/weights/lightning/model.ckpt  (Lightning checkpoint)
    assets/models/anomalib/<view>/model.onnx                    (ONNX，供树莓派推理)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ANOMALIB_INPUT_SIZE, ANOMALIB_MODEL_DIR, ST_DIR

VIEWS = ["front", "back", "left", "right", "top"]


# ─────────────────────────────────────────────────────────────────────────
# 预处理：RGBA 抠图 → 固定灰色背景 RGB
# ─────────────────────────────────────────────────────────────────────────

def rgba_to_rgb_gray_bg(
    img_rgba: np.ndarray,
    bg_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """将 RGBA 抠图的透明区域替换为固定灰色背景，返回 RGB uint8。"""
    if img_rgba.ndim == 3 and img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3].astype(np.float32) / 255.0
        rgb   = img_rgba[:, :, :3].astype(np.float32)
        bg    = np.full_like(rgb, bg_color, dtype=np.float32)
        out   = alpha[..., None] * rgb + (1 - alpha[..., None]) * bg
        return np.clip(out, 0, 255).astype(np.uint8)
    if img_rgba.ndim == 3 and img_rgba.shape[2] == 3:
        return cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2RGB)


def prepare_train_dir(view: str, augment: bool = False) -> str:
    """
    将 st/<view>/ 下的 RGBA PNG 预处理后复制到临时目录，
    返回该临时目录路径（调用方负责清理）。
    """
    src_dir = Path(ST_DIR) / view
    if not src_dir.exists():
        raise FileNotFoundError(f"抠图目录不存在: {src_dir}")

    png_files = sorted(src_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"未找到 PNG 文件: {src_dir}")

    tmp_dir = tempfile.mkdtemp(prefix=f"anomalib_{view}_")

    size = int(ANOMALIB_INPUT_SIZE)
    written = 0
    base_images: list[np.ndarray] = []

    for p in png_files:
        arr = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [警告] 无法读取: {p.name}")
            continue

        rgb = rgba_to_rgb_gray_bg(img)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        base_images.append(rgb)

        out_path = os.path.join(tmp_dir, p.stem + ".jpg")
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        written += 1

    if augment and base_images:
        written += _write_augmented(base_images, tmp_dir)

    print(f"  [{view}] 准备训练图 {written} 张 → {tmp_dir}")
    return tmp_dir


def _write_augmented(images: list[np.ndarray], out_dir: str) -> int:
    """简单数据增强：亮度、对比度、轻微旋转。"""
    count = 0
    for idx, img in enumerate(images):
        # 亮度 +20
        bright = np.clip(img.astype(np.int32) + 20, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"aug_bright_{idx}.jpg"),
                    cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))
        count += 1
        # 亮度 -20
        dark = np.clip(img.astype(np.int32) - 20, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"aug_dark_{idx}.jpg"),
                    cv2.cvtColor(dark, cv2.COLOR_RGB2BGR))
        count += 1
        # 轻微旋转 ±5°
        h, w = img.shape[:2]
        for angle in (5, -5):
            M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(out_dir, f"aug_rot{angle}_{idx}.jpg"),
                        cv2.cvtColor(rot, cv2.COLOR_RGB2BGR))
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────

def train_view(view: str, augment: bool = False) -> None:
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    print(f"\n{'='*56}")
    print(f"  视角: {view}")
    print(f"{'='*56}")

    out_dir = Path(ANOMALIB_MODEL_DIR) / view
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = prepare_train_dir(view, augment=augment)
    try:
        datamodule = Folder(
            name=view,
            root=tmp_dir,
            normal_dir=".",         # 正常图直接在 tmp_dir 下
            normal_split_ratio=0.2, # 20% 做验证集
            val_split_mode="from_train",
            val_split_ratio=0.2,
        )

        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,  # 减小内存占用，适合 Pi
            num_neighbors=9,
        )

        engine = Engine(default_root_dir=str(out_dir))
        engine.fit(model, datamodule=datamodule)

        # 导出 ONNX
        onnx_path = out_dir / "model.onnx"
        print(f"  导出 ONNX → {onnx_path}")
        try:
            engine.export(
                model=model,
                export_type="onnx",
                export_root=str(out_dir),
                input_size=(ANOMALIB_INPUT_SIZE, ANOMALIB_INPUT_SIZE),
            )
            # Anomalib 导出到 results/Patchcore/.../weights/onnx/model.onnx，移动到固定位置
            _move_onnx(out_dir, onnx_path)
        except Exception as e:
            print(f"  [警告] ONNX 导出失败: {e}（可继续使用 Lightning checkpoint 推理）")

        print(f"  [{view}] 训练完成，模型已保存至: {out_dir}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _move_onnx(search_root: Path, target: Path) -> None:
    """在 search_root 下递归查找第一个 model.onnx，移动到 target。"""
    for p in search_root.rglob("model.onnx"):
        if p.resolve() != target.resolve():
            shutil.move(str(p), str(target))
            print(f"  ONNX 已移动: {p} → {target}")
            return
    # 若目标已存在则无需移动
    if target.exists():
        return
    print(f"  [警告] 未找到导出的 model.onnx（搜索根: {search_root}）")


# ─────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="PatchCore 训练脚本（逐视角）")
    ap.add_argument(
        "--views", nargs="+", default=VIEWS,
        help=f"要训练的视角列表，默认全部 {VIEWS}",
    )
    ap.add_argument(
        "--augment", action="store_true",
        help="启用数据增强（亮度变化 + 轻微旋转），样本极少时推荐开启",
    )
    args = ap.parse_args()

    for view in args.views:
        if view not in VIEWS:
            print(f"[跳过] 未知视角: {view}")
            continue
        try:
            train_view(view, augment=args.augment)
        except Exception as e:
            print(f"[错误] 视角 {view} 训练失败: {e}")

    print("\n全部视角训练完成。")
    print(f"模型目录: {ANOMALIB_MODEL_DIR}")


if __name__ == "__main__":
    main()
