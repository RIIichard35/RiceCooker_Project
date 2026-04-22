"""
scripts/verify_hef.py — 在树莓派上验证 Hailo .hef 模型是否正常运行
=====================================================================
在 Pi 上运行：
    python3 scripts/verify_hef.py
    python3 scripts/verify_hef.py --hef assets/models/best20240919.hef
    python3 scripts/verify_hef.py --image assets/standards/back/125225back.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import HEF_PATH, MODELS_DIR, STANDARDS_DIR

# ─────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Hailo HEF 模型验证工具")
ap.add_argument("--hef",   default=HEF_PATH,   help=".hef 文件路径")
ap.add_argument("--image", default=None,        help="测试图片路径（不填则用随机噪声）")
ap.add_argument("--runs",  type=int, default=5, help="推理次数（用于测量平均耗时）")
args = ap.parse_args()

# ─────────────────────────────────────────────────────────────────────────
# 检查文件
# ─────────────────────────────────────────────────────────────────────────
hef_path = Path(args.hef)
if not hef_path.exists():
    print(f"[ERROR] 找不到 .hef 文件: {hef_path}")
    print("  请先在 WSL2 上运行 scripts/convert_to_hef.sh 完成转换")
    print("  然后将生成的 .hef 文件 scp 到此目录")
    sys.exit(1)

print(f"[INFO] HEF 文件: {hef_path}  ({hef_path.stat().st_size / 1024 / 1024:.1f} MB)")

# ─────────────────────────────────────────────────────────────────────────
# 检查 hailo_platform
# ─────────────────────────────────────────────────────────────────────────
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType,
    )
    print("[INFO] hailo_platform 导入成功")
except ImportError as e:
    print(f"[ERROR] hailo_platform 未安装: {e}")
    print("  在 Pi 上运行：pip install hailort")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# 准备测试图片
# ─────────────────────────────────────────────────────────────────────────
if args.image:
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"[ERROR] 无法读取图片: {args.image}")
        sys.exit(1)
    print(f"[INFO] 测试图片: {args.image}  {img_bgr.shape}")
else:
    # 尝试从标准图库取第一张图
    std_root = Path(STANDARDS_DIR)
    first_img = next(
        (f for d in sorted(std_root.iterdir()) if d.is_dir()
         for f in sorted(d.iterdir())
         if f.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        None,
    )
    if first_img:
        img_bgr = cv2.imread(str(first_img))
        print(f"[INFO] 使用标准图库图片: {first_img.name}  {img_bgr.shape}")
    else:
        img_bgr = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("[INFO] 使用随机噪声图片（640×640）")

# 预处理 → NHWC float32（Hailo 输入格式）
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_res  = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
inp_nhwc = img_res[np.newaxis]   # (1, 640, 640, 3)

# ─────────────────────────────────────────────────────────────────────────
# 初始化 Hailo 推理管道
# ─────────────────────────────────────────────────────────────────────────
print("[INFO] 初始化 Hailo 设备...")
t0 = time.time()

hef       = HEF(str(hef_path))
vdevice   = VDevice()
cfg_params = ConfigureParams.create_from_hef(
    hef, interface=HailoStreamInterface.PCIe
)
net_groups        = vdevice.configure(hef, cfg_params)
net_group         = net_groups[0]
net_group_params  = net_group.create_params()

in_params  = InputVStreamParams.make(net_group,  format_type=FormatType.FLOAT32)
out_params = OutputVStreamParams.make(net_group, format_type=FormatType.FLOAT32)

input_name = list(hef.get_input_vstream_infos())[0].name
print(f"[INFO] 设备初始化完成，耗时 {time.time()-t0:.2f}s")
print(f"[INFO] 输入层: {input_name}")

# 打印模型 I/O 信息
print("\n===== 模型 I/O 信息 =====")
for info in hef.get_input_vstream_infos():
    print(f"  输入: {info.name}  shape={info.shape}  dtype={info.format.type}")
for info in hef.get_output_vstream_infos():
    print(f"  输出: {info.name}  shape={info.shape}  dtype={info.format.type}")
print("=========================\n")

# ─────────────────────────────────────────────────────────────────────────
# 执行推理（多次，统计耗时）
# ─────────────────────────────────────────────────────────────────────────
timings = []

with InferVStreams(net_group, in_params, out_params) as pipeline:
    with net_group.activate(net_group_params):
        for i in range(args.runs):
            t_infer = time.time()
            raw_out = pipeline.infer({input_name: inp_nhwc})
            elapsed = time.time() - t_infer
            timings.append(elapsed)
            print(f"  推理[{i}]: {elapsed*1000:.1f} ms")

print(f"\n===== 推理耗时统计（{args.runs} 次） =====")
print(f"  平均: {sum(timings)/len(timings)*1000:.1f} ms")
print(f"  最快: {min(timings)*1000:.1f} ms")
print(f"  最慢: {max(timings)*1000:.1f} ms")

# ─────────────────────────────────────────────────────────────────────────
# 解析输出（简单验证，不做完整 NMS）
# ─────────────────────────────────────────────────────────────────────────
out_key = list(raw_out.keys())[0]
out_arr = np.squeeze(raw_out[out_key])
print(f"\n===== 输出张量 =====")
print(f"  键名: {out_key}")
print(f"  形状: {out_arr.shape}")
print(f"  值域: [{out_arr.min():.4f}, {out_arr.max():.4f}]")

# 简单统计高置信度检测框数量
if out_arr.ndim == 2:
    pred = out_arr if out_arr.shape[1] == 6 else out_arr.T  # → (N, 6)
    scores = pred[:, 4:].max(axis=1)
    high_conf = (scores > 0.25).sum()
    print(f"  置信度>0.25 的候选框: {high_conf}")

print("\n[PASS] Hailo-8L 推理验证通过 ✓")
print("  .hef 模型可正常在 AI HAT+ 上运行")
print("  Stage2Detector 将自动检测并使用 Hailo 加速")
