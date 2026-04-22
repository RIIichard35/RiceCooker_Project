#!/usr/bin/env bash
# =============================================================================
# convert_to_hef.sh — 将 best20240919.onnx 转换为 Hailo-8L .hef 格式
# =============================================================================
# 运行环境：WSL2 Ubuntu 22.04 x86_64
#
# 使用方法：
#   1. 在 WSL2 中进入项目根目录
#      cd /mnt/d/数据集/RiceCooker_Project
#   2. 赋予执行权限
#      chmod +x scripts/convert_to_hef.sh
#   3. 运行
#      bash scripts/convert_to_hef.sh
#
# 输出：
#   assets/models/best20240919.hef
#   把这个文件 scp 到树莓派的 assets/models/ 即可自动启用 Hailo 加速
# =============================================================================

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/assets/models"
ONNX_FILE="$MODEL_DIR/best20240919.onnx"
HEF_FILE="$MODEL_DIR/best20240919.hef"
CALIB_DIR="$PROJECT_ROOT/assets/standards"   # 用标准图库做量化校准集
WORK_DIR="$PROJECT_ROOT/hailo_workspace"

# ── 颜色输出 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 前置检查 ──────────────────────────────────────────────────────────────
info "检查 ONNX 模型文件..."
[[ -f "$ONNX_FILE" ]] || error "找不到 $ONNX_FILE，请确认模型已放置在 assets/models/"

info "检查 Python 环境..."
python3 --version || error "未找到 python3"

# ── 安装 Hailo DFC（若未安装） ────────────────────────────────────────────
info "检查 hailo-dataflow-compiler..."
if ! python3 -c "import hailo_sdk_client" 2>/dev/null; then
    warn "hailo-dataflow-compiler 未安装，开始安装..."
    warn "注意：需要从 Hailo Developer Zone 下载 whl 包"
    warn "  https://hailo.ai/developer-zone/sw-downloads/"
    warn "  选择：Hailo Dataflow Compiler → Linux x86 → Python 3.10"
    echo ""
    echo "请手动安装后重新运行本脚本："
    echo "  pip install hailo_dataflow_compiler-*.whl"
    exit 1
fi
info "hailo-dataflow-compiler 已就绪"

# ── 准备工作目录 ──────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ── 准备校准数据集（从标准图库采样） ──────────────────────────────────────
CALIB_IMAGES="$WORK_DIR/calib_images"
mkdir -p "$CALIB_IMAGES"

info "采集校准图片（从 assets/standards 复制最多 50 张）..."
COUNT=0
for view_dir in "$CALIB_DIR"/*/; do
    for img in "$view_dir"*.png "$view_dir"*.jpg; do
        [[ -f "$img" ]] || continue
        cp "$img" "$CALIB_IMAGES/"
        COUNT=$((COUNT + 1))
        [[ $COUNT -ge 50 ]] && break 2
    done
done
info "校准图片数量: $COUNT 张"

if [[ $COUNT -eq 0 ]]; then
    warn "未找到校准图片，将使用随机数据校准（精度可能略低）"
    USE_RANDOM_CALIB=true
else
    USE_RANDOM_CALIB=false
fi

# ── Step 1: 解析 ONNX → Hailo 中间格式 ──────────────────────────────────
info "Step 1/3: 解析 ONNX 模型..."
python3 - <<'PYEOF'
import sys
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l")
print("[DFC] 开始解析 ONNX...", flush=True)

# YOLOv8 输入尺寸 640×640
runner.translate_onnx_model(
    "../assets/models/best20240919.onnx",
    net_name="best20240919",
    start_node_names=None,   # 自动检测
    end_node_names=None,
)
runner.save_har("best20240919_parsed.har")
print("[DFC] 解析完成 → best20240919_parsed.har", flush=True)
PYEOF

# ── Step 2: 量化优化 ─────────────────────────────────────────────────────
info "Step 2/3: 量化优化（int8）..."
python3 - <<PYEOF
import sys, os, glob
from hailo_sdk_client import ClientRunner
from hailo_sdk_common.targets.inference_targets import SdkFPOptimized

runner = ClientRunner(hw_arch="hailo8l", har="best20240919_parsed.har")

use_random = ${USE_RANDOM_CALIB}

if not use_random:
    # 收集校准图片
    import cv2, numpy as np
    imgs = sorted(glob.glob("calib_images/*"))[:50]
    calib_data = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        calib_data.append(img[np.newaxis])   # (1, 640, 640, 3) NHWC
    
    if calib_data:
        calib_dataset = np.concatenate(calib_data, axis=0)  # (N, 640, 640, 3)
        print(f"[DFC] 校准数据集: {calib_dataset.shape}", flush=True)
        runner.optimize(calib_dataset)
    else:
        print("[DFC] 校准图片加载失败，改用随机校准", flush=True)
        runner.optimize(calib_set_size=64)
else:
    print("[DFC] 使用随机校准集（精度略低）", flush=True)
    runner.optimize(calib_set_size=64)

runner.save_har("best20240919_optimized.har")
print("[DFC] 量化完成 → best20240919_optimized.har", flush=True)
PYEOF

# ── Step 3: 编译 → .hef ──────────────────────────────────────────────────
info "Step 3/3: 编译到 Hailo-8L..."
python3 - <<'PYEOF'
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l", har="best20240919_optimized.har")
hef_bytes = runner.compile()

output_path = "../assets/models/best20240919.hef"
with open(output_path, "wb") as f:
    f.write(hef_bytes)
print(f"[DFC] 编译完成 → {output_path}", flush=True)
PYEOF

# ── 完成 ──────────────────────────────────────────────────────────────────
info "========================================"
info "转换完成！"
info "  输出文件: $HEF_FILE"
info ""
info "下一步（在 PC 上执行）："
info "  scp $HEF_FILE pi@<树莓派IP>:/home/pi/RiceCooker_Project/assets/models/"
info ""
info "在树莓派上验证："
info "  python3 scripts/verify_hef.py"
info "========================================"
