"""
电饭煲外观检测 — 树莓派精简前端（app3）
=========================================
设计原则：
  · 单页，无多标签切换，降低浏览器渲染压力
  · 侧边栏只保留最关键的参数
  · 自动读取已有 ROI 标定文件（标定工作在 PC 端 app2 完成后同步过来）
  · 检测结果只显示：结论卡片 + 关键数字 + 标注图
  · 支持「单张上传」和「摄像头文件夹轮询」两种模式

运行（树莓派）：
    streamlit run gui/app3.py --server.port 8502 \
        --server.headless true --server.enableCORS false
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── 路径 ──────────────────────────────────────────────────────────────────
GUI_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CALIB_DIR,
    FILM_COVERAGE_THRESHOLD,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    MODELS_DIR,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage3_yolov8 import Stage2Detector

_STD_ROOT  = Path(STANDARDS_DIR)
_CALIB_DIR = Path(CALIB_DIR)
_ONNX_PATH = Path(MODELS_DIR) / "best20240919.onnx"

# ── 页面配置（轻量） ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="电饭煲检测",
    page_icon="✅",
    layout="centered",          # 单列，适合小屏/触摸屏
    initial_sidebar_state="collapsed",
)

# 极简 CSS —— 减少渲染量
st.markdown("""
<style>
.result-pass { background:#d4edda; color:#155724; border-radius:12px;
               padding:20px; text-align:center; font-size:1.6rem;
               font-weight:700; margin:12px 0; }
.result-fail { background:#f8d7da; color:#721c24; border-radius:12px;
               padding:20px; text-align:center; font-size:1.6rem;
               font-weight:700; margin:12px 0; }
.result-warn { background:#fff3cd; color:#856404; border-radius:12px;
               padding:20px; text-align:center; font-size:1.6rem;
               font-weight:700; margin:12px 0; }
.kv-row { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
.kv    { flex:1; min-width:80px; background:#f0f2f6; border-radius:8px;
         padding:8px 6px; text-align:center; }
.kv-lb { font-size:0.70rem; color:#555; }
.kv-v  { font-size:1.05rem; font-weight:700; }
.issue { color:#721c24; font-size:0.88rem; margin:2px 0; }
.warn  { color:#856404; font-size:0.88rem; margin:2px 0; }
</style>
""", unsafe_allow_html=True)


# ── 工具 ─────────────────────────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def load_calib(view: str) -> list | None:
    p = _CALIB_DIR / f"{view}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("roi_rel")
    except Exception:
        return None

def rel_to_abs(roi_rel: list, w: int, h: int) -> list:
    return [int(roi_rel[0]*w), int(roi_rel[1]*h),
            int(roi_rel[2]*w), int(roi_rel[3]*h)]

VIEW_LABELS = {
    "front": "正面", "back": "后面",
    "left":  "左面", "right": "右面", "top": "顶部",
}

# ── 可用视角 ─────────────────────────────────────────────────────────────
if _STD_ROOT.exists():
    _views = sorted(
        d.name for d in _STD_ROOT.iterdir()
        if d.is_dir() and any(d.iterdir())
    )
else:
    _views = []

_view_display = [VIEW_LABELS.get(v, v) for v in _views]
_view_map     = dict(zip(_view_display, _views))


# ── 侧边栏（极简） ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置")

    sel_label = st.selectbox("检测视角", _view_display,
                             index=0 if _view_display else None)
    sel_view  = _view_map.get(sel_label, sel_label) if sel_label else None

    st.markdown("---")
    st.subheader("阈值")
    film_thresh    = st.slider("覆膜率上限 (%)", 10, 90,
                               int(FILM_COVERAGE_THRESHOLD * 100), 5)
    missing_thresh = st.slider("异常框阈值", 1, 10, MAX_MISSING_COUNT, 1)
    conf_thresh    = st.slider("划痕置信度", 0.05, 0.90, 0.25, 0.05)

    st.markdown("---")
    enable_s2 = st.checkbox("启用细筛（Stage 2）", value=True)
    show_ann  = st.checkbox("显示标注图", value=True)


# ── 检测器缓存 ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="加载标准图库…")
def _get_s1(view: str) -> Stage1Detector:
    return Stage1Detector(str(_STD_ROOT / view))

@st.cache_resource(show_spinner="加载 ONNX 模型…")
def _get_s2(conf: float) -> Stage2Detector | None:
    if not _ONNX_PATH.exists():
        return None
    return Stage2Detector(str(_ONNX_PATH), conf_threshold=conf)


# ── 主界面 ────────────────────────────────────────────────────────────────
st.title("电饭煲外观检测")

if not _views:
    st.error("未找到标准图库，请确认 assets/standards/ 目录存在。")
    st.stop()

if not sel_view:
    st.warning("请在左侧选择检测视角。")
    st.stop()

# ROI 标定状态
_roi_rel = load_calib(sel_view)
_roi_badge = "📐 ROI 已标定" if _roi_rel else "📐 ROI 未标定（自动定位）"
_roi_color = "#d4edda" if _roi_rel else "#fff3cd"
st.markdown(
    f'<span style="background:{_roi_color};padding:4px 10px;'
    f'border-radius:6px;font-size:0.82rem;">{_roi_badge}</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── 上传图片 ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "上传待检测图片",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded is None:
    st.info("👆 请上传一张待检测的产品图片")
    st.stop()

img_bgr = load_bgr(uploaded.read())
if img_bgr is None:
    st.error("图片解码失败，请重新上传。")
    st.stop()

h_img, w_img = img_bgr.shape[:2]
st.image(bgr_to_rgb(img_bgr),
         caption=f"原图  {w_img}×{h_img}px",
         width="stretch")

# ── 开始检测 ─────────────────────────────────────────────────────────────
det1 = _get_s1(sel_view)
calib_bbox = rel_to_abs(_roi_rel, w_img, h_img) if _roi_rel else None

with st.spinner("Stage 1 初筛中…"):
    r1 = det1.inspect_with_localization(
        img_bgr,
        min_match_count=MIN_MATCH_COUNT,
        missing_regions_fail_count=missing_thresh,
        calibrated_bbox=calib_bbox,
    )

s1 = r1["status"]

# ── Stage 1 结果 ─────────────────────────────────────────────────────────
_cls = {"PASS": "result-pass", "FAIL": "result-fail"}.get(s1, "result-warn")
_ico = {"PASS": "✅ 初筛通过", "FAIL": "❌ 初筛不合格",
        "RETAKE": "⚠️ 需重拍"}.get(s1, s1)
st.markdown(f'<div class="{_cls}">{_ico}</div>', unsafe_allow_html=True)

# 关键数字
st.markdown(
    f'<div class="kv-row">'
    f'<div class="kv"><div class="kv-lb">定位</div>'
    f'<div class="kv-v">{"已标定" if _roi_rel else "自动"}</div></div>'
    f'<div class="kv"><div class="kv-lb">覆膜率</div>'
    f'<div class="kv-v">{r1["film_coverage"]*100:.1f}%</div></div>'
    f'<div class="kv"><div class="kv-lb">相似度</div>'
    f'<div class="kv-v">{r1["similarity"]*100:.0f}%</div></div>'
    f'<div class="kv"><div class="kv-lb">异常框</div>'
    f'<div class="kv-v">{len(r1["missing_regions"])}</div></div>'
    f'<div class="kv"><div class="kv-lb">外侧标签</div>'
    f'<div class="kv-v">{r1.get("outer_label_count", 0)}</div></div>'
    f'<div class="kv"><div class="kv-lb">已滤杂质</div>'
    f'<div class="kv-v">{r1.get("outer_label_removed_count", 0)}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

for iss in r1["issues"]:
    st.markdown(f'<p class="issue">⛔ {iss}</p>', unsafe_allow_html=True)
for w in r1["warnings"]:
    st.markdown(f'<p class="warn">⚠ {w}</p>', unsafe_allow_html=True)

if show_ann and r1.get("annotated_image") is not None:
    st.image(bgr_to_rgb(r1["annotated_image"]),
             caption="Stage 1 标注图", width="stretch")

# ── Stage 2（仅 PASS 且启用时） ───────────────────────────────────────────
if s1 != "PASS" or not enable_s2:
    if s1 != "PASS":
        st.markdown("---")
        st.markdown(
            '<div class="result-fail">综合结论：产品不合格</div>',
            unsafe_allow_html=True,
        )
    st.stop()

st.markdown("---")
det2 = _get_s2(conf_thresh)
if det2 is None:
    st.warning(f"找不到 ONNX 模型（{_ONNX_PATH.name}），跳过细筛。")
    st.markdown(
        '<div class="result-pass">综合结论：初筛通过（细筛已跳过）</div>',
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Stage 2 划痕检测中…"):
    r2 = det2.inspect(img_bgr, r1)

s2 = r2["status"]
n_total   = len(r2["defects"])
n_scratch = r2["defect_counts"].get("scratch", 0)
n_crack   = r2["defect_counts"].get("crack",   0)

_cls2 = "result-pass" if s2 == "PASS" else "result-fail"
_ico2 = ("✅ 细筛通过 — 表面无缺陷"
         if s2 == "PASS"
         else f"❌ 发现 {n_total} 处缺陷（划痕 {n_scratch} / 裂纹 {n_crack}）")
st.markdown(f'<div class="{_cls2}">{_ico2}</div>', unsafe_allow_html=True)

if show_ann and r2.get("annotated_image") is not None:
    st.image(bgr_to_rgb(r2["annotated_image"]),
             caption="Stage 2 标注图", width="stretch")

# ── 综合结论 ─────────────────────────────────────────────────────────────
st.markdown("---")
final = "PASS" if s2 == "PASS" else "FAIL"
final_html = (
    '<div class="result-pass">综合结论：两阶段均通过 — 产品合格 ✅</div>'
    if final == "PASS" else
    '<div class="result-fail">综合结论：细筛发现缺陷 — 产品不合格 ❌</div>'
)
st.markdown(final_html, unsafe_allow_html=True)
