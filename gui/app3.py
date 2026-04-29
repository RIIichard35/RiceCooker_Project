"""
电饭煲外观检测 — 现场精简前端（app3）
=========================================
与 app1 保持同一套 UI 规范（样式类名、侧边栏用语、工具函数命名）。
流程：Stage 1（可带 ROI 标定）→ 可选 Stage 2 YOLO 细筛。

相对 app1：无 Anomalib；默认带 Stage 2；页面可部署在树莓派（仍建议用「开始检测」避免调参时重复推理）。

运行示例：
    streamlit run gui/app3.py --server.port 8502 \\
        --server.headless true --server.enableCORS false
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── 路径设置 ─────────────────────────────────────────────────────────────
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

std_root   = Path(STANDARDS_DIR)
calib_root = Path(CALIB_DIR)
onnx_path  = Path(MODELS_DIR) / "best20240919.onnx"

# ── 页面配置（和 app1 一致：宽屏 + 侧栏展开） ─────────────────────────────
st.set_page_config(
    page_title="电饭煲外观检测台（现场）",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }

.verdict-card {
    padding: 20px 28px; border-radius: 12px;
    text-align: center; margin-bottom: 16px;
}
.verdict-card h2 { margin: 0 0 10px; font-size: 1.5rem; }
.verdict-card p { margin: 3px 0; font-size: 0.88rem; line-height: 1.35; }
.verdict-pass   { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.verdict-fail   { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
.verdict-retake { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
.verdict-skip   { background: #e2e3e5; color: #383d41; border: 2px solid #d6d8db; }

.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 14px; }
.metric-box {
    flex: 1; min-width: 100px; background: #f0f2f6;
    border-radius: 8px; padding: 10px 6px; text-align: center;
}
.metric-label { font-size: 0.70rem; color: #666; margin-bottom: 4px; }
.metric-value { font-size: 1.15rem; font-weight: 700; color: #222; }

.step-header {
    font-size: 0.85rem; font-weight: 700; color: #0078d4;
    border-left: 3px solid #0078d4; padding-left: 8px;
    margin: 14px 0 6px;
}

.issue-item { color: #721c24; margin: 3px 0; font-size: 0.88rem; }
.warn-item  { color: #856404; margin: 3px 0; font-size: 0.88rem; }

.img-caption {
    text-align: center; font-size: 0.78rem; color: #555;
    margin-top: 4px; padding: 3px 6px;
    background: #f8f9fa; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ─────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgra_to_rgba(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def verdict_card_html(
    status: str,
    issues: list[str],
    warnings: list[str],
    title: str = "",
) -> str:
    css = {"PASS": "verdict-pass", "FAIL": "verdict-fail",
           "RETAKE": "verdict-retake", "SKIP": "verdict-skip"}
    icon = {"PASS": "✅", "FAIL": "❌", "RETAKE": "⚠️", "SKIP": "⏭️"}
    default_l = {"PASS": "初筛通过", "FAIL": "初筛不合格",
                 "RETAKE": "需要重拍", "SKIP": "已跳过"}
    c = css.get(status, "verdict-fail")
    i = icon.get(status, "❓")
    l = title or default_l.get(status, status)
    issues_html = "".join(
        f'<p class="issue-item">⛔ {s}</p>' for s in issues
    ) if issues else '<p style="color:#155724;font-size:0.88rem">无阻断性问题</p>'
    warns_html = "".join(
        f'<p class="warn-item">⚠ {w}</p>' for w in warnings
    ) if warnings else ""
    return (
        f'<div class="verdict-card {c}">'
        f'<h2 style="margin:0 0 10px;font-size:1.5rem">{i} {l}</h2>'
        f'{issues_html}{warns_html}</div>'
    )


def metric_row(metrics: list[tuple[str, str]]) -> str:
    boxes = "".join(
        f'<div class="metric-box">'
        f'<div class="metric-label">{lb}</div>'
        f'<div class="metric-value">{val}</div></div>'
        for lb, val in metrics
    )
    return f'<div class="metric-row">{boxes}</div>'


def step_header(text: str) -> None:
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)


def load_calib(view: str) -> list | None:
    p = calib_root / f"{view}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8")).get("roi_rel")
    except Exception:
        return None


def rel_to_abs(roi_rel: list, w: int, h: int) -> list[int]:
    return [
        int(roi_rel[0] * w),
        int(roi_rel[1] * h),
        int(roi_rel[2] * w),
        int(roi_rel[3] * h),
    ]


VIEW_LABELS = {"front": "正面", "back": "后面", "left": "左面",
               "right": "右面", "top": "顶部"}

if std_root.exists():
    available_views = sorted(
        d.name for d in std_root.iterdir()
        if d.is_dir() and any(d.iterdir())
    )
else:
    available_views = []

view_display = [VIEW_LABELS.get(v, v) for v in available_views]
view_map     = dict(zip(view_display, available_views))


# ── 侧边栏 ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 检测配置")

if not available_views:
    st.sidebar.error(f"找不到标准图库：\n{std_root}")

selected_label: str | None = st.sidebar.selectbox(
    "检测视角",
    options=view_display,
    index=0 if view_display else None,
)
selected_view = view_map.get(selected_label, selected_label) if selected_label else None

st.sidebar.markdown("---")
st.sidebar.subheader("检测阈值")

film_thresh = st.sidebar.slider(
    "塑料膜阈值 (%)", 10, 90, int(FILM_COVERAGE_THRESHOLD * 100), 5,
)
match_thresh = st.sidebar.slider("ORB 匹配点", 5, 60, MIN_MATCH_COUNT)
missing_thresh = st.sidebar.slider("缺件框阈值", 1, 10, MAX_MISSING_COUNT)

st.sidebar.markdown("---")
st.sidebar.subheader("检测阈值（Stage 2）")
conf_thresh = st.sidebar.slider(
    "划痕置信度", 0.05, 0.90, 0.25, 0.05,
    help="YOLO 检测框的最低置信度，越低越灵敏",
)
iou_thresh = st.sidebar.slider(
    "NMS IoU 阈值", 0.10, 0.80, 0.45, 0.05,
    help="重叠框合并阈值，越小保留框越少",
)
roi_padding = st.sidebar.slider(
    "产品框扩展比例", 0.00, 0.20, 0.05, 0.01,
    help="在 Stage1 定位框外额外扩展的比例，防止边缘划痕被裁掉",
)

st.sidebar.markdown("---")
enable_s2 = st.sidebar.checkbox("启用细筛（Stage 2）", value=True)
show_ann  = st.sidebar.checkbox("显示标注图", value=True)


@st.cache_resource(show_spinner="正在加载标准图库...")
def load_detector(view_name: str) -> Stage1Detector:
    return Stage1Detector(str(std_root / view_name))


@st.cache_resource(show_spinner="正在加载 ONNX 模型...")
def load_stage2(conf: float, iou: float, pad: float) -> Stage2Detector | None:
    if not onnx_path.exists():
        return None
    return Stage2Detector(
        str(onnx_path),
        conf_threshold=conf,
        iou_threshold=iou,
        roi_padding=pad,
    )


detector: Stage1Detector | None = None
stage2_det: Stage2Detector | None = None
if selected_view:
    detector = load_detector(selected_view)
    st.sidebar.success(
        f"已加载 **{selected_label}** 标准图库（{len(detector.reference_data)} 张）"
    )
    if enable_s2:
        stage2_det = load_stage2(conf_thresh, iou_thresh, roi_padding)
        if stage2_det:
            st.sidebar.success("ONNX 模型已加载")
        else:
            st.sidebar.error(f"找不到模型：{onnx_path.name}")
else:
    st.sidebar.warning("请选择检测视角")


# ── 主界面 ───────────────────────────────────────────────────────────────
st.title("✅ 电饭煲外观检测台（现场 · Stage 1 → Stage 2）")
st.caption("与 app1 共用 UI 规范；可选 ROI 标定（app2 同步 calibration）。")
st.markdown("---")

if not available_views:
    st.error("未找到标准图库，请确认 assets/standards/ 目录存在。")
    st.stop()

if not selected_view:
    st.warning("请在侧栏选择检测视角。")
    st.stop()

roi_rel = load_calib(selected_view)
_roi_badge = "ROI 已标定（使用 calibration）" if roi_rel else "ROI 未标定（自动定位）"
_roi_color = "#d4edda" if roi_rel else "#fff3cd"
st.markdown(
    f'<span style="background:{_roi_color};padding:4px 10px;'
    f'border-radius:6px;font-size:0.82rem;">📐 {_roi_badge}</span>',
    unsafe_allow_html=True,
)

if "app3_result" in st.session_state:
    _b = st.session_state["app3_result"]
    if _b.get("view") != selected_view:
        del st.session_state["app3_result"]

col_up, col_hint = st.columns([1, 2], gap="large")
with col_up:
    uploaded = st.file_uploader(
        "上传待测图片（PNG / JPG / BMP）",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="visible",
    )

with col_hint:
    if detector is None:
        st.info("请先选择视角")
    elif uploaded is None:
        st.info("请上传图片")

img_bgr: np.ndarray | None = None
upload_key = ""
if uploaded is not None:
    img_bgr = bytes_to_bgr(uploaded.read())
    if img_bgr is None:
        st.error("图片解码失败，请重新上传。")
        st.stop()
    upload_key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"

run_detect = False
if img_bgr is not None and detector is not None:
    run_detect = st.button("▶ 开始检测", type="primary", use_container_width=False)
    if not run_detect and "app3_result" not in st.session_state:
        st.info("点击开始检测")

_cache_valid = (
    "app3_result" in st.session_state
    and st.session_state["app3_result"].get("key") == upload_key
    and st.session_state["app3_result"].get("view") == selected_view
)

if img_bgr is None or detector is None:
    st.stop()

h_img, w_img = img_bgr.shape[:2]

if not run_detect and not _cache_valid:
    st.image(bgr_to_rgb(img_bgr), caption=f"原图  {w_img}×{h_img}px", use_container_width=True)
    st.stop()

if run_detect:
    calib_bbox = rel_to_abs(roi_rel, w_img, h_img) if roi_rel else None
    with st.spinner("Stage 1 检测中…"):
        r1 = detector.inspect_with_localization(
            img_bgr,
            min_match_count=match_thresh,
            missing_regions_fail_count=missing_thresh,
            calibrated_bbox=calib_bbox,
        )
    r2: dict | None = None
    if enable_s2 and r1["status"] == "PASS" and stage2_det is not None:
        with st.spinner("Stage 2 YOLO 检测中…"):
            r2 = stage2_det.inspect(img_bgr, r1)
    st.session_state["app3_result"] = {
        "r1": r1, "r2": r2, "key": upload_key, "view": selected_view,
    }
elif _cache_valid:
    _bundle = st.session_state["app3_result"]
    r1 = _bundle["r1"]
    r2 = _bundle.get("r2")

st.image(bgr_to_rgb(img_bgr), caption=f"原图  {w_img}×{h_img}px", use_container_width=True)

# ── 展示结果 ─────────────────────────────────────────────────────────────
step_header("Stage 1 — 结构初筛")
s1 = r1["status"]
st.markdown(
    verdict_card_html(
        s1,
        r1["issues"],
        r1["warnings"],
        title={
            "PASS": "Stage 1 通过",
            "FAIL": "Stage 1 不合格",
            "RETAKE": "Stage 1 需重拍",
        }.get(s1, s1),
    ),
    unsafe_allow_html=True,
)

st.markdown(
    metric_row([
        ("塑料膜阈值(侧栏)", f"{film_thresh}%"),
        ("定位方式", "已标定" if roi_rel else "自动"),
        ("覆膜率", f'{r1["film_coverage"] * 100:.1f}%'),
        ("相似度", f'{r1["similarity"] * 100:.0f}%'),
        ("缺件框", str(len(r1["missing_regions"]))),
        ("外侧标签", str(r1.get("outer_label_count", 0))),
        ("已滤杂质", str(r1.get("outer_label_removed_count", 0))),
    ]),
    unsafe_allow_html=True,
)

if show_ann and r1.get("annotated_image") is not None:
    st.image(
        bgr_to_rgb(r1["annotated_image"]),
        caption="Stage 1 标注图",
        use_container_width=True,
    )

st.markdown("---")
step_header("Stage 2 — 划痕/裂纹细筛")

if s1 != "PASS":
    st.markdown(
        verdict_card_html(
            "SKIP", [], [],
            title="已跳过（Stage 1 未通过）",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        verdict_card_html("FAIL", r1["issues"], [], title="综合结论：产品不合格"),
        unsafe_allow_html=True,
    )
    st.stop()

if not enable_s2:
    st.markdown(
        verdict_card_html(
            "SKIP", [], [],
            title="未启用 Stage 2 细筛",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        verdict_card_html("PASS", [], [], title="综合结论：初筛通过（细筛未启用）"),
        unsafe_allow_html=True,
    )
    st.stop()

if stage2_det is None:
    st.warning(f"找不到 ONNX 模型（{onnx_path.name}），跳过细筛。")
    st.markdown(
        verdict_card_html("PASS", [], [], title="综合结论：初筛通过（细筛已跳过）"),
        unsafe_allow_html=True,
    )
    st.stop()

if r2 is None:
    with st.spinner("Stage 2 YOLO 检测中…"):
        r2 = stage2_det.inspect(img_bgr, r1)
    st.session_state["app3_result"]["r2"] = r2

s2 = r2["status"]
n_total   = len(r2["defects"])
n_scratch = r2["defect_counts"].get("scratch", 0)
n_crack   = r2["defect_counts"].get("crack", 0)

st.markdown(
    verdict_card_html(
        s2,
        r2["issues"],
        [],
        title=(
            "细筛通过 — 表面无缺陷"
            if s2 == "PASS"
            else f"发现 {n_total} 处缺陷（划痕 {n_scratch} / 裂纹 {n_crack}）"
        ),
    ),
    unsafe_allow_html=True,
)

st.markdown(
    metric_row([
        ("划痕 scratch", str(n_scratch)),
        ("裂纹 crack", str(n_crack)),
        ("总缺陷数", str(n_total)),
        ("置信度阈值", f"{conf_thresh:.2f}"),
    ]),
    unsafe_allow_html=True,
)

if show_ann and r2.get("annotated_image") is not None:
    st.image(
        bgr_to_rgb(r2["annotated_image"]),
        caption="Stage 2 标注图",
        use_container_width=True,
    )

st.markdown("---")
final = "PASS" if s2["status"] == "PASS" else "FAIL"
final_msg = (
    "综合结论：两阶段均通过 — 产品合格"
    if final == "PASS"
    else "综合结论：细筛发现缺陷 — 产品不合格"
)
st.markdown(
    verdict_card_html(final, [], [], title=final_msg),
    unsafe_allow_html=True,
)
