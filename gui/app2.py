"""
电饭煲外观检测系统 — Stage 2 细筛前端（Streamlit）
===================================================
流程：Stage 1 初筛（结构完整性）→ 通过 → Stage 2 细筛（划痕/裂纹）

运行方式：
    streamlit run gui/app2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── 路径（相对路径，兼容树莓派） ────────────────────────────────────────
GUI_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FILM_COVERAGE_THRESHOLD,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    MODELS_DIR,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage3_yolov8 import Stage2Detector

# ── 默认路径 ─────────────────────────────────────────────────────────────
_ONNX_PATH  = Path(MODELS_DIR) / "best20240919.onnx"
_STD_ROOT   = Path(STANDARDS_DIR)

# ── 页面配置 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电饭煲细筛检测台",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── 整体字体（与 app1 一致） ── */
html, body, [class*="css"] { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }

/* ── 结论卡片 ── */
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

/* ── 指标行 ── */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 14px; }
.metric-box {
    flex: 1; min-width: 100px; background: #f0f2f6;
    border-radius: 8px; padding: 10px 6px; text-align: center;
}
.metric-label { font-size: 0.70rem; color: #666; margin-bottom: 4px; }
.metric-value { font-size: 1.15rem; font-weight: 700; color: #222; }

/* ── 流程步骤标题 ── */
.step-header {
    font-size: 0.85rem; font-weight: 700; color: #0078d4;
    border-left: 3px solid #0078d4; padding-left: 8px;
    margin: 14px 0 6px;
}

.issue-item { color: #721c24; margin: 3px 0; font-size: 0.88rem; }
.warn-item  { color: #856404; margin: 3px 0; font-size: 0.88rem; }

/* 细筛页：展示图下边距收紧 */
section.main [data-testid="stImage"] { margin-bottom: 0.15rem !important; }
section.main div[data-testid="stMarkdownContainer"] hr {
  margin: 0.35rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ─────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgra_to_rgba(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def cutout_to_display(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return bgra_to_rgba(img)
    return bgr_to_rgb(img)


def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def verdict_card_html(
    status: str,
    issues: list[str],
    warnings: list[str],
    title: str = "",
) -> str:
    """与 app1 verdict 样式一致；可传入 title 覆盖默认文案（细筛多阶段）。"""
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


def stage2_crop_with_defects(
    crop_bgr: np.ndarray,
    defects: list,
    crop_box: list | None,
) -> np.ndarray:
    """在 Stage2 产品裁剪图（BGR）上绘制缺陷框，坐标从全图换算为裁剪图局部。"""
    if crop_bgr is None or crop_bgr.size == 0 or not crop_box:
        return crop_bgr
    vis = crop_bgr.copy()
    ch, cw = vis.shape[:2]
    cx1, cy1, _, _ = (int(crop_box[0]), int(crop_box[1]),
                      int(crop_box[2]), int(crop_box[3]))
    for d in defects or []:
        ox1, oy1, ox2, oy2 = (int(d["xyxy"][0]), int(d["xyxy"][1]),
                              int(d["xyxy"][2]), int(d["xyxy"][3]))
        lx1 = ox1 - cx1
        ly1 = oy1 - cy1
        lx2 = ox2 - cx1
        ly2 = oy2 - cy1
        lx1 = max(0, min(lx1, cw - 1))
        ly1 = max(0, min(ly1, ch - 1))
        lx2 = max(0, min(lx2, cw))
        ly2 = max(0, min(ly2, ch))
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        cls_name = d.get("class_name", "")
        score = float(d.get("score", 0.0))
        color = (0, 0, 220) if cls_name == "scratch" else (0, 100, 255)
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 2)
        label = f"{cls_name} {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lab_y = ly1 - 4
        if lab_y < lh + 8:
            lab_y = min(ch - 1, ly2 + lh + 8)
        cv2.rectangle(
            vis,
            (lx1, lab_y - lh - 6),
            (lx1 + lw + 4, lab_y),
            color,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (lx1 + 2, lab_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    return vis


# ── 视角映射 ─────────────────────────────────────────────────────────────
VIEW_LABELS = {"front": "正面", "back": "后面", "left": "左面",
               "right": "右面", "top": "顶部"}

if _STD_ROOT.exists():
    available_views = sorted(
        d.name for d in _STD_ROOT.iterdir()
        if d.is_dir() and any(d.iterdir())
    )
else:
    available_views = []

view_display = [VIEW_LABELS.get(v, v) for v in available_views]
view_map     = dict(zip(view_display, available_views))


# ── 侧边栏 ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 检测配置")

selected_label: str | None = st.sidebar.selectbox(
    "检测视角", options=view_display,
    index=0 if view_display else None,
)
selected_view = view_map.get(selected_label, selected_label) if selected_label else None

st.sidebar.markdown("---")
st.sidebar.subheader("检测阈值（Stage 1）")
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
show_crop   = st.sidebar.checkbox(
    "Stage 2 横向显示裁剪图", value=True,
    help="开启时与全图标注左右并列；关闭时仅显示全图标注。",
)
show_detail = st.sidebar.checkbox("显示调试详情", value=False)


# ── 加载检测器（缓存） ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="正在加载标准图库...")
def get_stage1(view_name: str) -> Stage1Detector:
    return Stage1Detector(str(_STD_ROOT / view_name))

@st.cache_resource(show_spinner="正在加载 ONNX 模型...")
def get_stage2(conf: float, iou: float, pad: float) -> Stage2Detector | None:
    if not _ONNX_PATH.exists():
        return None
    return Stage2Detector(
        str(_ONNX_PATH),
        conf_threshold=conf,
        iou_threshold=iou,
        roi_padding=pad,
    )

if selected_view:
    det1 = get_stage1(selected_view)
    st.sidebar.success(
        f"已加载 **{selected_label}** 标准图库（{len(det1.reference_data)} 张）"
    )
else:
    det1 = None

det2 = get_stage2(conf_thresh, iou_thresh, roi_padding)
if det2:
    st.sidebar.success("ONNX 模型已加载")
else:
    st.sidebar.error(f"找不到模型：{_ONNX_PATH.name}")


# ── 主界面 ───────────────────────────────────────────────────────────────
st.title("🔎 电饭煲外观细筛系统（Stage 1 → Stage 2）")
st.caption("整图检测 · 标签保留 · 姿态评估 → Stage 1 初筛 → Stage 2 YOLO 划痕/裂纹检测")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════
# 检测
# ════════════════════════════════════════════════════════════════════════
col_up, col_res = st.columns([1, 1.8], gap="large")

# ── 左：上传 ─────────────────────────────────────────────────────────
with col_up:
    st.subheader("① 上传待测图片")
    uploaded = st.file_uploader(
        "拖拽或点击上传（PNG / JPG / BMP）",
        type=["png","jpg","jpeg","bmp"],
        label_visibility="collapsed",
    )
    img_bgr: np.ndarray | None = None
    if uploaded:
        img_bgr = bytes_to_bgr(uploaded.read())
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            st.image(bgr_to_rgb(img_bgr),
                     caption=f"待测原图  {w}×{h}px", width="stretch")
        else:
            st.error("图片解码失败，请重新上传。")

# ── 右：结果 ─────────────────────────────────────────────────────────
with col_res:
    st.subheader("② 检测结果")

    if img_bgr is None:
        st.info("👈 请先上传待测图片")
    elif det1 is None:
        st.warning("请在左侧选择检测视角。")
    elif det2 is None:
        st.error(f"ONNX 模型文件不存在：{_ONNX_PATH}")
    else:
        r2: dict | None = None

        # ════════════════════════════════════════
        # Stage 1
        # ════════════════════════════════════════
        step_header("Stage 1 — 结构初筛")

        with st.spinner("Stage 1 检测中…"):
            r1 = det1.inspect_with_localization(
                img_bgr,
                min_match_count=match_thresh,
                missing_regions_fail_count=missing_thresh,
                calibrated_bbox=None,
            )

        s1_status = r1["status"]
        st.markdown(
            verdict_card_html(
                s1_status,
                r1["issues"],
                r1["warnings"],
                title={
                    "PASS":   "Stage 1 通过 → 进入细筛",
                    "FAIL":   "Stage 1 不合格 → 终止",
                    "RETAKE": "Stage 1 需重拍 → 终止",
                }.get(s1_status, s1_status),
            ),
            unsafe_allow_html=True,
        )

        pose_angle   = r1.get("pose_angle", 0.0)
        pose_warning = r1.get("pose_warning")
        label_ratio  = r1.get("label_area_ratio", 0.0) * 100
        loc_method   = r1.get("localization_method", "自动")

        if pose_warning:
            if pose_angle >= 28.0:
                st.error(f"🔴 Stage1 姿态告警：{pose_warning}")
            else:
                st.warning(f"⚠️ Stage1 姿态提示：{pose_warning}")

        st.markdown(
            metric_row([
                ("定位方式",   loc_method),
                ("覆膜率",     f"{r1['film_coverage']*100:.1f}%"),
                ("相似度",     f"{r1['similarity']*100:.0f}%"),
                ("定位置信度", f"{r1['localization_score']:.3f}"),
                ("缺件框",     f"{len(r1['missing_regions'])}"),
                ("标签区域",   f"{label_ratio:.1f}%"),
                ("偏斜角",     f"{pose_angle:.1f}°"),
            ]),
            unsafe_allow_html=True,
        )

        ann1 = r1.get("annotated_image")
        cut1 = r1.get("cutout_image")
        cut_rgba1 = r1.get("cutout_rgba")
        if ann1 is not None or cut1 is not None or cut_rgba1 is not None:
            c_stage1_left, c_stage1_mid, c_stage1_right = st.columns(3, gap="medium")
            with c_stage1_left:
                if cut_rgba1 is not None:
                    st.image(
                        bgra_to_rgba(cut_rgba1),
                        caption="Stage 1 透明抠图",
                        width="stretch",
                    )
                else:
                    st.info("透明抠图未生成")
            with c_stage1_mid:
                if cut1 is not None:
                    st.image(
                        cutout_to_display(cut1),
                        caption="Stage 1 透明抠图（兼容字段）",
                        width="stretch",
                    )
                else:
                    st.info("兼容抠图未生成")
            with c_stage1_right:
                if ann1 is not None:
                    st.image(
                        bgr_to_rgb(ann1),
                        caption="Stage 1 标注图（黄框=产品区域）",
                        width="stretch",
                    )
                else:
                    st.info("标注图未生成")

        # ════════════════════════════════════════
        # Stage 2（仅在 Stage 1 PASS 时执行）
        # ════════════════════════════════════════
        step_header("Stage 2 — 划痕/裂纹细筛")

        if s1_status != "PASS":
            st.markdown(
                verdict_card_html(
                    "SKIP", [], [],
                    title="已跳过（Stage 1 未通过）",
                ),
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Stage 2 YOLO 检测中…"):
                r2 = det2.inspect(img_bgr, r1)

            s2_status = r2["status"]
            n_scratch  = r2["defect_counts"].get("scratch", 0)
            n_crack    = r2["defect_counts"].get("crack",   0)
            n_total    = len(r2["defects"])

            st.markdown(
                verdict_card_html(
                    s2_status,
                    r2["issues"],
                    [],
                    title={
                        "PASS": "表面无缺陷 — 合格",
                        "FAIL": f"发现 {n_total} 处表面缺陷",
                    }.get(s2_status, s2_status),
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                metric_row([
                    ("划痕 scratch", str(n_scratch)),
                    ("裂纹 crack",   str(n_crack)),
                    ("总缺陷数",     str(n_total)),
                    ("置信度阈值",   f"{conf_thresh:.2f}"),
                ]),
                unsafe_allow_html=True,
            )

            ann2 = r2.get("annotated_image")
            crop2 = r2.get("crop_image")
            box2 = r2.get("crop_box")

            if ann2 is not None and show_crop and crop2 is not None:
                c2_left, c2_right = st.columns(2, gap="small")
                with c2_left:
                    st.image(
                        bgr_to_rgb(ann2),
                        caption="全图标注（蓝=划痕 | 橙=裂纹 | 黄=产品 ROI）",
                        width="stretch",
                    )
                with c2_right:
                    crop_vis = stage2_crop_with_defects(
                        crop2, r2.get("defects") or [], box2,
                    )
                    st.image(
                        bgr_to_rgb(crop_vis),
                        caption="产品裁剪 ROI（与 YOLO 输入一致；框为局部坐标）",
                        width="stretch",
                    )
            elif ann2 is not None:
                st.image(
                    bgr_to_rgb(ann2),
                    caption="Stage 2 标注图（蓝框=划痕 | 橙框=裂纹 | 黄框=产品区域）",
                    width="stretch",
                )
            elif show_crop and crop2 is not None:
                crop_vis = stage2_crop_with_defects(
                    crop2, r2.get("defects") or [], box2,
                )
                st.image(
                    bgr_to_rgb(crop_vis),
                    caption="产品裁剪 ROI",
                    width="stretch",
                )

            if show_detail and r2["defects"]:
                st.markdown("**缺陷详情：**")
                for i, d in enumerate(r2["defects"], 1):
                    b = d["xyxy"]
                    st.markdown(
                        f"- [{i}] **{d['class_name']}**  "
                        f"置信度={d['score']:.3f}  "
                        f"坐标=({b[0]},{b[1]})→({b[2]},{b[3]})  "
                        f"面积={d['area']}px²"
                    )

        # ── 综合结论 ─────────────────────────────────────────────────
        st.markdown("---")
        if s1_status == "PASS" and r2 is not None:
            final = "PASS" if r2["status"] == "PASS" else "FAIL"
            final_msg = {
                "PASS": "两阶段检测全部通过，产品合格",
                "FAIL": "Stage 2 细筛发现表面缺陷，产品不合格",
            }[final]
            st.markdown(
                verdict_card_html(final, [], [], title=final_msg),
                unsafe_allow_html=True,
            )
        elif s1_status in ("FAIL", "RETAKE"):
            st.markdown(
                verdict_card_html(
                    "FAIL",
                    r1["issues"],
                    [],
                    title="Stage 1 未通过，产品不合格",
                ),
                unsafe_allow_html=True,
            )

