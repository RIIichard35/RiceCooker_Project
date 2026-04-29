"""
电饭煲外观初筛系统 — Stage 1 前端（Streamlit）
整图检测版：无需 ROI 标定，含白色标签保留与姿态偏斜告警

运行方式：
    streamlit run gui/app1.py
"""

from __future__ import annotations

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
    ANOMALIB_MODEL_DIR,
    ANOMALIB_SCORE_THRESHOLD,
    ANOMALIB_HOLE_FAIL_THRESH,
    ANOMALIB_TEMPLATE_FAIL_THRESH,
    ANOMALIB_TEMPLATE_GATE_ENABLED,
    FILM_COVERAGE_THRESHOLD,
    get_anomalib_threshold,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage2_anomalib import AnomalibDetector

# ── 页面配置 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电饭煲初筛检测台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── 整体字体 ── */
html, body, [class*="css"] { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }

/* ── 结论卡片 ── */
.verdict-card {
    padding: 20px 28px; border-radius: 12px;
    text-align: center; margin-bottom: 16px;
}
.verdict-pass   { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.verdict-fail   { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
.verdict-retake { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }

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

/* ── 问题 / 警告列表 ── */
.issue-item { color: #721c24; margin: 3px 0; font-size: 0.88rem; }
.warn-item  { color: #856404; margin: 3px 0; font-size: 0.88rem; }

/* ── 图片说明标签 ── */
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


def cutout_to_display(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return bgra_to_rgba(img)
    return bgr_to_rgb(img)


def load_st_cutout(path: str) -> np.ndarray | None:
    """从 st/ 目录加载预处理透明抠图（BGRA → RGBA）。"""
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def verdict_html(status: str, issues: list[str], warnings: list[str]) -> str:
    css   = {"PASS": "verdict-pass", "FAIL": "verdict-fail", "RETAKE": "verdict-retake"}
    icon  = {"PASS": "✅", "FAIL": "❌", "RETAKE": "⚠️"}
    label = {"PASS": "初筛通过", "FAIL": "初筛不合格", "RETAKE": "需要重拍"}
    c = css.get(status, "verdict-fail")
    i = icon.get(status, "❓")
    l = label.get(status, status)
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
        f'<div class="metric-value">{val}</div>'
        f'</div>'
        for lb, val in metrics
    )
    return f'<div class="metric-row">{boxes}</div>'

def step_header(text: str) -> None:
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)

def make_label_overlay(bgr: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    """将标签区域以青黄色半透明叠加到原图。"""
    overlay = np.zeros_like(bgr)
    overlay[label_mask > 0] = (0, 220, 200)
    return cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)


# ── 视角映射 ─────────────────────────────────────────────────────────────
VIEW_LABELS = {"front": "正面", "back": "后面", "left": "左面",
               "right": "右面", "top": "顶部"}

std_root = Path(STANDARDS_DIR)
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
match_thresh = st.sidebar.slider(
    "ORB 匹配点", 5, 60, MIN_MATCH_COUNT,
)
missing_thresh = st.sidebar.slider(
    "缺件框阈值", 1, 10, MAX_MISSING_COUNT,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Anomalib 特征比对")
enable_anomalib = st.sidebar.checkbox(
    "启用 Anomalib 缺件检测", value=True,
)
anomalib_threshold = st.sidebar.slider(
    "异常分阈值", 0.10, 0.95,
    float(get_anomalib_threshold(selected_view) if selected_view else ANOMALIB_SCORE_THRESHOLD),
    0.05,
    key=f"anom_thr_{selected_view or 'none'}",
    help="默认随视角来自 config（ANOMALIB_THRESH_BY_VIEW）；可手动覆盖单会话",
)

st.sidebar.markdown("---")
show_anomalib_heatmap = st.sidebar.checkbox("显示异常热图", value=True)


# ── 加载检测器（缓存） ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载标准图库...")
def load_detector(view_name: str) -> Stage1Detector:
    return Stage1Detector(str(std_root / view_name))


@st.cache_resource(show_spinner="正在加载 Anomalib 模型...")
def load_anomalib_detector(view_name: str) -> AnomalibDetector:
    return AnomalibDetector(view=view_name, model_dir=ANOMALIB_MODEL_DIR)


detector: Stage1Detector | None = None
anomalib_det: AnomalibDetector | None = None
if selected_view:
    detector = load_detector(selected_view)
    ref_count = len(detector.reference_data)
    st.sidebar.success(f"已加载 **{selected_label}** 标准图库（{ref_count} 张）")

    if enable_anomalib:
        try:
            anomalib_det = load_anomalib_detector(selected_view)
            if anomalib_det.is_ready:
                st.sidebar.success(
                    f"Anomalib 模型已就绪（{anomalib_det._backend}）"
                )
            else:
                st.sidebar.warning(
                    "Anomalib 模型未找到，请先运行：\n"
                    "`python scripts/train_anomalib.py`"
                )
        except Exception as e:
            st.sidebar.error(f"Anomalib 加载失败: {e}")


# ── 主界面 ───────────────────────────────────────────────────────────────
st.title("🔍 电饭煲外观初筛系统")
st.caption("初筛结果展示")
st.markdown("---")

# ── 上传区 ───────────────────────────────────────────────────────────────
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
if uploaded:
    img_bgr = bytes_to_bgr(uploaded.read())
    if img_bgr is None:
        st.error("图片解码失败，请重新上传。")

# ── 手动触发检测（避免调整侧边栏时自动重跑抠图） ─────────────────────────
run_detect = False
if img_bgr is not None and detector is not None:
    run_detect = st.button("▶ 开始检测", type="primary", use_container_width=False)
    if not run_detect and "last_result" not in st.session_state:
        st.info("点击开始检测")

    # 有缓存结果时直接展示，避免重复推理
    if not run_detect and "last_result" in st.session_state:
        run_detect = False   # 展示由下方缓存逻辑接管

# ── 检测主流程 ───────────────────────────────────────────────────────────
if img_bgr is not None and detector is not None and (run_detect or "last_result" in st.session_state):
    h_img, w_img = img_bgr.shape[:2]

    if run_detect:
        with st.spinner("Stage 1 检测中…"):
            result = detector.inspect_with_localization(
                img_bgr,
                min_match_count=match_thresh,
                missing_regions_fail_count=missing_thresh,
            )
        # Anomalib 推理（Stage1 质量门控通过后，Anomalib 是主判决者）
        anomalib_result: dict | None = None
        if (
            enable_anomalib
            and anomalib_det is not None
            and anomalib_det.is_ready
            and result.get("status") in ("PASS", "FAIL")
        ):
            cutout = result.get("cutout_rgba")
            mask_crop = result.get("product_mask_crop")
            # 只取产品 bbox 区域喂给 Anomalib，坐标对齐裁剪图
            _pb = result.get("product_bbox")
            if cutout is not None and _pb is not None:
                _px1, _py1, _px2, _py2 = _pb
                if _px2 > _px1 and _py2 > _py1:
                    cutout = cutout[_py1:_py2, _px1:_px2]
                    # product_mask_crop 已在 Stage1 按同一 bbox 裁好，勿二次切片
            if cutout is not None and mask_crop is not None:
                if mask_crop.shape[:2] != cutout.shape[:2]:
                    mask_crop = cv2.resize(
                        mask_crop,
                        (cutout.shape[1], cutout.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            if cutout is not None:
                with st.spinner("Anomalib 特征比对中…"):
                    _det_copy = AnomalibDetector.__new__(AnomalibDetector)
                    _det_copy.__dict__.update(anomalib_det.__dict__)
                    _det_copy.threshold = anomalib_threshold
                    anomalib_result = _det_copy.inspect(cutout, stage1_fg_mask=mask_crop)
                result["anomalib_result"] = anomalib_result
                # 若 Anomalib 判 FAIL，更新顶层 status
                if anomalib_result.get("status") == "FAIL":
                    result["status"] = "FAIL"
                    result["issues"] = result.get("issues", []) + anomalib_result.get("issues", [])
        st.session_state["last_result"] = result
        st.session_state["last_anomalib"] = result.get("anomalib_result")
    else:
        result = st.session_state["last_result"]
        anomalib_result = st.session_state.get("last_anomalib")

    status       = result["status"]
    issues       = result["issues"]
    warnings     = result["warnings"]
    pose_angle   = result.get("pose_angle", 0.0)
    pose_warning = result.get("pose_warning")
    label_ratio  = result.get("label_area_ratio", 0.0) * 100
    loc_method   = result.get("localization_method", "自动")
    seg_backend  = result.get("segmentation_backend", "unknown")
    seg_reason   = result.get("segmentation_fallback_reason")
    film_pct     = result["film_coverage"] * 100
    loc_score    = result["localization_score"]


    ann_img         = result.get("annotated_image")
    cut_rgba        = result.get("cutout_rgba")
    ref_cutout_path = result.get("ref_cutout_path")
    lmask           = result.get("label_mask")

    # st/ 预处理透明抠图（最优先）
    st_cutout_rgba: np.ndarray | None = None
    if ref_cutout_path:
        st_cutout_rgba = load_st_cutout(ref_cutout_path)

    # ════════════════════════════════════════════════════════════════════
    # 区域一：三图并列——原图 | 抠图 | 标注图
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### 图像")
    c_orig, c_cut, c_ann = st.columns(3, gap="medium")

    with c_orig:
        step_header("① 输入原图")
        st.image(bgr_to_rgb(img_bgr),
                 caption="原图",
                 use_container_width=True)

    with c_cut:
        step_header("② 参考图")
        if st_cutout_rgba is not None:
            st.image(
                st_cutout_rgba,
                caption="参考抠图（标准库）",
                use_container_width=True,
            )
        elif cut_rgba is not None:
            st.image(
                bgra_to_rgba(cut_rgba),
                caption="产品抠图",
                use_container_width=True,
            )
        else:
            st.info("参考图缺失")

    with c_ann:
        step_header("③ 检测结果")
        _clean = (anomalib_result or {}).get("clean_cutout")
        if _clean is not None and _clean.any():
            # 渲染灰色背景 BGR
            if _clean.ndim == 3 and _clean.shape[2] == 4:
                _a = _clean[:, :, 3:4].astype(np.float32) / 255.0
                _bgr_base = _clean[:, :, :3].astype(np.float32)
                _bg = np.full_like(_bgr_base, 128.0)
                final_ann = (_a * _bgr_base + (1 - _a) * _bg).astype(np.uint8)
            else:
                final_ann = _clean[:, :, :3].copy()

            _h, _w = final_ann.shape[:2]

            if anomalib_result is not None:
                a_boxes   = anomalib_result.get("anomaly_boxes", [])
                h_boxes   = anomalib_result.get("hole_boxes", [])
                a_status  = anomalib_result.get("status", "SKIP")
                a_score   = anomalib_result.get("anomaly_score", 0.0)
                h_score   = anomalib_result.get("hole_score", 0.0)

                # 半透明红色蒙版覆盖异常框区域
                overlay_layer = final_ann.copy()
                all_defect_boxes = list(a_boxes)
                for hb in h_boxes:
                    if not any(
                        max(hb[0], eb[0]) < min(hb[2], eb[2]) and
                        max(hb[1], eb[1]) < min(hb[3], eb[3])
                        for eb in all_defect_boxes
                    ):
                        all_defect_boxes.append(hb)

                for box in all_defect_boxes:
                    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(overlay_layer, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
                # 40% 透明度红色蒙版
                final_ann = cv2.addWeighted(overlay_layer, 0.38, final_ann, 0.62, 0)

                # 实线红框 + 标注文字
                for idx, box in enumerate(all_defect_boxes):
                    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    is_hole = any(
                        hb[0] == box[0] and hb[1] == box[1] for hb in h_boxes
                    )
                    label = "破损" if is_hole else f"异常{idx+1}"
                    color = (0, 0, 230) if is_hole else (30, 30, 220)
                    cv2.rectangle(final_ann, (bx1, by1), (bx2, by2), color, 2)
                    # 标签背景
                    txt_y = max(by1 - 4, 18)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(final_ann, (bx1, txt_y - th - 4), (bx1 + tw + 4, txt_y + 2), color, -1)
                    cv2.putText(final_ann, label, (bx1 + 2, txt_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 整体产品黄框
                cv2.rectangle(final_ann, (2, 2), (_w - 3, _h - 3), (0, 215, 255), 3)

                # 底部状态文字
                s_color = (0, 180, 0) if a_status == "PASS" else (
                    (0, 0, 220) if a_status == "FAIL" else (180, 140, 0)
                )
                status_txt = f"{a_status}  破损:{h_score*100:.1f}%  综合:{a_score:.3f}"
                cv2.putText(final_ann, status_txt,
                            (8, max(20, _h - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_color, 2)

            caption_txt = "黄框=产品  红框=破损/异常区域" if anomalib_result else "检测图"
            st.image(bgr_to_rgb(final_ann), caption=caption_txt, use_container_width=True)

        elif ann_img is not None:
            st.image(bgr_to_rgb(ann_img), caption="检测图", use_container_width=True)
        else:
            st.info("标注图未生成")

    # ── Anomalib 热图 + 破洞专项展示 ────────────────────────────────────
    anomalib_result = result.get("anomalib_result") or st.session_state.get("last_anomalib")
    if show_anomalib_heatmap and anomalib_result is not None:
        a_status  = anomalib_result.get("status", "SKIP")
        h_score   = anomalib_result.get("hole_score", 0.0)
        h_boxes   = anomalib_result.get("hole_boxes", [])
        a_overlay = anomalib_result.get("heatmap_overlay")
        pc_score  = anomalib_result.get("patchcore_score")
        tpl_score = anomalib_result.get("template_score")
        if pc_score is None:
            pc_score = float(anomalib_result.get("anomaly_score", 0.0))
        else:
            pc_score = float(pc_score)
        if tpl_score is None:
            tpl_score = 0.0
        else:
            tpl_score = float(tpl_score)
        tpl_thr = float(ANOMALIB_TEMPLATE_FAIL_THRESH)
        combo_max = float(
            max(
                pc_score,
                tpl_score if ANOMALIB_TEMPLATE_GATE_ENABLED else 0.0,
                float(h_score) * 5.0,
            )
        )

        # ── 破洞专项告警栏 ────────────────────────────────────────────
        if h_score > ANOMALIB_HOLE_FAIL_THRESH:
            hole_pct = h_score * 100
            st.markdown(
                f"""
<div style="background:#fff0f0; border:2px solid #dc3545; border-radius:10px;
     padding:14px 18px; margin:10px 0; display:flex; align-items:center; gap:14px;">
  <span style="font-size:2rem;">⚠️</span>
  <div>
    <div style="font-size:1.05rem; font-weight:700; color:#dc3545;">
      检测到明显破损 / 破洞区域
    </div>
    <div style="font-size:0.88rem; color:#555; margin-top:4px;">
      破损面积占比 <b>{hole_pct:.1f}%</b> · 共 <b>{len(h_boxes)}</b> 处
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("### 异常热图")
        col_heat, col_ainfo = st.columns([1.4, 1], gap="large")

        with col_heat:
            step_header("热图（JET：蓝→红 异常程度递增）")
            if a_overlay is not None:
                st.image(
                    bgr_to_rgb(a_overlay),
                    caption=(
                        f"PatchCore {pc_score:.3f} | 模板 {tpl_score:.3f}"
                        f"{' (参与判阈 {:.2f})'.format(tpl_thr) if ANOMALIB_TEMPLATE_GATE_ENABLED else ' (仅展示)'} "
                        f"| 破洞 {h_score*100:.1f}% | {a_status}"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("热图未生成")

        with col_ainfo:
            step_header("关键分数")
            pc_color = "#dc3545" if pc_score > anomalib_threshold else (
                "#ffc107" if pc_score > anomalib_threshold * 0.75 else "#28a745"
            )
            if ANOMALIB_TEMPLATE_GATE_ENABLED:
                tpl_color = "#dc3545" if tpl_score > tpl_thr else (
                    "#ffc107" if tpl_score > tpl_thr * 0.75 else "#28a745"
                )
            else:
                tpl_color = "#6c757d"
            hole_color = "#dc3545" if h_score > ANOMALIB_HOLE_FAIL_THRESH else "#28a745"
            st.markdown(
                f"""
<div style="text-align:center; padding:12px; border-radius:10px;
     background:#f8f9fa; border: 2px solid {pc_color}; margin-bottom:8px">
  <div style="font-size:0.78rem; color:#666; margin-bottom:2px">PatchCore 异常分</div>
  <div style="font-size:1.5rem; font-weight:700; color:{pc_color}">{pc_score:.3f}</div>
  <div style="font-size:0.82rem; color:#444; margin-top:4px">阈值 {anomalib_threshold:.2f}</div>
</div>
<div style="text-align:center; padding:12px; border-radius:10px;
     background:#f8f9fa; border: 2px solid {tpl_color}; margin-bottom:8px">
  <div style="font-size:0.78rem; color:#666; margin-bottom:2px">模板一致性（ORB 残差）</div>
  <div style="font-size:1.5rem; font-weight:700; color:{tpl_color}">{tpl_score:.3f}</div>
  <div style="font-size:0.82rem; color:#444; margin-top:4px">阈值 {tpl_thr:.2f}
    ({'参与终判' if ANOMALIB_TEMPLATE_GATE_ENABLED else '仅展示，不参与终判'})</div>
</div>
<div style="text-align:center; padding:10px; border-radius:10px;
     background:#fafafa; border: 1px dashed #bbb; margin-bottom:10px; font-size:0.78rem; color:#555">
  综合展示 max {combo_max:.3f}
  （{'PatchCore、模板、破洞×5 取最大后按分项 OR 判 FAIL' if ANOMALIB_TEMPLATE_GATE_ENABLED else '当前未计模板；PatchCore 与 破洞×5 取最大，按 PatchCore/破洞 OR 判 FAIL'}）
</div>
<div style="text-align:center; padding:12px; border-radius:10px;
     background:#f8f9fa; border: 2px solid {hole_color}; margin-bottom:10px">
  <div style="font-size:0.78rem; color:#666; margin-bottom:2px">破损面积占比</div>
  <div style="font-size:2.0rem; font-weight:700; color:{hole_color}">{h_score*100:.1f}%</div>
  <div style="font-size:0.82rem; color:#444; margin-top:4px">共 {len(h_boxes)} 处</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if a_status == "FAIL" and anomalib_result.get("issues"):
                step_header("不通过原因")
                for iss in anomalib_result["issues"]:
                    st.warning(iss)
            if a_status == "FAIL":
                st.error("检测不通过（见上方分项与原因）")
            elif a_status == "PASS":
                st.success("检测通过")
            else:
                st.info("未检测")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # 区域二：综合判定结果
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### 结论")

    col_verdict, col_metrics = st.columns([1, 1.6], gap="large")

    with col_verdict:
        st.markdown(verdict_html(status, issues[:1], []), unsafe_allow_html=True)

    with col_metrics:
        step_header("关键指标")
        _a_score_str = (
            f"{anomalib_result.get('anomaly_score', 0.0):.3f}"
            if anomalib_result else "—"
        )
        _a_status_str = (
            anomalib_result.get("status", "SKIP")
            if anomalib_result else "未运行"
        )
        st.markdown(
            metric_row([
                ("Anomalib分",  _a_score_str),
                ("Anomalib判定", _a_status_str),
                ("覆膜率",      f"{film_pct:.1f}%"),
                ("偏斜角",      f"{pose_angle:.1f}°"),
            ]),
            unsafe_allow_html=True,
        )

        # Anomalib 异常框列表
        if anomalib_result and anomalib_result.get("anomaly_boxes"):
            step_header("异常区域")
            for i, box in enumerate(anomalib_result["anomaly_boxes"], 1):
                x1, y1, x2, y2 = box
                st.markdown(
                    f"异常{i}: ({x1}, {y1}) → ({x2}, {y2})"
                )
