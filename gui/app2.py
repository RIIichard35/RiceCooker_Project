"""
电饭煲外观检测系统 — Stage 2 细筛前端（Streamlit）
===================================================
流程：Stage 1 初筛（结构完整性）→ 通过 → Stage 2 细筛（划痕/裂纹）

运行方式：
    streamlit run gui/app2.py
"""

from __future__ import annotations

import json
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
    CALIB_DIR,
    FILM_COVERAGE_THRESHOLD,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    MODELS_DIR,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector
from src.stage2_scratch import Stage2Detector

# ── 默认路径 ─────────────────────────────────────────────────────────────
_ONNX_PATH  = Path(MODELS_DIR) / "best20240919.onnx"
_STD_ROOT   = Path(STANDARDS_DIR)
_CALIB_DIR  = Path(CALIB_DIR)

# ── 页面配置 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电饭煲细筛检测台",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card { padding:18px 22px; border-radius:10px; text-align:center;
        margin-bottom:14px; font-family:"Microsoft YaHei",sans-serif; }
.card-pass   { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
.card-fail   { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
.card-retake { background:#fff3cd; color:#856404; border:1px solid #ffeeba; }
.card-skip   { background:#e2e3e5; color:#383d41; border:1px solid #d6d8db; }

.metric-row { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }
.metric-box { flex:1; min-width:100px; background:#f0f2f6;
              border-radius:8px; padding:10px 8px; text-align:center; }
.metric-label { font-size:0.72rem; color:#555; margin-bottom:4px; }
.metric-value { font-size:1.15rem; font-weight:700; color:#222; }

.stage-divider { font-size:1.1rem; font-weight:700; color:#444;
                 border-left:4px solid #0078d4; padding-left:10px;
                 margin:16px 0 8px; }
.issue-item { color:#721c24; margin:3px 0; }
.warn-item  { color:#856404; margin:3px 0; }
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ─────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def card_html(status: str, issues: list[str], warnings: list[str],
              title_override: str = "") -> str:
    css   = {"PASS":"card-pass","FAIL":"card-fail",
             "RETAKE":"card-retake","SKIP":"card-skip"}
    icon  = {"PASS":"✅","FAIL":"❌","RETAKE":"⚠️","SKIP":"⏭️"}
    label = {"PASS":"通过","FAIL":"不合格","RETAKE":"需要重拍","SKIP":"已跳过"}
    c = css.get(status, "card-fail")
    i = icon.get(status, "❓")
    l = title_override or label.get(status, status)
    issues_html = "".join(
        f'<p class="issue-item">⛔ {s}</p>' for s in issues
    ) if issues else '<p style="color:#155724">无问题</p>'
    warns_html = "".join(
        f'<p class="warn-item">⚠ {w}</p>' for w in warnings
    ) if warnings else ""
    return (f'<div class="card {c}"><h2 style="margin:0 0 8px">{i} {l}</h2>'
            f'{issues_html}{warns_html}</div>')

def metric_html(metrics: list[tuple[str, str]]) -> str:
    boxes = "".join(
        f'<div class="metric-box">'
        f'<div class="metric-label">{lb}</div>'
        f'<div class="metric-value">{val}</div></div>'
        for lb, val in metrics
    )
    return f'<div class="metric-row">{boxes}</div>'

def load_calibration(view: str) -> list | None:
    p = _CALIB_DIR / f"{view}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("roi_rel")
    except Exception:
        return None

def calib_to_abs(roi_rel: list, w: int, h: int) -> list:
    return [int(roi_rel[0]*w), int(roi_rel[1]*h),
            int(roi_rel[2]*w), int(roi_rel[3]*h)]

def save_calibration(view: str, roi_rel: list, ref_wh: tuple) -> None:
    from datetime import datetime
    _CALIB_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "view":       view,
        "roi_rel":    roi_rel,
        "ref_size":   list(ref_wh),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (_CALIB_DIR / f"{view}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def draw_roi_preview(img_bgr: np.ndarray, roi_rel: list) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = calib_to_abs(roi_rel, w, h)
    vis     = img_bgr.copy()
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(vis, f"ROI  {x2-x1}x{y2-y1}px",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 200, 0), 2)
    return vis


# ── 视角映射 ─────────────────────────────────────────────────────────────
VIEW_LABELS = {"front":"正面","back":"后面","left":"左面",
               "right":"右面","top":"顶部"}

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

# 标定状态
if selected_view:
    roi_rel = load_calibration(selected_view)
    badge = "📐 ROI 已标定" if roi_rel else "📐 ROI 未标定（自动定位）"
    color = "#d4edda" if roi_rel else "#fff3cd"
    st.sidebar.markdown(
        f'<span style="background:{color};padding:4px 10px;'
        f'border-radius:6px;font-size:0.82rem;">{badge}</span>',
        unsafe_allow_html=True,
    )
else:
    roi_rel = None

st.sidebar.markdown("---")
st.sidebar.subheader("Stage 1 阈值")
film_thresh    = st.sidebar.slider("塑料膜阈值 (%)", 10, 90,
                                   int(FILM_COVERAGE_THRESHOLD * 100), 5)
match_thresh   = st.sidebar.slider("ORB 匹配点", 5, 60, MIN_MATCH_COUNT)
missing_thresh = st.sidebar.slider("缺件框阈值", 1, 10, MAX_MISSING_COUNT)

st.sidebar.markdown("---")
st.sidebar.subheader("Stage 2 阈值")
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
show_crop   = st.sidebar.checkbox("显示产品裁剪图", value=True)
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
st.caption("先通过 Stage 1 结构初筛，通过后在产品区域内执行 YOLO 划痕/裂纹检测")
st.markdown("---")

tab_detect, tab_calib = st.tabs(["🔎  检测", "📐  ROI 标定"])

# ════════════════════════════════════════════════════════════════════════
# Tab 1 — 检测
# ════════════════════════════════════════════════════════════════════════
with tab_detect:
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
            h_img, w_img = img_bgr.shape[:2]
            calib_bbox = calib_to_abs(roi_rel, w_img, h_img) if roi_rel else None

            # ════════════════════════════════════════
            # Stage 1
            # ════════════════════════════════════════
            st.markdown('<div class="stage-divider">Stage 1 — 结构初筛</div>',
                        unsafe_allow_html=True)

            with st.spinner("Stage 1 检测中..."):
                r1 = det1.inspect_with_localization(
                    img_bgr,
                    min_match_count=match_thresh,
                    missing_regions_fail_count=missing_thresh,
                    calibrated_bbox=calib_bbox,
                )

            s1_status = r1["status"]
            st.markdown(
                card_html(s1_status, r1["issues"], r1["warnings"],
                          title_override={
                              "PASS":   "Stage 1 通过 → 进入细筛",
                              "FAIL":   "Stage 1 不合格 → 终止",
                              "RETAKE": "Stage 1 需重拍 → 终止",
                          }.get(s1_status, s1_status)),
                unsafe_allow_html=True,
            )

            st.markdown(
                metric_html([
                    ("定位方式", "已标定" if roi_rel else "自动"),
                    ("覆膜率",   f"{r1['film_coverage']*100:.1f}%"),
                    ("相似度",   f"{r1['similarity']*100:.0f}%"),
                    ("定位置信", f"{r1['localization_score']:.3f}"),
                    ("异常框",   f"{len(r1['missing_regions'])}"),
                ]),
                unsafe_allow_html=True,
            )

            ann1 = r1.get("annotated_image")
            if ann1 is not None:
                st.image(bgr_to_rgb(ann1),
                         caption="Stage 1 标注图（黄框=产品区域）",
                         width="stretch")

            # ════════════════════════════════════════
            # Stage 2（仅在 Stage 1 PASS 时执行）
            # ════════════════════════════════════════
            st.markdown('<div class="stage-divider">Stage 2 — 划痕/裂纹细筛</div>',
                        unsafe_allow_html=True)

            if s1_status != "PASS":
                st.markdown(
                    card_html("SKIP", [], [],
                              title_override="已跳过（Stage 1 未通过）"),
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Stage 2 YOLO 检测中..."):
                    r2 = det2.inspect(img_bgr, r1)

                s2_status = r2["status"]
                n_scratch  = r2["defect_counts"].get("scratch", 0)
                n_crack    = r2["defect_counts"].get("crack",   0)
                n_total    = len(r2["defects"])

                st.markdown(
                    card_html(
                        s2_status, r2["issues"], [],
                        title_override={
                            "PASS": "表面无缺陷 — 合格",
                            "FAIL": f"发现 {n_total} 处表面缺陷",
                        }.get(s2_status, s2_status),
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    metric_html([
                        ("划痕 scratch", str(n_scratch)),
                        ("裂纹 crack",   str(n_crack)),
                        ("总缺陷数",     str(n_total)),
                        ("置信度阈值",   f"{conf_thresh:.2f}"),
                    ]),
                    unsafe_allow_html=True,
                )

                ann2 = r2.get("annotated_image")
                if ann2 is not None:
                    st.image(
                        bgr_to_rgb(ann2),
                        caption="Stage 2 标注图（蓝框=划痕 | 橙框=裂纹 | 黄框=产品区域）",
                        width="stretch",
                    )

                if show_crop:
                    crop = r2.get("crop_image")
                    if crop is not None:
                        st.markdown("**产品裁剪图（Stage 2 输入）**")
                        st.image(bgr_to_rgb(crop),
                                 caption="产品 ROI（含 padding）",
                                 width="stretch")

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
            if s1_status == "PASS" and "r2" in dir():
                final = "PASS" if r2["status"] == "PASS" else "FAIL"
                final_msg = {
                    "PASS": "两阶段检测全部通过，产品合格",
                    "FAIL": "Stage 2 细筛发现表面缺陷，产品不合格",
                }[final]
                st.markdown(
                    card_html(final, [], [], title_override=final_msg),
                    unsafe_allow_html=True,
                )
            elif s1_status in ("FAIL", "RETAKE"):
                st.markdown(
                    card_html("FAIL", r1["issues"], [],
                              title_override="Stage 1 未通过，产品不合格"),
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════
# Tab 2 — ROI 标定
# ════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.markdown(
        "通过拖动滑块标定**固定产品区域（ROI）**，"
        "保存后检测时会跳过自动定位，直接使用该区域。"
    )

    if not available_views:
        st.error("未找到任何视角的标准图库，无法标定。")
    else:
        c_sel, c_prev = st.columns([1, 1.5], gap="large")

        with c_sel:
            calib_view_label = st.selectbox(
                "选择要标定的视角",
                options=view_display,
                key="calib_view_sel",
            )
            calib_view = view_map.get(calib_view_label, calib_view_label)

            # 读取当前标定值（如有）
            existing = load_calibration(calib_view) or [0.10, 0.10, 0.90, 0.90]

            st.markdown("#### 拖动滑块调整 ROI 边界（单位：图像宽/高比例）")
            roi_x1 = st.slider("左边界 X1", 0.00, 0.90, float(existing[0]), 0.01,
                                key="rx1")
            roi_y1 = st.slider("上边界 Y1", 0.00, 0.90, float(existing[1]), 0.01,
                                key="ry1")
            roi_x2 = st.slider("右边界 X2", 0.10, 1.00, float(existing[2]), 0.01,
                                key="rx2")
            roi_y2 = st.slider("下边界 Y2", 0.10, 1.00, float(existing[3]), 0.01,
                                key="ry2")

            if roi_x2 <= roi_x1:
                st.warning("⚠ X2 必须大于 X1")
            if roi_y2 <= roi_y1:
                st.warning("⚠ Y2 必须大于 Y1")

            st.markdown("---")
            ref_upload = st.file_uploader(
                "上传一张参考图预览 ROI（可选，不上传则仅保存坐标）",
                type=["png","jpg","jpeg","bmp"],
                key="calib_ref_upload",
            )

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("💾 保存标定", use_container_width=True):
                    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                        roi_wh = (1920, 1080)
                        if ref_upload:
                            ref_img = bytes_to_bgr(ref_upload.read())
                            if ref_img is not None:
                                roi_wh = (ref_img.shape[1], ref_img.shape[0])
                        save_calibration(
                            calib_view,
                            [roi_x1, roi_y1, roi_x2, roi_y2],
                            roi_wh,
                        )
                        st.success(
                            f"已保存 **{calib_view_label}** ROI → "
                            f"[{roi_x1:.2f}, {roi_y1:.2f}, "
                            f"{roi_x2:.2f}, {roi_y2:.2f}]"
                        )
                        st.rerun()
                    else:
                        st.error("ROI 参数无效，请检查 X1<X2 且 Y1<Y2。")

            with col_clear:
                if st.button("🗑 清除标定", use_container_width=True):
                    p = _CALIB_DIR / f"{calib_view}.json"
                    if p.exists():
                        p.unlink()
                        st.success(f"已清除 **{calib_view_label}** 的标定文件")
                        st.rerun()
                    else:
                        st.info("该视角暂无标定文件。")

        with c_prev:
            st.markdown("#### 预览")
            cur_roi = load_calibration(calib_view)
            status_text = (
                f"已标定：[{cur_roi[0]:.2f}, {cur_roi[1]:.2f}, "
                f"{cur_roi[2]:.2f}, {cur_roi[3]:.2f}]"
                if cur_roi else "未标定（自动定位）"
            )
            badge_color = "#d4edda" if cur_roi else "#fff3cd"
            st.markdown(
                f'<span style="background:{badge_color};padding:4px 12px;'
                f'border-radius:6px;font-size:0.85rem;">{status_text}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            def _load_bgr_pil(path_or_bytes) -> np.ndarray | None:
                """用 PIL 加载图像，兼容中文路径，返回 BGR ndarray。"""
                try:
                    from PIL import Image as _PILImage
                    import io as _io
                    if isinstance(path_or_bytes, (str, Path)):
                        pil = _PILImage.open(str(path_or_bytes)).convert("RGB")
                    else:
                        pil = _PILImage.open(_io.BytesIO(path_or_bytes)).convert("RGB")
                    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    return None

            roi_now = [roi_x1, roi_y1, roi_x2, roi_y2]

            if ref_upload is not None:
                ref_upload.seek(0)
                ref_img_data = _load_bgr_pil(ref_upload.read())
                if ref_img_data is not None:
                    preview = draw_roi_preview(ref_img_data, roi_now)
                    st.image(bgr_to_rgb(preview),
                             caption="ROI 预览（绿框 = 标定区域）", width="stretch")
                else:
                    st.error("参考图解码失败，请重新上传。")
            else:
                # 自动用标准图库第一张图作预览底图
                std_dir = _STD_ROOT / calib_view
                first_std = None
                if std_dir.is_dir():
                    first_std = next(
                        (f for f in sorted(std_dir.iterdir())
                         if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}),
                        None,
                    )

                if first_std:
                    std_img = _load_bgr_pil(first_std)
                    if std_img is not None:
                        preview = draw_roi_preview(std_img, roi_now)
                        st.image(
                            bgr_to_rgb(preview),
                            caption=f"底图：{first_std.name}（绿框 = 当前 ROI）",
                            width="stretch",
                        )
                    else:
                        st.warning(
                            f"标准图加载失败（{first_std.name}），"
                            "请在左侧上传参考图手动预览。"
                        )
                else:
                    st.info(
                        f"视角 **{calib_view_label}** 的标准图库为空，"
                        "请上传参考图预览 ROI。"
                    )
