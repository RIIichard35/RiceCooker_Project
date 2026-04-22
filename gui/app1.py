"""
电饭煲外观初筛系统 — Stage 1 前端（Streamlit）
兼容 Windows 开发环境 & 树莓派 5 生产环境

运行方式：
    streamlit run gui/app1.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── 路径设置（相对路径，兼容树莓派） ────────────────────────────────────
GUI_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CALIB_DIR,
    FILM_COVERAGE_THRESHOLD,
    MAX_MISSING_COUNT,
    MIN_MATCH_COUNT,
    STANDARDS_DIR,
)
from src.stage1_vision import Stage1Detector

# ── 页面基础配置 ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="电饭煲初筛检测台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card {
    padding: 18px 22px; border-radius: 10px;
    text-align: center; margin-bottom: 14px;
    font-family: "Microsoft YaHei", sans-serif;
}
.card-pass   { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
.card-fail   { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
.card-retake { background:#fff3cd; color:#856404; border:1px solid #ffeeba; }

.metric-row  { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }
.metric-box  {
    flex:1; min-width:110px; background:#f0f2f6;
    border-radius:8px; padding:10px 8px; text-align:center;
}
.metric-label { font-size:0.72rem; color:#555; margin-bottom:4px; }
.metric-value { font-size:1.20rem; font-weight:700; color:#222; }

.issue-item { color:#721c24; margin:3px 0; }
.warn-item  { color:#856404; margin:3px 0; }

.calib-badge-on  { background:#d4edda; color:#155724; padding:4px 10px;
                   border-radius:6px; font-size:0.82rem; }
.calib-badge-off { background:#f8d7da; color:#721c24; padding:4px 10px;
                   border-radius:6px; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)


# ── 工具函数 ────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def status_card(status: str, issues: list[str], warnings: list[str]) -> str:
    css   = {"PASS": "card-pass", "FAIL": "card-fail", "RETAKE": "card-retake"}
    icon  = {"PASS": "✅", "FAIL": "❌", "RETAKE": "⚠️"}
    label = {"PASS": "通过初筛", "FAIL": "初筛不合格", "RETAKE": "需要重拍"}
    c = css.get(status, "card-fail")
    i = icon.get(status, "❓")
    l = label.get(status, status)
    issues_html = "".join(
        f'<p class="issue-item">⛔ {iss}</p>' for iss in issues
    ) if issues else '<p style="color:#155724">无阻断性问题</p>'
    warns_html = "".join(
        f'<p class="warn-item">⚠ {w}</p>' for w in warnings
    ) if warnings else ""
    return f"""
<div class="card {c}">
  <h2 style="margin:0 0 8px">{i} {l}</h2>
  {issues_html}
  {warns_html}
</div>"""


def metric_row_html(metrics: list[tuple[str, str]]) -> str:
    boxes = "".join(
        f'<div class="metric-box">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
        for label, value in metrics
    )
    return f'<div class="metric-row">{boxes}</div>'


# ── 标定读写函数 ────────────────────────────────────────────────────────

_CALIB_DIR = Path(CALIB_DIR)


def load_calibration(view: str) -> list | None:
    """读取视角标定 ROI（相对坐标 [x1r, y1r, x2r, y2r]），无则返回 None。"""
    p = _CALIB_DIR / f"{view}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("roi_rel")
    except Exception:
        return None


def save_calibration(view: str, roi_rel: list, ref_wh: tuple) -> None:
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


def calib_to_abs(roi_rel: list, w: int, h: int) -> list:
    return [
        int(roi_rel[0] * w), int(roi_rel[1] * h),
        int(roi_rel[2] * w), int(roi_rel[3] * h),
    ]


def draw_roi_preview(img_bgr: np.ndarray, roi_rel: list) -> np.ndarray:
    """在图像上绘制半透明 ROI 框用于预览。"""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = calib_to_abs(roi_rel, w, h)
    vis     = img_bgr.copy()
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(
        vis, f"ROI  {x2-x1}x{y2-y1}px",
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 200, 0), 2,
    )
    return vis


# ── 视角名称映射 ────────────────────────────────────────────────────────

VIEW_LABELS: dict[str, str] = {
    "front": "正面",
    "back":  "后面",
    "left":  "左面",
    "right": "右面",
    "top":   "顶部",
}

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


# ── 侧边栏 ──────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 检测配置")

if not available_views:
    st.sidebar.error(f"找不到标准图库目录：\n{std_root}")

selected_label: str | None = st.sidebar.selectbox(
    "检测视角（按面选择）",
    options=view_display,
    index=0 if view_display else None,
    help="请选择当前拍摄的产品面",
)
selected_view = view_map.get(selected_label, selected_label) if selected_label else None

# 标定状态徽章
if selected_view:
    roi_rel = load_calibration(selected_view)
    if roi_rel:
        st.sidebar.markdown(
            f'<span class="calib-badge-on">📐 ROI 已标定</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="calib-badge-off">📐 ROI 未标定（使用自动定位）</span>',
            unsafe_allow_html=True,
        )
else:
    roi_rel = None

st.sidebar.markdown("---")
st.sidebar.subheader("阈值调节")

film_thresh = st.sidebar.slider(
    "塑料膜覆盖阈值 (%)", 10, 90,
    int(FILM_COVERAGE_THRESHOLD * 100), 5,
    help="产品被塑料膜遮盖超过此比例时判为 RETAKE",
)
match_thresh = st.sidebar.slider(
    "ORB 匹配点阈值", 5, 60, MIN_MATCH_COUNT,
    help="与标准图 ORB 特征匹配点数下限",
)
missing_thresh = st.sidebar.slider(
    "缺件框数量阈值", 1, 10, MAX_MISSING_COUNT,
    help="缺件框数量超过此值则判为 FAIL",
)

st.sidebar.markdown("---")
show_cutout = st.sidebar.checkbox("显示白底抠图", value=True)
show_detail = st.sidebar.checkbox("显示调试详情", value=False)


# ── 加载检测器 ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="正在加载标准图库...")
def load_detector(view_name: str) -> Stage1Detector:
    return Stage1Detector(str(std_root / view_name))


if selected_view:
    detector = load_detector(selected_view)
    ref_count = len(detector.reference_data)
    st.sidebar.success(f"已加载 **{selected_label}** 标准图库（{ref_count} 张）")
else:
    detector = None


# ── 主界面 ──────────────────────────────────────────────────────────────
st.title("🏭 电饭煲外观初筛系统（Stage 1）")
st.markdown("---")

tab_detect, tab_calib, tab_monitor = st.tabs(["🔍  检测", "📐  ROI 标定", "📷  实时监控 Demo"])


# ════════════════════════════════════════════════════════════════════════
# Tab 1 — 检测
# ════════════════════════════════════════════════════════════════════════
with tab_detect:
    col_upload, col_result = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.subheader("① 上传待测图片")
        uploaded = st.file_uploader(
            "拖拽或点击上传（PNG / JPG / BMP）",
            type=["png", "jpg", "jpeg", "bmp"],
            label_visibility="collapsed",
            key="detect_upload",
        )

        img_bgr: np.ndarray | None = None
        if uploaded:
            raw_bytes = uploaded.read()
            img_bgr   = bytes_to_bgr(raw_bytes)
            if img_bgr is not None:
                h, w = img_bgr.shape[:2]
                st.image(
                    bgr_to_rgb(img_bgr),
                    caption=f"待测原图  {w}×{h}px",
                    use_container_width=True,
                )
            else:
                st.error("图片解码失败，请重新上传。")

    with col_result:
        st.subheader("② 检测结果")

        if img_bgr is None:
            st.info("👈 请先上传待测图片")
        elif detector is None:
            st.warning("请在左侧侧边栏选择检测视角后重试。")
        else:
            # 计算标定 bbox（绝对坐标）
            h_img, w_img = img_bgr.shape[:2]
            calib_bbox = calib_to_abs(roi_rel, w_img, h_img) if roi_rel else None

            with st.spinner("正在检测中，请稍候..."):
                result = detector.inspect_with_localization(
                    img_bgr,
                    min_match_count=match_thresh,
                    missing_regions_fail_count=missing_thresh,
                    calibrated_bbox=calib_bbox,
                )

            status   = result["status"]
            issues   = result["issues"]
            warnings = result["warnings"]

            st.markdown(status_card(status, issues, warnings), unsafe_allow_html=True)

            film_pct  = result["film_coverage"] * 100
            sim_pct   = result["similarity"] * 100
            loc_score = result["localization_score"]
            miss_cnt  = len(result["missing_regions"])
            extra_cnt = len(result["extra_regions"])

            calib_flag = "已标定" if roi_rel else "自动"
            st.markdown(
                metric_row_html([
                    ("定位方式", calib_flag),
                    ("覆膜率",   f"{film_pct:.1f}%"),
                    ("相似度",   f"{sim_pct:.0f}%"),
                    ("定位置信度", f"{loc_score:.3f}"),
                    ("异常框",   f"{miss_cnt}"),
                ]),
                unsafe_allow_html=True,
            )

            ann = result.get("annotated_image")
            if ann is not None:
                st.image(
                    bgr_to_rgb(ann),
                    caption="标注图（黄框=产品区域 | 红框=异常区域）",
                    use_container_width=True,
                )
            else:
                st.warning("未能生成标注图。")

            if show_cutout:
                cut = result.get("cutout_image")
                if cut is not None:
                    st.markdown("**白底抠图（产品主体）**")
                    st.image(cut, caption="GrabCut 抠图结果（灰度白底）",
                             use_container_width=True)

            if show_detail:
                st.markdown("---")
                st.markdown("**🔧 调试详情**")
                q = result.get("quality", {})
                dc = st.columns(5)
                dc[0].metric("清晰度",   f"{q.get('sharpness', 0):.1f}")
                dc[1].metric("纹理比",   f"{q.get('texture_ratio', 0):.3f}")
                dc[2].metric("反光比",   f"{q.get('reflection_ratio', 0):.3f}")
                dc[3].metric("过曝率",   f"{q.get('overexpose_ratio', 0)*100:.1f}%")
                dc[4].metric("欠曝率",   f"{q.get('underexpose_ratio', 0)*100:.1f}%")
                st.markdown(
                    f"- **最佳匹配标准图**: `{result.get('best_ref_name', '—')}`  \n"
                    f"- **产品定位框**: `{result.get('bbox')}`  \n"
                    f"- **ORB 匹配点**: `{result.get('score', 0)}`"
                )
                if result["missing_regions"]:
                    st.markdown("**异常区域坐标：**")
                    for i, box in enumerate(result["missing_regions"], 1):
                        st.code(f"[{i}] x1={box[0]} y1={box[1]} x2={box[2]} y2={box[3]}")


# ════════════════════════════════════════════════════════════════════════
# Tab 2 — ROI 标定
# ════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.subheader("📐 产品区域 ROI 标定")
    st.info(
        "固定相机安装后，只需标定一次。上传一张清晰的产品图，"
        "用滑块框出产品所在区域，保存后检测时将自动跳过不稳定的模板匹配，"
        "直接使用标定区域，大幅提升定位准确率。"
    )

    if not selected_view:
        st.warning("请先在左侧侧边栏选择检测视角。")
    else:
        st.markdown(f"**当前视角：{selected_label}**")

        # 显示已有标定信息
        existing = load_calibration(selected_view)
        if existing:
            p = _CALIB_DIR / f"{selected_view}.json"
            info = json.loads(p.read_text(encoding="utf-8"))
            st.success(
                f"已有标定：ROI = {[f'{v:.3f}' for v in existing]}  |  "
                f"标定时间：{info.get('created_at', '未知')}"
            )
            if st.button("🗑️ 清除标定（恢复自动定位）", key="clear_calib"):
                (p).unlink(missing_ok=True)
                st.rerun()

        st.markdown("---")
        st.markdown("#### 操作步骤")
        st.markdown(
            "1. 上传一张相机实际拍摄的产品图  \n"
            "2. 调整下方四个滑块，让绿色方框恰好框住产品主体（略有余量即可）  \n"
            "3. 点击「💾 保存标定」完成"
        )

        calib_upload = st.file_uploader(
            "上传标定参考图（PNG / JPG / BMP）",
            type=["png", "jpg", "jpeg", "bmp"],
            key="calib_upload",
        )

        if calib_upload:
            calib_bgr = bytes_to_bgr(calib_upload.read())
            if calib_bgr is None:
                st.error("图片解码失败，请重新上传。")
            else:
                ch, cw = calib_bgr.shape[:2]

                # 默认值：先看是否有已有标定
                default_rel = existing if existing else [0.05, 0.05, 0.95, 0.95]

                c_sl, c_prev = st.columns([1, 1.8], gap="large")

                with c_sl:
                    st.markdown("**拖动滑块调整 ROI 边界（百分比）**")
                    x1_pct = st.slider("左边界 ←",  0, 99,
                                       int(default_rel[0] * 100), key="cx1")
                    y1_pct = st.slider("上边界 ↑",  0, 99,
                                       int(default_rel[1] * 100), key="cy1")
                    x2_pct = st.slider("右边界 →",  1, 100,
                                       int(default_rel[2] * 100), key="cx2")
                    y2_pct = st.slider("下边界 ↓",  1, 100,
                                       int(default_rel[3] * 100), key="cy2")

                    # 校验：左 < 右，上 < 下
                    if x1_pct >= x2_pct:
                        st.warning("左边界必须小于右边界")
                    elif y1_pct >= y2_pct:
                        st.warning("上边界必须小于下边界")
                    else:
                        new_roi = [
                            x1_pct / 100, y1_pct / 100,
                            x2_pct / 100, y2_pct / 100,
                        ]
                        abs_box = calib_to_abs(new_roi, cw, ch)
                        roi_w = abs_box[2] - abs_box[0]
                        roi_h = abs_box[3] - abs_box[1]
                        st.markdown(
                            f"**ROI 尺寸**：{roi_w} × {roi_h} px  \n"
                            f"**ROI 坐标**：({abs_box[0]}, {abs_box[1]}) → "
                            f"({abs_box[2]}, {abs_box[3]})"
                        )

                        if st.button("💾 保存标定", type="primary", key="save_calib"):
                            save_calibration(selected_view, new_roi, (cw, ch))
                            st.success(
                                f"标定已保存！下次检测 **{selected_label}** 视角时将自动使用此 ROI。"
                            )
                            st.rerun()

                with c_prev:
                    st.markdown("**预览（绿框 = 当前 ROI）**")
                    if x1_pct < x2_pct and y1_pct < y2_pct:
                        preview_roi = [
                            x1_pct / 100, y1_pct / 100,
                            x2_pct / 100, y2_pct / 100,
                        ]
                        preview_img = draw_roi_preview(calib_bgr, preview_roi)
                        st.image(
                            bgr_to_rgb(preview_img),
                            caption=f"标定参考图  {cw}×{ch}px",
                            use_container_width=True,
                        )
                    else:
                        st.image(
                            bgr_to_rgb(calib_bgr),
                            caption=f"标定参考图  {cw}×{ch}px",
                            use_container_width=True,
                        )
        else:
            st.markdown(
                "_上传图片后，此处将显示带 ROI 标注框的预览图。_"
            )


# ════════════════════════════════════════════════════════════════════════
# Tab 3 — 实时监控 Demo（PC 模拟流水线触发）
# ════════════════════════════════════════════════════════════════════════
with tab_monitor:
    from src.trigger import ProductTrigger

    st.subheader("📷 实时监控 Demo — 流水线触发模拟")
    st.info(
        "**PC Demo 模式**：上传一批模拟帧图片（模拟摄像头连续抓帧），"
        "系统自动分析每帧的产品占比，找出产品完整进入镜头的触发时机，"
        "并对触发帧执行 Stage 1 检测。  \n"
        "**迁移到树莓派**：只需将 MockCamera 换成 PiCamera，其余代码零修改。"
    )

    if not selected_view:
        st.warning("请先在左侧侧边栏选择检测视角。")
    elif roi_rel is None:
        st.warning("请先在「📐 ROI 标定」页面完成标定，监控模式依赖标定 ROI。")
    elif detector is None:
        st.warning("检测器未加载，请检查标准图库路径。")
    else:
        # ── 参数配置 ───────────────────────────────────────────────────
        st.markdown("#### ① 触发 & 连拍参数配置")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4, cfg_col5 = st.columns(5)
        with cfg_col1:
            fill_threshold = st.slider(
                "产品占比触发阈值",
                min_value=0.50, max_value=1.00, value=0.95, step=0.01,
                help="ROI 内深色像素占比超过此值，且连续稳定，才触发拍照",
            )
        with cfg_col2:
            stable_frames = st.slider(
                "稳定帧数",
                min_value=2, max_value=8, value=4,
                help="连续多少帧满足条件才触发（越大越稳定，越慢响应）",
            )
        with cfg_col3:
            dark_thresh = st.slider(
                "深色像素阈值（灰度）",
                min_value=60, max_value=180, value=110,
                help="灰度值低于此数值的像素视为产品前景（电饭煲为深色）",
            )
        with cfg_col4:
            burst_count = st.slider(
                "连拍张数",
                min_value=1, max_value=5, value=3,
                help="触发后连续拍摄的张数，从中选最清晰的一张送检测",
            )
        with cfg_col5:
            burst_interval = st.slider(
                "连拍间隔 (ms)",
                min_value=1, max_value=50, value=5,
                help="每两次连拍之间的时间间隔（毫秒）",
            )

        st.markdown("---")

        # ── 上传模拟帧 ─────────────────────────────────────────────────
        st.markdown("#### ② 上传模拟帧序列（按顺序上传，模拟摄像头连续抓帧）")
        frame_files = st.file_uploader(
            "支持 PNG / JPG / BMP，可多选",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="monitor_frames",
        )

        if not frame_files:
            st.markdown(
                "_提示：可把同一视角的多张图片一次性拖入，系统会按文件名顺序分析。_"
            )
        else:
            frame_files = sorted(frame_files, key=lambda f: f.name)
            n_frames    = len(frame_files)
            st.caption(f"已上传 {n_frames} 帧")

            # ── 逐帧分析占比 ───────────────────────────────────────────
            trigger = ProductTrigger(
                camera            = None,   # type: ignore  # Demo 模式不用相机对象
                roi_rel           = roi_rel,
                fill_threshold    = fill_threshold,
                stable_frames     = stable_frames,
                dark_thresh       = dark_thresh,
                burst_count       = burst_count,
                burst_interval_ms = float(burst_interval),
            )

            fill_ratios:     list[float]          = []
            triggered_idxs:  list[int]            = []   # 触发帧在帧序列中的索引
            burst_groups:    list[list[int]]       = []   # 每次触发对应的连拍帧索引组

            frames_bgr: list[np.ndarray] = []
            for ff in frame_files:
                bgr = bytes_to_bgr(ff.read())
                frames_bgr.append(bgr)
                fill = trigger.compute_fill_ratio(bgr) if bgr is not None else 0.0
                fill_ratios.append(fill)
                if trigger.check_trigger(fill):
                    tidx = len(fill_ratios) - 1
                    triggered_idxs.append(tidx)
                    # Demo 模式：连拍取触发帧及其后续帧（模拟时序）
                    group = [
                        min(tidx + i, n_frames - 1)
                        for i in range(burst_count)
                    ]
                    burst_groups.append(group)
                    trigger.reset()

            # ── 帧序列概览 ─────────────────────────────────────────────
            st.markdown("#### ③ 帧序列概览")
            import math
            cols_per_row = 5
            n_rows = math.ceil(n_frames / cols_per_row)

            triggered_set = set(triggered_idxs)
            for row in range(n_rows):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    frame_idx = row * cols_per_row + col_idx
                    if frame_idx >= n_frames:
                        break
                    fill   = fill_ratios[frame_idx]
                    is_trg = frame_idx in triggered_set
                    bgr_f  = frames_bgr[frame_idx]
                    fname  = frame_files[frame_idx].name

                    with col:
                        if bgr_f is not None:
                            thumb = draw_roi_preview(bgr_f, roi_rel) if is_trg else bgr_f
                            st.image(bgr_to_rgb(thumb), use_container_width=True)
                        border = "3px solid #28a745" if is_trg else "1px solid #dee2e6"
                        bg     = "#d4edda"            if is_trg else "#f8f9fa"
                        label  = "📸 触发"             if is_trg else f"{fill*100:.0f}%"
                        st.markdown(
                            f'<div style="border:{border};background:{bg};'
                            f'border-radius:6px;padding:4px 6px;text-align:center;'
                            f'font-size:0.78rem;">'
                            f'<b>{label}</b><br>'
                            f'<span style="color:#555">{fname}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            # ── 连拍结果 + Stage 1 检测 ────────────────────────────────
            if triggered_idxs:
                st.markdown(
                    f"#### ④ 连拍结果 & Stage 1 检测（共触发 {len(triggered_idxs)} 次）"
                )
                for rank, (tidx, group) in enumerate(
                    zip(triggered_idxs, burst_groups), 1
                ):
                    fname = frame_files[tidx].name
                    with st.expander(
                        f"触发事件 [{rank}]  —  {fname}  "
                        f"（占比 {fill_ratios[tidx]*100:.1f}%）",
                        expanded=(rank == 1),
                    ):
                        # 获取连拍帧列表（Demo: 取对应索引的帧）
                        burst_imgs = [
                            frames_bgr[i] for i in group
                            if frames_bgr[i] is not None
                        ]

                        # 计算每张清晰度分数
                        scores = ProductTrigger.sharpness_scores(burst_imgs)
                        best_result = ProductTrigger.select_sharpest(burst_imgs)

                        # ── 连拍三张对比展示 ───────────────────────────
                        st.markdown(
                            f"**连拍 {len(burst_imgs)} 张**"
                            f"（间隔 {burst_interval} ms 一张，"
                            f"自动选最清晰帧送检测）"
                        )
                        burst_cols = st.columns(len(burst_imgs))
                        for bi, (bc, bimg, bscore) in enumerate(
                            zip(burst_cols, burst_imgs, scores)
                        ):
                            is_best = (best_result is not None and bi == best_result[0])
                            with bc:
                                st.image(
                                    bgr_to_rgb(bimg),
                                    caption=(
                                        f"第 {bi+1} 张  +{bi*burst_interval}ms\n"
                                        f"清晰度 {bscore:.0f}"
                                    ),
                                    use_container_width=True,
                                )
                                if is_best:
                                    st.markdown(
                                        '<div style="text-align:center;'
                                        'color:#155724;font-weight:700;'
                                        'font-size:0.82rem;">✅ 最清晰，送检</div>',
                                        unsafe_allow_html=True,
                                    )

                        # ── Stage 1 检测（最清晰帧） ───────────────────
                        st.markdown("---")
                        if best_result is None:
                            st.error("连拍帧全部解码失败。")
                        else:
                            best_idx, best_img = best_result
                            h_b, w_b = best_img.shape[:2]
                            calib_bbox_b = calib_to_abs(roi_rel, w_b, h_b)

                            with st.spinner("正在对最清晰帧执行 Stage 1 检测..."):
                                res = detector.inspect_with_localization(
                                    best_img,
                                    min_match_count=match_thresh,
                                    missing_regions_fail_count=missing_thresh,
                                    calibrated_bbox=calib_bbox_b,
                                )

                            rc1, rc2 = st.columns([1, 1.5])
                            with rc1:
                                st.image(
                                    bgr_to_rgb(best_img),
                                    caption=f"送检帧（第 {best_idx+1} 张）",
                                    use_container_width=True,
                                )
                            with rc2:
                                st.markdown(
                                    status_card(
                                        res["status"], res["issues"], res["warnings"]
                                    ),
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    metric_row_html([
                                        ("覆膜率", f"{res['film_coverage']*100:.1f}%"),
                                        ("相似度", f"{res['similarity']*100:.0f}%"),
                                        ("异常框", f"{len(res['missing_regions'])}"),
                                    ]),
                                    unsafe_allow_html=True,
                                )
                                ann = res.get("annotated_image")
                                if ann is not None:
                                    st.image(
                                        bgr_to_rgb(ann),
                                        caption="标注图（黄框=产品 | 红框=异常）",
                                        use_container_width=True,
                                    )
            else:
                st.warning(
                    f"未找到触发帧（阈值={fill_threshold:.0%}）。"
                    "建议降低占比阈值，或检查深色像素阈值设置。"
                )

        # ── 树莓派迁移指南 ─────────────────────────────────────────────
        st.markdown("---")
        with st.expander("🔧 树莓派迁移指南（点击展开）"):
            st.markdown("""
**只需修改两处，其他代码零改动：**

**第 1 处** — 相机实例化（`src/trigger.py` 已预留）
```python
# PC Demo（当前）
from src.trigger import MockCamera
cam = MockCamera(image_folder="path/to/frames", fps=5)

# 树莓派（迁移后，取消 PiCamera.start() 里的注释即可）
from src.trigger import PiCamera
cam = PiCamera(resolution=(1280, 960), exposure_us=2000)
```

**第 2 处** — 主循环（替换 Streamlit 上传为 GPIO/相机循环）
```python
cam.start()
trigger = ProductTrigger(cam, roi_rel=roi_rel)
while True:
    frame = cam.capture_frame()
    fill  = trigger.compute_fill_ratio(frame)
    if trigger.check_trigger(fill):
        hires  = cam.capture_hires()
        result = detector.inspect_with_localization(
            hires, calibrated_bbox=calib_to_abs(roi_rel, *hires.shape[1::-1])
        )
        # → 上报结果 / 控制流水线
        trigger.reset()
```

**曝光建议**
- 曝光时间：先从 2000μs 开始，根据实际亮度调整
- 关闭自动曝光（`AeEnable=False`）避免过曝/欠曝
- 可配合 LED 环形补光灯稳定光照
            """)

