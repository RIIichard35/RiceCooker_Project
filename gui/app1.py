import streamlit as st
import cv2
import numpy as np
import os
import sys

# === 1. 路径设置 (确保能导入 src 模块) ===
# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.stage1_vision import Stage1Detector
from src.config import PROJECT_ROOT, MIN_MATCH_COUNT

# === 2. 页面基础配置 ===
st.set_page_config(
    page_title="电饭煲初筛检测台",
    page_icon="🔍",
    layout="wide"
)

# 自定义 CSS 样式，让结果显示更漂亮
st.markdown("""
    <style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .pass { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .anomaly { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .fail { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# === 3. 侧边栏：配置区 ===
st.sidebar.header("🛠️ 检测配置")

# 自动扫描 assets/images_ref 下有哪些子文件夹 (front, top, back...)
ref_root = os.path.join(PROJECT_ROOT, "assets", "images_ref")
available_views = []
if os.path.exists(ref_root):
    available_views = [d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))]
else:
    st.sidebar.error(f"找不到标准样目录: {ref_root}")

# 下拉菜单选择视角
selected_view = st.sidebar.selectbox(
    "请选择当前检测的面 (View):",
    options=available_views,
    index=0 if available_views else None
)

# 阈值微调 (方便你在网页上实时测试效果)
current_threshold = st.sidebar.slider("匹配合格阈值 (MIN_MATCH_COUNT)", 5, 50, MIN_MATCH_COUNT)

# === 4. 核心逻辑：加载检测器 ===
# 使用 @st.cache_resource 缓存检测器，避免每次点击都重新加载金样，提高速度
@st.cache_resource
def load_detector(view_name):
    if not view_name: return None
    folder_path = os.path.join(ref_root, view_name)
    print(f"[GUI] 正在加载标准库: {folder_path}")
    return Stage1Detector(folder_path)

if selected_view:
    detector = load_detector(selected_view)
    st.sidebar.success(f"✅ 已加载 {selected_view} 金样库")
else:
    detector = None
    st.warning("⚠️ 请先在 assets/images_ref 下创建 front/top 等子文件夹并放入金样")

# === 5. 主界面：上传与检测 ===
st.title("🏭 电饭煲外观缺陷初筛系统 (Stage 1)")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 上传待测图片")
    uploaded_file = st.file_uploader("支持 PNG, JPG, BMP", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        # --- 关键步骤：内存读取图片 (完美避开中文路径问题) ---
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # 显示原图
        st.image(uploaded_file, caption="待测原图", use_column_width=True)

with col2:
    st.subheader("2. 检测结果分析")
    
    if uploaded_file is not None and detector is not None:
        with st.spinner("正在进行多模板特征匹配..."):
            # 这里的 check_integrity 需要稍微改一下 src 里的逻辑才能返回 result_img
            # 但为了简单，我们直接复用现有的，或者把逻辑写在这里
            
            # --- 调用检测逻辑 ---
            # 为了在GUI显示更多细节，我们这里手动调用 detector 内部的方法
            # 这样我们可以拿到 best_match_count 来做三级分类
            
            kp_target, des_target = detector.orb.detectAndCompute(target_img, None)
            
            best_match_count = 0
            best_ref_data = None
            best_matches_list = []
            
            if des_target is not None and len(kp_target) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                for ref_data in detector.reference_features:
                    matches = bf.match(ref_data['des'], des_target)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = matches[:int(len(matches) * 0.15)]
                    
                    if len(good_matches) > best_match_count:
                        best_match_count = len(good_matches)
                        best_ref_data = ref_data
                        best_matches_list = good_matches

            # --- 结果三级判定逻辑 ---
            # 1. 通过：匹配点 > 阈值
            # 2. 异常：匹配点 < 阈值 但 > 5 (说明有点像，但不对劲)
            # 3. 破损/错误：匹配点 <= 5 (完全不像，或者严重缺失)
            
            status = ""
            css_class = ""
            icon = ""
            
            if best_match_count >= current_threshold:
                status = "通过初筛 (Passed)"
                css_class = "pass"
                icon = "✅"
                detail_msg = "产品结构完整，特征匹配良好。"
            elif 5 < best_match_count < current_threshold:
                status = "存在异常 (Anomaly Detected)"
                css_class = "anomaly"
                icon = "⚠️"
                detail_msg = "特征匹配不足，疑似位置偏移或轻微形变。"
            else:
                status = "出现破损 / 严重缺失 (Damaged/Missing)"
                css_class = "fail"
                icon = "❌"
                detail_msg = "几乎无法识别目标特征，判定为严重缺陷或未放置产品。"

            # --- 显示结果卡片 ---
            st.markdown(f"""
                <div class="result-card {css_class}">
                    <h2>{icon} {status}</h2>
                    <p><b>匹配得分: {best_match_count}</b> (合格线: {current_threshold})</p>
                    <p>{detail_msg}</p>
                </div>
            """, unsafe_allow_html=True)

            # --- 显示可视化调试图 ---
            if best_ref_data:
                st.write("🔍 **特征匹配可视化 (调试用):**")
                debug_img = cv2.drawMatches(
                    best_ref_data['img'], best_ref_data['kp'],
                    target_img, kp_target,
                    best_matches_list, None, flags=2
                )
                st.image(debug_img, caption=f"最佳匹配参考图: {best_ref_data['name']}", use_column_width=True)
            else:
                st.error("无法提取图像特征，请检查图片清晰度。")

    elif detector is None:
        st.info("👈 请先在左侧选择检测视角")
    else:
        st.info("👋 请上传图片开始检测")