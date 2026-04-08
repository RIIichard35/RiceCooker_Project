import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image, ImageOps 

# ================= ⚙️ 系统配置区 =================

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)  # gui 的父目录就是项目根目录

# ⚠️ 请确保这里指向的是你最新训练的模型路径！
# 使用绝对路径，确保从任何目录运行都能正确找到模型
MODEL_PATH = os.path.join(PROJECT_ROOT, 'assets', 'models', 'best20240919.pt')
STD_IMG_DIR = os.path.join(PROJECT_ROOT, 'assets', 'images_ref') 

# 默认设置
DEFAULT_CONF = 0.10          # 默认调低，为了抓细小划痕
FILM_AREA_THRESHOLD = 0.30   
STAGE1_MATCH_MIN = 12        
BLUR_THRESHOLD = 35.0        
MIN_RESOLUTION = 300         

st.set_page_config(page_title="电饭煲划痕检测展示端", layout="wide")

# ================= 🧠 核心工具链 =================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 找不到模型文件: {MODEL_PATH}")
        st.warning("💡 提示：如果你刚训练了新模型，请将 `MODEL_PATH` 变量修改为新生成的 .pt 文件路径。")
        return None
    return YOLO(MODEL_PATH)

def load_standard_images():
    standards = []
    if not os.path.exists(STD_IMG_DIR):
        os.makedirs(STD_IMG_DIR, exist_ok=True)
    for f in os.listdir(STD_IMG_DIR):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(STD_IMG_DIR, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None: standards.append(img)
    return standards

def process_uploaded_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image) 
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        return None

def check_image_quality(img_bgr, threshold):
    h, w = img_bgr.shape[:2]
    if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
        return False, f"分辨率过低 ({w}x{h})", 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score < threshold:
        return False, f"图像模糊 (得分: {score:.1f})", score
    return True, "OK", score

def enhance_contrast(image_bgr):
    """CLAHE 增强"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def stage1_inspection_strict(target_img_bgr, standard_imgs):
    if not standard_imgs: return False, 0, None
    orb = cv2.ORB_create(nfeatures=1500)
    kp_tgt, des_tgt = orb.detectAndCompute(target_img_bgr, None)
    if des_tgt is None: return False, 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_inliers = 0
    best_vis = None
    for std_gray in standard_imgs:
        kp_std, des_std = orb.detectAndCompute(std_gray, None)
        if des_std is None: continue
        if len(des_tgt) < 2 or len(des_std) < 2: continue
        matches = bf.knnMatch(des_std, des_tgt, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) > 8:
            src_pts = np.float32([kp_std[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                matchesMask = mask.ravel().tolist()
                inliers = sum(matchesMask)
                if inliers > best_inliers:
                    best_inliers = inliers
                    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
                    best_vis = cv2.drawMatches(std_gray, kp_std, target_img_bgr, kp_tgt, good, None, **draw_params)
    return best_inliers >= STAGE1_MATCH_MIN, best_inliers, best_vis

def draw_box(img, box, color, label=None):
    """画框辅助函数"""
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) # 线宽改为2
    if label:
        # 在框上方绘制文字背景，确保文字清晰
        (w, h), _ = cv2.getTextSize(label, 0, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), 0, 0.6, (255, 255, 255), 1)

# ================= 🖥️ 交互界面 UI =================

st.title("🔎 电饭煲划痕检测系统")
st.markdown("---")

# --- 侧边栏 ---
st.sidebar.header("🛠️ 参数设置")

# 1. 调试模式开关
debug_mode = st.sidebar.toggle("🐛 开启调试模式 (显示所有检测)", value=False, help="开启后，将显示所有置信度极低的框，用于检查是否漏检。")

# 2. 阈值控制
conf_val = 0.01 if debug_mode else DEFAULT_CONF
conf_thres = st.sidebar.slider("AI 敏感度 (Confidence)", 0.01, 1.0, conf_val)

# 3. 分辨率
inference_size = st.sidebar.select_slider(
    "检测分辨率 (imgsz)", 
    options=[640, 960, 1280, 1600], 
    value=1280,
    help="针对微小划痕，请使用 1280 或 1600。"
)

use_enhancement = st.sidebar.checkbox("图像增强 (CLAHE)", value=True)

model = load_model()
standards = load_standard_images()

# --- 主界面 ---
uploaded_file = st.file_uploader("📤 上传图片进行检测", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img_bgr = process_uploaded_image(uploaded_file)
    if img_bgr is None:
        st.error("❌ 图片损坏")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 📷 原始图片")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    if st.button("开始检测", type="primary"):
        # 1. 初筛
        if standards: # 只有在有标准图时才跑初筛
            s1_pass, inliers, s1_vis = stage1_inspection_strict(img_bgr, standards)
            if not s1_pass:
                st.error(f"⛔ 初筛未通过：产品结构不匹配 (有效点: {inliers})")
                if s1_vis is not None: st.image(s1_vis, channels="BGR")
                st.stop()
        
        # 2. AI 推理
        img_input = enhance_contrast(img_bgr) if use_enhancement else img_bgr
        results = model.predict(img_input, conf=conf_thres, imgsz=inference_size)
        
        # 3. 数据提取
        result = results[0]
        boxes = result.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        final_img = img_bgr.copy()
        detected_defects = []

        # 4. 绘制 (无逻辑过滤，直接展示 AI 看到的一切)
        for box in boxes:
            cls_id = int(box[5])
            cls_name = model.names[cls_id]
            conf = float(box[4])
            
            # 标签格式：类别 + 置信度 (参考你的 PDF)
            label_text = f"{cls_name} {conf:.2f}"
            
            if cls_name == 'plastic_film':
                # 膜用蓝色
                draw_box(final_img, box, (255, 0, 0), label_text)
            else:
                # 缺陷用红色 (scratch, dent, bump)
                draw_box(final_img, box, (0, 0, 255), label_text)
                detected_defects.append(cls_name)

        # 5. 结果输出 (按照你的要求，没有合格/不合格，只有发现/未发现)
        with col2:
            st.write("#### 🛡️ 检测结果")
            st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
            if len(detected_defects) > 0:
                # 有划痕，输出警告并列出数量
                st.warning(f"⚠️ **检测出 {len(detected_defects)} 处异常/划痕**")
                for d in detected_defects:
                    st.write(f"- 🔴 发现 {d}")
            else:
                # 无划痕
                st.info("✅ **未检测出划痕**")
                
        # 调试信息
        if debug_mode:
            st.write("---")
            st.markdown("### 🐛 调试数据")
            st.write(f"当前使用的模型: `{MODEL_PATH}`")
            st.write(f"当前置信度阈值: `{conf_thres}`")
            st.write(f"检测到的原始框数据: {len(boxes)} 个")
            if len(boxes) == 0:
                st.error("AI 根本没看到任何东西。建议：1. 检查模型路径是否正确 2. 进一步调低阈值 3. 检查分辨率设置")