import os

# ==========================================
# 1. 路径配置
# ==========================================
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 资源路径
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
REF_IMG_DIR = os.path.join(ASSETS_DIR, "images_ref")  # 金样文件夹
TEST_IMG_DIR = os.path.join(ASSETS_DIR, "images_test") # 测试图文件夹

# ==========================================
# 2. 一级检测 (Stage 1: CV) 阈值
# ==========================================
# [关键参数] ORB特征匹配阈值
# 缺少这个就会报错！
MIN_MATCH_COUNT = 15  

# 结构相似度 (SSIM) 阈值
SSIM_THRESHOLD = 0.80

# ==========================================
# 3. 三级规则 (Stage 3: Logic) 阈值
# ==========================================
PIXELS_PER_MM = 4.8      
MIN_DEFECT_SIZE_MM = 1.0