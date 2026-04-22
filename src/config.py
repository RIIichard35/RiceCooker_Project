import os

# ==========================================
# 1. 路径配置（全部使用相对路径，兼容树莓派）
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR     = os.path.join(PROJECT_ROOT, "assets")
STANDARDS_DIR  = os.path.join(ASSETS_DIR, "standards")   # 标准图库（按面分子文件夹）
CALIB_DIR      = os.path.join(ASSETS_DIR, "calibration") # ROI 标定文件（每视角一个 json）
MODELS_DIR     = os.path.join(ASSETS_DIR, "models")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "output")
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")

# ==========================================
# 2. Stage1 — ORB 特征匹配
# ==========================================
MIN_MATCH_COUNT      = 15    # ORB 最低匹配点数
SSIM_THRESHOLD       = 0.80  # 结构相似度阈值（备用）

# ==========================================
# 3. Stage1 — 产品定位
# ==========================================
MIN_LOCALIZATION_SCORE = 0.04   # 多尺度模板匹配最低置信度
MIN_LOCALIZATION_AREA  = 10000  # 产品 ROI 最小面积（像素）

# ==========================================
# 4. Stage1 — 图像质量门控
# ==========================================
MIN_SHARPNESS              = 20.0   # Laplacian 方差，低于此值判为模糊 → RETAKE
MIN_RESOLUTION             = 200    # 图像宽/高最小像素数，低于则 RETAKE
OVEREXPOSE_RATIO_THRESHOLD = 0.35   # 超过 35% 像素亮度>240 → 过曝 RETAKE
UNDEREXPOSE_RATIO_THRESHOLD= 0.70   # 超过 70% 像素亮度<25  → 欠曝 RETAKE

# ==========================================
# 5. Stage1 — 塑料膜检测（HSV 空间）
# ==========================================
FILM_COVERAGE_THRESHOLD = 0.60  # 覆盖率超过 % → RETAKE（无法细筛）
FILM_V_MIN              = 200   # 高亮度阈值（Value 通道）
FILM_S_MAX              = 50    # 低饱和阈值（Saturation 通道）

# ==========================================
# 6. Stage1 — 差异图与缺件框选
# ==========================================
DIFF_THRESHOLD       = 0.28   # 归一化像素差异阈值（越小越灵敏，建议 0.22~0.35）
MIN_DEFECT_BOX_AREA  = 2500   # 缺件框最小面积（像素），过滤噪点
MAX_MISSING_COUNT    = 3      # 缺件框超过此数量则判 FAIL

# ==========================================
# 7. 增强版视觉检测参数（兼容旧模块）
# ==========================================
MIN_FEATURE_SCORE      = 0.30
MIN_SHAPE_SCORE        = 0.70
MIN_MISSING_AREA_RATIO = 0.01

# ==========================================
# 8. Stage3 规则引擎
# ==========================================
PIXELS_PER_MM    = 4.8
MIN_DEFECT_SIZE_MM = 1.0

# ==========================================
# 9. Hailo AI 加速器（AI HAT+，Hailo-8L）
# ==========================================
HEF_PATH    = os.path.join(MODELS_DIR, "best20240919.hef")  # 转换后放这里即自动启用
HAILO_BATCH = 1                                              # 推理 batch size

# ==========================================
# 10. 调试
# ==========================================
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'