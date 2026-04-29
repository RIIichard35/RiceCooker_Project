import os

# ==========================================
# 1. 路径配置（全部使用相对路径，兼容树莓派）
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR     = os.path.join(PROJECT_ROOT, "assets")
STANDARDS_DIR  = os.path.join(ASSETS_DIR, "standards")   # 标准图库（按面分子文件夹）
ST_DIR         = os.path.join(PROJECT_ROOT, "st")        # 预处理透明抠图库（按面分子文件夹）
CALIB_DIR      = os.path.join(ASSETS_DIR, "calibration") # ROI 标定文件（每视角一个 json）
MODELS_DIR     = os.path.join(ASSETS_DIR, "models")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "output")
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")
DB_DIR         = os.path.join(PROJECT_ROOT, "data")
INSPECTION_DB_PATH = os.path.join(DB_DIR, "inspection_records.db")

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
DIFF_THRESHOLD       = 0.5   # 归一化像素差异阈值（越小越灵敏，建议 0.22~0.35）
MIN_DEFECT_BOX_AREA  = 3500   # 缺件框最小面积（像素），过滤噪点
MAX_MISSING_COUNT    = 2      # 缺件框超过此数量则判 FAIL

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

# ==========================================
# 11. 白色标签保留策略（HSV 空间 + 自适应）
# ==========================================
LABEL_RETAIN_V_MIN     = 170   # HSV V 通道最低值（偏灰白也能被收进来）
LABEL_RETAIN_S_MAX     = 90    # HSV S 通道最高值（允许轻度带色）
LABEL_BORDER_DIST_PX   = 150   # 候选标签区域距主体 mask 边界最大距离（像素）
LABEL_MIN_AREA_PX      = 300   # 最小标签连通域面积（过滤噪点）
LABEL_WEIGHT           = 2.5   # 差异图中标签区域权重放大倍数
LABEL_V_PERCENTILE     = 85    # 自适应 V 阈值：bbox 内 V 通道的分位数
LABEL_ADAPTIVE_ENABLED = True  # 启用自适应标签阈值

# ==========================================
# 11b. 抠图（GrabCut）增强参数
# ==========================================
GRABCUT_ITERATIONS       = 5      # GrabCut 迭代次数（3~5）
GRABCUT_KEEP_RATIO       = 0.20   # 保留面积 > 最大连通域 × 此比例 的所有区域（提高门槛过滤噪点）
GRABCUT_FILL_HOLES       = True   # 填充 mask 内部孔洞
GRABCUT_USE_MASK_INIT    = True   # 用 MASK 方式初始化（比 RECT 更鲁棒）
GRABCUT_VERT_BRIDGE_PX   = 75     # 纵向闭运算核高度（像素），用于桥接上下断裂
# 工业现场背景复杂时，凸包易把斜向大块产线裁进「多边形状」蒙版，与 st/（rembg 柔边）观感不一致；
# GrabCut 仅作回退时建议 False；若产品轮廓极凹再改为 True。
GRABCUT_USE_CONVEX_HULL  = False

# ==========================================
# 11c. MODNet 抠图后端（主路径，离线部署）
# ==========================================
# st/ 透明标准图由 scripts/make_st_transparent.py 使用 rembg(isnet-general-use) 生成；
# 在线抠图若要与 st/ 一致，应使用 rembg（或 auto：MODNet→rembg→grabcut）。
# 树莓派仅 CPU：rembg 比 grabcut 慢但更贴近 st；无模型时可改 grabcut / auto。
SEGMENT_BACKEND = "rembg"  # modnet | auto | rembg | grabcut

# MODNet（ONNX）
MODNET_MODEL_PATH = os.path.join(MODELS_DIR, "modnet_photographic_portrait_matting.onnx")
MODNET_INPUT_SIZE = 512          # 建议 512/640；越大边缘越细，速度越慢
MODNET_ALPHA_THRESHOLD = 95      # alpha 二值阈值（越小越保守）
MODNET_ALLOW_FALLBACK = True     # MODNet 异常时回退 rembg/grabcut
MODNET_MIN_AREA_RATIO = 0.11     # mask 面积 / bbox 面积下限
MODNET_MAX_AREA_RATIO = 1.18     # mask 面积 / bbox 面积上限
MODNET_MAX_COMPONENTS = 6        # 主要连通域数量上限
MODNET_MIN_BOTTOM_COVERAGE = 0.84  # 最低底部覆盖率

# rembg 模型目录（离线优先）
REMBG_MODELS_DIR = os.path.join(MODELS_DIR, "rembg")
REMBG_MODEL_NAME = "isnet-general-use"       # u2net | u2netp | isnet-general-use
REMBG_ROI_PAD_RATIO = 0.28        # ROI 扩边略大，给 isnet 更多上下文（接近整图观感）
REMBG_ALPHA_THRESHOLD = 105       # alpha 二值化阈值（略低更「饱满」，与 st 略差异可试 95~115）
REMBG_ALLOW_FALLBACK = True       # rembg 异常时是否回退 grabcut
REMBG_ALPHA_MATTING = False       # isnet 原生输出已是柔和 alpha，matting 对硬边产品收益极小
REMBG_MAX_INPUT_SIDE = 1024       # 模型内部就在 1024 处理，限制输入不影响分割精度

# rembg mask 质量门控（过严会拒掉合理 mask → 退回 grabcut，抠图会像「多边形色块」与 st 不一致）
REMBG_MIN_AREA_RATIO = 0.12       # mask 面积 / bbox 面积下限
REMBG_MAX_AREA_RATIO = 1.22       # 略放宽：ROI 内夹具/高亮块略多时仍保留 rembg 结果
REMBG_MAX_COMPONENTS = 6          # 主要连通域数量上限
REMBG_MIN_HEIGHT_COVERAGE = 0.50  # mask 高度 / bbox 高度下限
# 顶视/悬空件时主体未必贴近 ROI 矩形下缘，0.86 易误拒绝 rembg
REMBG_MIN_BOTTOM_COVERAGE = 0.74  # mask 最底行在 ROI 高度中的最低归一化位置
REMBG_ALT_PAD_RATIO = 0.38        # 第二档更大扩边，与 REMBG_ROI_PAD_RATIO 不同才会多跑一轮候选
REMBG_FULL_RETRY_ENABLED = True   # ROI 全失败时整图 rembg（与 make_st_transparent 一致），较慢但更稳
# True：每帧先对整图跑一次 rembg（与生成 st/ 方式一致），大分辨率下最像 st，耗时会明显增加
REMBG_PREFLIGHT_FULL_FRAME = False
REMBG_ROTATE_DEGS = [0]           # 多角度更稳但会按角度数倍增耗时

# ==========================================
# 12. 姿态偏斜检测（最小外接矩形法）
# ==========================================
POSE_SKEW_WARN_DEG   = 12.0  # 偏斜角度超过此值输出 warning（但不停止检测）
POSE_SKEW_RETAKE_DEG = 28.0  # 偏斜角度超过此值触发 RETAKE

# ==========================================
# 13. 伪异常测试（数据增强，解决异常样本不足）
# ==========================================
PSEUDO_ANOMALY_ENABLED   = False   # 是否启用伪异常生成（仅用于实验评估）
PSEUDO_ANOMALY_TYPES     = ["erase_part", "occlude_label", "local_blur", "strong_reflection"]

# ==========================================
# 14. Anomalib PatchCore 异常检测
# ==========================================
ANOMALIB_MODEL_DIR       = os.path.join(MODELS_DIR, "anomalib")  # 按视角子文件夹
ANOMALIB_SCORE_THRESHOLD = 0.50    # 全局默认；未在 ANOMALIB_THRESH_BY_VIEW 列出的视角用此值
# 按视角 PatchCore 阈值（合格品误报多发面略抬高，由 batch CSV / eval 校准）
ANOMALIB_THRESH_BY_VIEW  = {
    "front": 0.50,
    "back":  0.56,
    "left":  0.62,
    "right": 0.62,
    "top":   0.62,
}
# 破洞面积占比超过此值 → FAIL（暗区启发；金属曲面易假阳时可略抬高）
ANOMALIB_HOLE_FAIL_THRESH = 0.003
# 模板 ORB 残差分位阈值（与 PatchCore 阈值无关）；抠图轮廓处残差常被拉高，宜单独标定
ANOMALIB_TEMPLATE_FAIL_THRESH = 0.28
# False：模板分不参与终判（仅 PatchCore + 破洞），热图也不叠加模板层，可减少抠图边缘误报
ANOMALIB_TEMPLATE_GATE_ENABLED = False
ANOMALIB_HEATMAP_ALPHA   = 0.45   # 热图叠加透明度（可视化用）
ANOMALIB_INPUT_SIZE      = 256    # 模型输入尺寸（训练与推理须一致）


def get_anomalib_threshold(view: str | None) -> float:
    """PatchCore 判 FAIL 用阈值；优先按视角，否则 ANOMALIB_SCORE_THRESHOLD。"""
    if not view:
        return float(ANOMALIB_SCORE_THRESHOLD)
    return float(ANOMALIB_THRESH_BY_VIEW.get(str(view), ANOMALIB_SCORE_THRESHOLD))