"""
优化版Stage1视觉初筛模块
结合Anomalib/PaDiM思想，使用预训练CNN特征+统计异常检测
无需不良品样本，只学习合格品的特征分布

核心改进：
1. 预训练CNN特征提取（ResNet18，轻量高效）
2. 位置敏感的特征统计（类似PaDiM）
3. 马氏距离异常检测
4. 多尺度特征融合
5. 自适应阈值策略
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from collections import defaultdict

# PyTorch相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# 项目内部导入
from .config import (
    MIN_FEATURE_SCORE, MIN_LOCALIZATION_SCORE,
    MIN_LOCALIZATION_AREA, PIXELS_PER_MM,
    DEBUG_MODE, MIN_MATCH_COUNT, SSIM_THRESHOLD
)

warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

class AdvancedStage1Config:
    """高级Stage1配置"""
    # 特征提取网络
    BACKBONE = 'resnet18'  # resnet18轻量，resnet50更准
    FEATURE_LAYERS = ['layer2', 'layer3']  # 提取哪些层的特征
    
    # 异常检测参数
    ANOMALY_THRESHOLD = 0.15  # 马氏距离阈值
    PIXEL_ANOMALY_THRESHOLD = 0.5  # 像素级异常阈值
    MIN_MISSING_AREA_MM2 = 10  # 最小缺件面积 mm²
    
    # 模板匹配参数
    TEMPLATE_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # 多尺度
    TEMPLATE_MATCH_METHOD = cv2.TM_CCOEFF_NORMED
    
    # 特征融合权重
    CNN_FEATURE_WEIGHT = 0.4
    ORB_FEATURE_WEIGHT = 0.3
    SHAPE_FEATURE_WEIGHT = 0.3
    
    # 自适应阈值
    ADAPTIVE_THRESHOLD_PERCENTILE = 95  # 取前5%作为异常
    
    # 推理设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AdvancedDetectionResult:
    """高级检测结果"""
    passed: bool
    status: str
    bbox: List[int]
    confidence: float
    
    # 异常分数
    anomaly_score: float  # 整体异常分数
    cnn_feature_score: float  # CNN特征得分
    orb_feature_score: float  # ORB特征得分
    shape_score: float  # 形状得分
    
    # 异常区域
    missing_regions: List[Dict]  # 缺件区域
    extra_regions: List[Dict]  # 多余区域
    
    # 异常图
    anomaly_map: Optional[np.ndarray]  # 像素级异常热力图
    
    # 详细信息
    issues: List[str]
    warnings: List[str]
    details: Dict
    
    # 标注图像
    annotated_image: Optional[np.ndarray]


# ============================================================================
# 预训练特征提取器（类似PaDiM）
# ============================================================================

class PretrainedFeatureExtractor:
    """
    预训练CNN特征提取器
    使用ImageNet预训练的ResNet，提取多层特征
    类似PaDiM方法，但更简化
    """
    
    def __init__(self, backbone='resnet18', layers=['layer2', 'layer3'], device='cpu'):
        self.device = device
        self.layers = layers
        
        # 加载预训练模型
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.dim = 128 * len(layers)  # layer2: 128, layer3: 256 -> 合计
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.dim = 512 * len(layers)
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # 注册hook获取中间层特征
        self.features = {}
        for name, layer in self.model.named_children():
            if name in layers:
                layer.register_forward_hook(self._hook(name))
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _hook(self, name):
        """Hook函数提取中间层特征"""
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像特征
        返回: 展平的特征向量 (H*W, C)
        """
        # 预处理
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 前向传播
        self.features = {}
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # 合并多层特征
        feature_list = []
        target_size = None
        
        for layer_name in self.layers:
            feat = self.features[layer_name]  # (1, C, H, W)
            feat = feat.squeeze(0)  # (C, H, W)
            
            # 记录第一层的空间尺寸作为目标
            if target_size is None:
                target_size = (feat.shape[1], feat.shape[2])
            
            # 上采样到目标尺寸（如果需要）
            if feat.shape[1] != target_size[0] or feat.shape[2] != target_size[1]:
                feat = F.interpolate(
                    feat.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            feat = feat.permute(1, 2, 0)  # (H, W, C)
            feat = feat.reshape(-1, feat.shape[-1])  # (H*W, C)
            feature_list.append(feat.cpu().numpy())
        
        # 合并特征
        combined = np.concatenate(feature_list, axis=1)  # (H*W, C_total)
        
        return combined
    
    def extract_spatial(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        提取空间特征图（保留空间位置信息）
        返回: (H, W, C) 特征图
        """
        # 预处理
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 前向传播
        self.features = {}
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # 合并多层特征
        feature_maps = []
        for layer_name in self.layers:
            feat = self.features[layer_name].squeeze(0)  # (C, H, W)
            feature_maps.append(feat)
        
        # 上采样到统一尺寸并合并
        target_h, target_w = feature_maps[0].shape[1], feature_maps[0].shape[2]
        upsampled = []
        for feat in feature_maps:
            if feat.shape[1] != target_h or feat.shape[2] != target_w:
                feat = F.interpolate(
                    feat.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            upsampled.append(feat)
        
        combined = torch.cat(upsampled, dim=0)  # (C_total, H, W)
        combined = combined.permute(1, 2, 0)  # (H, W, C_total)
        
        # 上采样到目标尺寸
        if target_size:
            combined = F.interpolate(
                combined.permute(2, 0, 1).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        
        return combined.cpu().numpy()


# ============================================================================
# 特征统计模型（类似PaDiM）
# ============================================================================

class FeatureStatistics:
    """
    特征统计模型
    学习正常样本的特征分布（均值和协方差）
    使用马氏距离计算异常分数
    """
    
    def __init__(self):
        self.means = None  # 各位置的均值 (H*W, C)
        self.covs = None   # 各位置的协方差矩阵 (H*W, C, C)
        self.inv_covs = None  # 协方差逆矩阵
        self.n_samples = 0
        self.feature_buffer = []  # 存储训练特征
    
    def fit(self, features_list: List[np.ndarray]):
        """
        训练：计算特征分布
        features_list: 多个样本的特征列表，每个 (H*W, C)
        """
        # 收集所有特征
        all_features = np.stack(features_list, axis=0)  # (N, H*W, C)
        self.n_samples = all_features.shape[0]
        
        # 计算各位置的均值和协方差
        n_positions = all_features.shape[1]
        n_channels = all_features.shape[2]
        
        self.means = np.mean(all_features, axis=0)  # (H*W, C)
        
        # 计算协方差（添加正则化确保可逆）
        self.covs = np.zeros((n_positions, n_channels, n_channels))
        eps = 1e-6  # 正则化系数
        
        for i in range(n_positions):
            pos_features = all_features[:, i, :]  # (N, C)
            cov = np.cov(pos_features.T) + eps * np.eye(n_channels)
            self.covs[i] = cov
        
        # 预计算逆协方差矩阵
        self.inv_covs = np.zeros_like(self.covs)
        for i in range(n_positions):
            try:
                self.inv_covs[i] = np.linalg.inv(self.covs[i])
            except:
                # 如果不可逆，使用伪逆
                self.inv_covs[i] = np.linalg.pinv(self.covs[i])
    
    def compute_anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """
        计算异常分数（马氏距离）
        features: (H*W, C)
        返回: (H*W,) 各位置的异常分数
        """
        if self.means is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 计算马氏距离
        delta = features - self.means  # (H*W, C)
        
        # 批量计算马氏距离
        anomaly_scores = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            d = delta[i:i+1]  # (1, C)
            # 马氏距离 = sqrt(d * Σ^-1 * d^T)
            mahal = np.sqrt(np.dot(np.dot(d, self.inv_covs[i]), d.T))
            anomaly_scores[i] = mahal[0, 0]
        
        return anomaly_scores
    
    def compute_anomaly_map(self, features: np.ndarray, 
                           original_size: Tuple[int, int]) -> np.ndarray:
        """
        计算异常热力图
        """
        scores = self.compute_anomaly_score(features)
        
        # 重塑为空间图
        h = w = int(np.sqrt(len(scores)))
        if h * w != len(scores):
            # 如果不是完美平方，调整
            h = int(np.sqrt(len(scores)))
            w = len(scores) // h
        
        anomaly_map = scores.reshape(h, w)
        
        # 上采样到原始尺寸
        anomaly_map = cv2.resize(anomaly_map, 
                                  (original_size[1], original_size[0]),
                                  interpolation=cv2.INTER_LINEAR)
        
        # 归一化到0-1
        anomaly_map = (anomaly_map - anomaly_map.min()) / \
                      (anomaly_map.max() - anomaly_map.min() + 1e-8)
        
        return anomaly_map
    
    def save(self, path: str):
        """保存模型"""
        np.savez(path,
                 means=self.means,
                 covs=self.covs,
                 inv_covs=self.inv_covs,
                 n_samples=self.n_samples)
    
    def load(self, path: str):
        """加载模型"""
        data = np.load(path)
        self.means = data['means']
        self.covs = data['covs']
        self.inv_covs = data['inv_covs']
        self.n_samples = int(data['n_samples'])


# ============================================================================
# 多模板管理器
# ============================================================================

class MultiTemplateManager:
    """
    多模板管理器
    管理不同角度的参考模板，支持多尺度匹配
    """
    
    def __init__(self):
        self.templates = defaultdict(list)  # angle -> [templates]
        self.template_features = defaultdict(list)  # angle -> [CNN features]
        self.template_orb = defaultdict(list)  # angle -> [ORB features]
        self.feature_extractor = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        
    def add_template(self, angle: str, image: np.ndarray):
        """添加模板"""
        self.templates[angle].append(image)
    
    def extract_all_features(self, feature_extractor: PretrainedFeatureExtractor):
        """预提取所有模板特征"""
        self.feature_extractor = feature_extractor
        
        for angle, templates in self.templates.items():
            for template in templates:
                # CNN特征
                cnn_feat = feature_extractor.extract(template)
                self.template_features[angle].append(cnn_feat)
                
                # ORB特征
                kp, des = self.orb.detectAndCompute(
                    cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None
                )
                self.template_orb[angle].append((kp, des))
    
    def find_best_match(self, image: np.ndarray, angle: str = None) -> Tuple[np.ndarray, float, str]:
        """
        找最佳匹配模板
        返回: (best_template, score, matched_angle)
        """
        test_feat = self.feature_extractor.extract(image)
        
        best_score = -1
        best_template = None
        best_angle = None
        
        # 搜索范围
        search_angles = [angle] if angle else list(self.templates.keys())
        
        for ang in search_angles:
            for i, template_feat in enumerate(self.template_features[ang]):
                # 简单的余弦相似度
                score = np.mean([
                    np.dot(test_feat.flatten(), tf.flatten()) / 
                    (np.linalg.norm(test_feat.flatten()) * np.linalg.norm(tf.flatten()) + 1e-8)
                    for tf in [template_feat]
                ])
                
                if score > best_score:
                    best_score = score
                    best_template = self.templates[ang][i]
                    best_angle = ang
        
        return best_template, best_score, best_angle


# ============================================================================
# 高级缺件检测器
# ============================================================================

class AdvancedMissingPartsDetector:
    """
    高级缺件检测器
    结合CNN异常检测和传统图像差分
    """
    
    def __init__(self, min_area_mm2: float = 10, pixels_per_mm: float = 4.8):
        self.min_area_pixels = min_area_mm2 * pixels_per_mm * pixels_per_mm
        self.feature_extractor = None
        
    def set_feature_extractor(self, extractor: PretrainedFeatureExtractor):
        self.feature_extractor = extractor
    
    def detect(self, reference: np.ndarray, test: np.ndarray,
               anomaly_map: np.ndarray = None) -> Tuple[List[Dict], List[Dict]]:
        """
        检测缺件和多余区域
        
        返回: (missing_regions, extra_regions)
        """
        # 1. 图像对齐
        ref_aligned, test_aligned = self._align_images(reference, test)
        
        # 2. 计算差分图
        diff_map = self._compute_difference_map(ref_aligned, test_aligned)
        
        # 3. 如果有异常图，融合异常信息
        if anomaly_map is not None:
            # 调整大小匹配
            anomaly_map_resized = cv2.resize(
                anomaly_map, (diff_map.shape[1], diff_map.shape[0])
            )
            # 加权融合
            diff_map = 0.5 * diff_map + 0.5 * anomaly_map_resized
        
        # 4. 自适应阈值
        threshold = self._adaptive_threshold(diff_map)
        
        # 5. 区域检测
        binary = (diff_map > threshold).astype(np.uint8)
        
        # 6. 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 7. 连通域分析
        missing, extra = self._analyze_regions(binary, ref_aligned, test_aligned)
        
        return missing, extra
    
    def _align_images(self, ref: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """图像对齐"""
        # 使用ORB特征对齐
        orb = cv2.ORB_create(nfeatures=500)
        
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if len(ref.shape) == 3 else ref
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) if len(test.shape) == 3 else test
        
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(test_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return ref, test
        
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 筛选好匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            return ref, test
        
        # 计算变换矩阵
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return ref, test
        
        # 变换测试图
        h, w = ref_gray.shape
        test_aligned = cv2.warpPerspective(test, M, (w, h))
        
        return ref, test_aligned
    
    def _compute_difference_map(self, ref: np.ndarray, test: np.ndarray) -> np.ndarray:
        """计算差分图"""
        # 确保尺寸一致
        if ref.shape != test.shape:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
        
        # 转换为灰度
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if len(ref.shape) == 3 else ref
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) if len(test.shape) == 3 else test
        
        # 多尺度差分
        diff_scales = []
        for scale in [1.0, 0.5, 0.25]:
            if scale != 1.0:
                ref_scaled = cv2.resize(ref_gray, None, fx=scale, fy=scale)
                test_scaled = cv2.resize(test_gray, None, fx=scale, fy=scale)
            else:
                ref_scaled = ref_gray
                test_scaled = test_gray
            
            diff = cv2.absdiff(ref_scaled, test_scaled)
            if scale != 1.0:
                diff = cv2.resize(diff, (ref_gray.shape[1], ref_gray.shape[0]))
            diff_scales.append(diff)
        
        # 融合多尺度差分
        combined = np.mean(diff_scales, axis=0)
        
        # 归一化
        combined = combined.astype(np.float32) / 255.0
        
        return combined
    
    def _adaptive_threshold(self, diff_map: np.ndarray) -> float:
        """自适应阈值"""
        # 使用OTSU或百分比阈值
        if np.std(diff_map) < 0.01:
            # 图像变化很小，使用较低阈值
            return 0.1
        
        # 使用95百分位作为阈值
        threshold = np.percentile(diff_map[diff_map > 0.05], 
                                  AdvancedStage1Config.ADAPTIVE_THRESHOLD_PERCENTILE)
        
        return max(0.1, min(threshold, 0.5))
    
    def _analyze_regions(self, binary: np.ndarray, ref: np.ndarray, test: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """分析连通域，区分缺件和多余区域"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        missing_regions = []
        extra_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_pixels:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 判断是缺件还是多余
            # 通过对比参考图和测试图对应区域的亮度
            ref_region = ref[y:y+h, x:x+w]
            test_region = test[y:y+h, x:x+w]
            
            ref_mean = np.mean(ref_region)
            test_mean = np.mean(test_region)
            
            is_missing = ref_mean > test_mean  # 参考图更亮，说明测试图缺件
            
            region_info = {
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'area_pixels': float(area),
                'area_mm2': float(area / (PIXELS_PER_MM ** 2)),
                'center': [int(x + w/2), int(y + h/2)],
                'type': 'missing' if is_missing else 'extra'
            }
            
            if is_missing:
                missing_regions.append(region_info)
            else:
                extra_regions.append(region_info)
        
        return missing_regions, extra_regions


# ============================================================================
# 主检测器类
# ============================================================================

class AdvancedStage1Detector:
    """
    高级Stage1检测器
    结合预训练CNN特征和传统视觉方法
    """
    
    def __init__(self, ref_folder: str, model_path: str = None):
        """
        初始化检测器
        
        Args:
            ref_folder: 参考图文件夹路径
            model_path: 预训练的特征统计模型路径（可选）
        """
        self.ref_folder = Path(ref_folder)
        self.config = AdvancedStage1Config()
        
        # 初始化组件
        self.feature_extractor = PretrainedFeatureExtractor(
            backbone=self.config.BACKBONE,
            layers=self.config.FEATURE_LAYERS,
            device=self.config.DEVICE
        )
        
        self.feature_stats = FeatureStatistics()
        self.template_manager = MultiTemplateManager()
        self.missing_detector = AdvancedMissingPartsDetector()
        
        # 加载参考模板
        self._load_reference_templates()
        
        # 加载预训练模型（如果有）
        if model_path and Path(model_path).exists():
            self.feature_stats.load(model_path)
            self._model_trained = True
        else:
            self._model_trained = False
        
        # ORB特征检测器（用于快速匹配）
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _load_reference_templates(self):
        """加载参考模板"""
        if not self.ref_folder.exists():
            raise FileNotFoundError(f"参考图文件夹不存在: {self.ref_folder}")
        
        # 按角度组织模板
        for angle_dir in self.ref_folder.iterdir():
            if angle_dir.is_dir():
                angle = angle_dir.name
                for img_file in angle_dir.glob("*.png"):
                    try:
                        img = self._read_image(str(img_file))
                        if img is not None:
                            self.template_manager.add_template(angle, img)
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"加载模板失败 {img_file}: {e}")
        
        # 提取所有模板特征
        self.template_manager.extract_all_features(self.feature_extractor)
    
    def _read_image(self, path: str) -> Optional[np.ndarray]:
        """读取图像（支持中文路径）"""
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except:
            return None
    
    def train(self, save_path: str = None):
        """
        训练特征统计模型
        只需要正常样本
        """
        print("开始训练特征统计模型...")
        
        features_list = []
        
        # 提取所有参考模板的特征
        for angle, templates in self.template_manager.templates.items():
            for template in templates:
                features = self.feature_extractor.extract(template)
                features_list.append(features)
        
        if len(features_list) > 0:
            self.feature_stats.fit(features_list)
            self._model_trained = True
            
            if save_path:
                self.feature_stats.save(save_path)
                print(f"模型已保存到: {save_path}")
        else:
            print("警告: 没有找到参考模板")
    
    def inspect(self, test_image: str, angle_hint: str = None) -> AdvancedDetectionResult:
        """
        执行检测
        
        Args:
            test_image: 测试图片路径
            angle_hint: 角度提示（可选，如 'front', 'back'）
        
        Returns:
            AdvancedDetectionResult
        """
        # 读取测试图像
        test_img = self._read_image(test_image)
        if test_img is None:
            return self._create_error_result(f"无法读取图像: {test_image}")
        
        # ========== Step 1: 产品定位 ==========
        localization_result = self._localize_product(test_img, angle_hint)
        if localization_result['confidence'] < MIN_LOCALIZATION_SCORE:
            return self._create_fail_result(
                test_img,
                f"产品定位失败，置信度: {localization_result['confidence']:.3f}",
                localization_result['bbox'],
                localization_result['confidence']
            )
        
        bbox = localization_result['bbox']
        matched_angle = localization_result['angle']
        matched_template = localization_result['template']
        
        # 提取产品区域
        x1, y1, x2, y2 = bbox
        product_roi = test_img[y1:y2, x1:x2]
        
        # ========== Step 2: CNN特征异常检测 ==========
        cnn_score, anomaly_map = self._compute_cnn_anomaly(product_roi, matched_template)
        
        # ========== Step 3: ORB特征匹配 ==========
        orb_score = self._compute_orb_similarity(product_roi, matched_template)
        
        # ========== Step 4: 形状匹配 ==========
        shape_score = self._compute_shape_similarity(product_roi, matched_template)
        
        # ========== Step 5: 缺件检测 ==========
        missing_regions, extra_regions = self.missing_detector.detect(
            matched_template, product_roi, anomaly_map
        )
        
        # ========== Step 6: 综合判定 ==========
        # 加权融合分数
        final_score = (
            self.config.CNN_FEATURE_WEIGHT * cnn_score +
            self.config.ORB_FEATURE_WEIGHT * orb_score +
            self.config.SHAPE_FEATURE_WEIGHT * shape_score
        )
        
        # 异常分数（越高越异常）
        anomaly_score = 1.0 - final_score
        
        # 判定是否通过
        passed = (
            final_score >= MIN_FEATURE_SCORE and
            len(missing_regions) == 0 and
            len(extra_regions) == 0
        )
        
        # 收集问题
        issues = []
        warnings = []
        
        if len(missing_regions) > 0:
            issues.append(f"检测到 {len(missing_regions)} 处缺件")
        if len(extra_regions) > 0:
            issues.append(f"检测到 {len(extra_regions)} 处多余件")
        if cnn_score < MIN_FEATURE_SCORE:
            warnings.append(f"CNN特征得分较低: {cnn_score:.3f}")
        if orb_score < MIN_MATCH_COUNT / 100:
            warnings.append(f"ORB特征匹配较少: {orb_score:.3f}")
        if shape_score < 0.5:
            warnings.append(f"形状相似度较低: {shape_score:.3f}")
        
        # 生成标注图像
        annotated = self._annotate_result(
            test_img, bbox, anomaly_map, missing_regions, extra_regions,
            passed, final_score, matched_angle
        )
        
        return AdvancedDetectionResult(
            passed=passed,
            status="PASS" if passed else "FAIL",
            bbox=bbox,
            confidence=localization_result['confidence'],
            anomaly_score=anomaly_score,
            cnn_feature_score=cnn_score,
            orb_feature_score=orb_score,
            shape_score=shape_score,
            missing_regions=missing_regions,
            extra_regions=extra_regions,
            anomaly_map=anomaly_map,
            issues=issues,
            warnings=warnings,
            details={
                'matched_angle': matched_angle,
                'localization_conf': localization_result['confidence'],
                'template_idx': localization_result['template_idx']
            },
            annotated_image=annotated
        )
    
    def _localize_product(self, image: np.ndarray, angle_hint: str = None) -> Dict:
        """产品定位"""
        best_template, score, matched_angle = self.template_manager.find_best_match(
            image, angle_hint
        )
        
        # 如果模板匹配得分不够高，尝试使用轮廓检测
        if score < 0.3:
            # 使用GrabCut进行前景提取
            bbox = self._grabcut_localize(image)
            if bbox is not None:
                return {
                    'bbox': bbox,
                    'confidence': 0.5,
                    'angle': matched_angle,
                    'template': best_template,
                    'template_idx': 0
                }
        
        # 使用模板尺寸估算边界框
        if best_template is not None:
            h, w = best_template.shape[:2]
            img_h, img_w = image.shape[:2]
            
            # 估算缩放比例
            scale = min(img_w / w, img_h / h) * 0.8
            new_w, new_h = int(w * scale), int(h * scale)
            
            x1 = (img_w - new_w) // 2
            y1 = (img_h - new_h) // 2
            
            return {
                'bbox': [x1, y1, x1 + new_w, y1 + new_h],
                'confidence': float(score),
                'angle': matched_angle,
                'template': best_template,
                'template_idx': 0
            }
        
        # 默认返回全图
        return {
            'bbox': [0, 0, image.shape[1], image.shape[0]],
            'confidence': 0.1,
            'angle': matched_angle,
            'template': best_template,
            'template_idx': 0
        }
    
    def _grabcut_localize(self, image: np.ndarray) -> Optional[List[int]]:
        """使用GrabCut进行产品定位"""
        h, w = image.shape[:2]
        
        # 使用图像中心作为初始框
        rect = (w//4, h//4, w//2, h//2)
        
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 提取前景区域
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # 找最大连通域
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                return [x, y, x+w, y+h]
        except:
            pass
        
        return None
    
    def _compute_cnn_anomaly(self, test_roi: np.ndarray, 
                            ref_roi: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算CNN特征异常分数"""
        if self._model_trained:
            # 使用训练好的统计模型
            test_features = self.feature_extractor.extract(test_roi)
            anomaly_map = self.feature_stats.compute_anomaly_map(
                test_features, test_roi.shape[:2]
            )
            
            # 整体异常分数
            anomaly_score = np.mean(anomaly_map)
            cnn_score = 1.0 - anomaly_score
            
        else:
            # 没有训练模型，使用简单的特征对比
            test_feat = self.feature_extractor.extract(test_roi)
            ref_feat = self.feature_extractor.extract(ref_roi)
            
            # 余弦相似度
            similarity = np.dot(test_feat.flatten(), ref_feat.flatten()) / \
                        (np.linalg.norm(test_feat.flatten()) * np.linalg.norm(ref_feat.flatten()) + 1e-8)
            
            cnn_score = float(similarity)
            anomaly_map = np.zeros(test_roi.shape[:2], dtype=np.float32)
        
        return cnn_score, anomaly_map
    
    def _compute_orb_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算ORB特征相似度"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # 特征匹配
        matches = self.bf_matcher.knnMatch(des1, des2, k=2)
        
        # 筛选好匹配
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        # 相似度
        score = len(good) / max(len(kp1), len(kp2), 1)
        
        return float(score)
    
    def _compute_shape_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算形状相似度（Hu矩）"""
        # 边缘检测
        edges1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1, 50, 150)
        edges2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2, 50, 150)
        
        # 轮廓
        contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 0.0
        
        # 取最大轮廓
        c1 = max(contours1, key=cv2.contourArea)
        c2 = max(contours2, key=cv2.contourArea)
        
        # Hu矩对比
        hu1 = cv2.HuMoments(cv2.moments(c1)).flatten()
        hu2 = cv2.HuMoments(cv2.moments(c2)).flatten()
        
        # 归一化Hu矩差异
        diff = np.abs(hu1 - hu2)
        # 忽略log变换
        diff = -np.sign(diff) * np.log10(np.abs(diff) + 1e-10)
        
        # 相似度
        similarity = 1.0 / (1.0 + np.mean(np.abs(diff)))
        
        return float(similarity)
    
    def _annotate_result(self, image: np.ndarray, bbox: List[int],
                         anomaly_map: np.ndarray,
                         missing: List[Dict], extra: List[Dict],
                         passed: bool, score: float, angle: str) -> np.ndarray:
        """生成标注结果图"""
        annotated = image.copy()
        
        # 产品边界框
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # 异常热力图叠加
        if anomaly_map is not None and np.max(anomaly_map) > 0:
            heatmap = cv2.applyColorMap(
                (anomaly_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            # 只在产品区域叠加
            roi = annotated[y1:y2, x1:x2]
            if roi.shape[:2] == heatmap.shape[:2]:
                cv2.addWeighted(roi, 0.7, heatmap, 0.3, 0, roi)
        
        # 标注缺件区域
        for region in missing:
            rx1, ry1, rx2, ry2 = region['bbox']
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(annotated, "MISSING", (rx1, ry1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 标注多余区域
        for region in extra:
            rx1, ry1, rx2, ry2 = region['bbox']
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            cv2.putText(annotated, "EXTRA", (rx1, ry1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 状态文字
        status_text = f"{'PASS' if passed else 'FAIL'} - {angle}"
        score_text = f"Score: {score:.3f}"
        cv2.putText(annotated, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, score_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return annotated
    
    def _create_error_result(self, message: str) -> AdvancedDetectionResult:
        """创建错误结果"""
        return AdvancedDetectionResult(
            passed=False,
            status="ERROR",
            bbox=[0, 0, 0, 0],
            confidence=0.0,
            anomaly_score=1.0,
            cnn_feature_score=0.0,
            orb_feature_score=0.0,
            shape_score=0.0,
            missing_regions=[],
            extra_regions=[],
            anomaly_map=None,
            issues=[message],
            warnings=[],
            details={'error': message},
            annotated_image=None
        )
    
    def _create_fail_result(self, image: np.ndarray, message: str,
                           bbox: List[int], confidence: float) -> AdvancedDetectionResult:
        """创建失败结果"""
        return AdvancedDetectionResult(
            passed=False,
            status="FAIL",
            bbox=bbox,
            confidence=confidence,
            anomaly_score=1.0,
            cnn_feature_score=0.0,
            orb_feature_score=0.0,
            shape_score=0.0,
            missing_regions=[],
            extra_regions=[],
            anomaly_map=None,
            issues=[message],
            warnings=[],
            details={'stage': 'localization'},
            annotated_image=image
        )


# ============================================================================
# 便捷函数
# ============================================================================

def create_advanced_detector(ref_folder: str, model_path: str = None) -> AdvancedStage1Detector:
    """
    创建高级检测器的便捷函数
    
    Args:
        ref_folder: 参考图文件夹路径
        model_path: 预训练模型路径（可选）
    
    Returns:
        AdvancedStage1Detector实例
    """
    detector = AdvancedStage1Detector(ref_folder, model_path)
    
    # 如果没有预训练模型，训练一个
    if not detector._model_trained:
        model_save_path = Path(ref_folder).parent / "models" / "feature_stats.npz"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        detector.train(str(model_save_path))
    
    return detector


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python stage1_advanced.py <测试图片路径> [角度提示]")
        sys.exit(1)
    
    test_img = sys.argv[1]
    angle_hint = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建检测器
    ref_folder = Path(__file__).parent.parent / "assets" / "images_ref"
    detector = create_advanced_detector(str(ref_folder))
    
    # 执行检测
    result = detector.inspect(test_img, angle_hint)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"检测结果: {result.status}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"异常分数: {result.anomaly_score:.3f}")
    print(f"CNN特征: {result.cnn_feature_score:.3f}")
    print(f"ORB特征: {result.orb_feature_score:.3f}")
    print(f"形状: {result.shape_score:.3f}")
    print(f"缺件数: {len(result.missing_regions)}")
    print(f"多余数: {len(result.extra_regions)}")
    
    if result.issues:
        print(f"\n问题:")
        for issue in result.issues:
            print(f"  - {issue}")
    
    if result.warnings:
        print(f"\n警告:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # 保存结果图
    if result.annotated_image is not None:
        output_path = Path(test_img).parent.parent / "output" / f"result_{Path(test_img).name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', result.annotated_image)[1].tofile(str(output_path))
        print(f"\n结果图已保存: {output_path}")
