"""
增强版视觉检测模块 - 电饭煲缺陷检测系统
结合特征匹配、形状匹配和模板匹配的多模态检测方案

主要功能：
1. 准确检测电饭煲位置并框出
2. 特征匹配 (ORB/SIFT) 
3. 形状匹配 (轮廓、Hu矩、形状上下文)
4. 缺件/漏件检测
5. 缺陷识别与分类
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .config import MIN_MATCH_COUNT, SSIM_THRESHOLD


def imread_safe(file_path: str) -> Optional[np.ndarray]:
    """
    安全读取图片，支持中文路径
    """
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[读取失败] {file_path}: {e}")
        return None


def imread_gray_safe(file_path: str) -> Optional[np.ndarray]:
    """
    安全读取灰度图片，支持中文路径
    """
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"[读取失败] {file_path}: {e}")
        return None


@dataclass
class DetectionResult:
    """检测结果数据类"""
    passed: bool
    status: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    feature_score: float
    shape_score: float
    similarity: float
    issues: List[str]
    warnings: List[str]
    missing_parts: List[Dict]  # 缺件区域
    extra_parts: List[Dict]    # 漏件区域
    annotated_image: Optional[np.ndarray] = None


class ShapeMatcher:
    """形状匹配器 - 使用Hu矩和轮廓匹配"""
    
    def __init__(self):
        self.contour_match_threshold = 0.8
        self.hu_moment_threshold = 0.05
    
    def compute_hu_moments(self, image: np.ndarray) -> np.ndarray:
        """计算Hu矩"""
        # 确保是二值图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算矩
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # 取对数使数值更稳定
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def compute_contour_similarity(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """计算轮廓相似度"""
        # 使用Hu矩比较
        hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
        hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()
        
        # 计算Hu矩差异
        diff = np.abs(hu1 - hu2)
        similarity = 1.0 - np.mean(diff)
        
        return max(0.0, min(1.0, similarity))
    
    def match_shapes(self, ref_img: np.ndarray, target_img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        形状匹配
        返回: (相似度分数, 差异图)
        """
        # 转换为灰度
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img.copy()
        
        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_img.copy()
        
        # 确保尺寸一致
        if ref_gray.shape != target_gray.shape:
            target_gray = cv2.resize(target_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # 边缘检测
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        target_edges = cv2.Canny(target_gray, 50, 150)
        
        # 计算边缘差异
        edge_diff = cv2.absdiff(ref_edges, target_edges)
        
        # 形态学处理消除噪声
        kernel = np.ones((3, 3), np.uint8)
        edge_diff = cv2.morphologyEx(edge_diff, cv2.MORPH_OPEN, kernel)
        
        # 计算形状相似度
        ref_edge_ratio = np.count_nonzero(ref_edges) / ref_edges.size
        diff_ratio = np.count_nonzero(edge_diff) / edge_diff.size
        
        if ref_edge_ratio > 0:
            shape_similarity = 1.0 - (diff_ratio / ref_edge_ratio)
        else:
            shape_similarity = 0.0
        
        shape_similarity = max(0.0, min(1.0, shape_similarity))
        
        return shape_similarity, edge_diff
    
    def extract_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """提取轮廓"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面积排序，返回主要轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return contours[:10]  # 返回前10个最大轮廓


class FeatureMatcher:
    """特征匹配器 - 使用ORB和SIFT"""
    
    def __init__(self, use_sift: bool = False):
        self.use_sift = use_sift
        if use_sift:
            try:
                self.detector = cv2.SIFT_create()
            except:
                self.detector = cv2.ORB_create(nfeatures=2000)
                self.use_sift = False
        else:
            self.detector = cv2.ORB_create(nfeatures=2000)
        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """提取特征点和描述符"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray) -> List:
        """特征匹配"""
        if des1 is None or des2 is None:
            return []
        
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    def compute_match_score(self, matches: List, kp1: List, kp2: List, 
                           img_shape: Tuple[int, int]) -> float:
        """计算匹配得分"""
        if len(matches) < 4:
            return 0.0
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return 0.0
            
            # 计算内点比例
            inlier_ratio = np.sum(mask) / len(matches)
            
            # 计算匹配点覆盖范围
            h, w = img_shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # 检查变换后的角点是否合理
            area = cv2.contourArea(transformed_corners)
            expected_area = w * h
            area_ratio = min(area / expected_area, expected_area / area) if expected_area > 0 else 0
            
            score = inlier_ratio * 0.6 + area_ratio * 0.4
            
            return min(1.0, max(0.0, score))
            
        except:
            return 0.0


class ProductLocalizer:
    """产品定位器 - 多尺度多方法定位"""
    
    def __init__(self):
        self.scales = [0.6, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5]
        self.template_methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    
    def multiscale_template_match(self, target: np.ndarray, 
                                   template: np.ndarray) -> Tuple[float, List[int], float]:
        """
        多尺度模板匹配
        返回: (最佳得分, 最佳边界框, 最佳缩放比例)
        """
        if len(target.shape) == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target.copy()
        
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        # 边缘增强
        target_edge = cv2.Canny(cv2.GaussianBlur(target_gray, (5, 5), 0), 50, 150)
        template_edge = cv2.Canny(cv2.GaussianBlur(template_gray, (5, 5), 0), 50, 150)
        
        best_score = -1
        best_bbox = [0, 0, target.shape[1], target.shape[0]]
        best_scale = 1.0
        
        h_t, w_t = target_gray.shape
        h_r, w_r = template_gray.shape
        
        for scale in self.scales:
            w_scaled = int(w_r * scale)
            h_scaled = int(h_r * scale)
            
            if w_scaled < 60 or h_scaled < 60:
                continue
            if w_scaled >= w_t or h_scaled >= h_t:
                continue
            
            # 缩放模板
            template_scaled = cv2.resize(template_gray, (w_scaled, h_scaled))
            template_edge_scaled = cv2.resize(template_edge, (w_scaled, h_scaled))
            
            for method in self.template_methods:
                try:
                    # 灰度匹配
                    result1 = cv2.matchTemplate(target_gray, template_scaled, method)
                    # 边缘匹配
                    result2 = cv2.matchTemplate(target_edge, template_edge_scaled, method)
                    
                    # 融合结果
                    result = result1 * 0.4 + result2 * 0.6
                    
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_score:
                        best_score = float(max_val)
                        x1, y1 = max_loc
                        x2, y2 = x1 + w_scaled, y1 + h_scaled
                        best_bbox = [x1, y1, x2, y2]
                        best_scale = scale
                        
                except Exception as e:
                    continue
        
        return best_score, best_bbox, best_scale
    
    def contour_based_localize(self, image: np.ndarray) -> Optional[List[int]]:
        """基于轮廓的定位"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # OTSU二值化
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学处理
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 添加边距
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        
        return [x, y, x + w, y + h]
    
    def refine_bbox(self, image: np.ndarray, initial_bbox: List[int], 
                    ref_img: np.ndarray) -> List[int]:
        """精修边界框"""
        x1, y1, x2, y2 = initial_bbox
        
        # 提取ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return initial_bbox
        
        # 基于GrabCut精修
        try:
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            rect = (5, 5, roi.shape[1]-10, roi.shape[0]-10)
            cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 提取前景掩码
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # 查找前景轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                rx, ry, rw, rh = cv2.boundingRect(max_contour)
                
                # 映射回原图坐标
                x1 = x1 + rx
                y1 = y1 + ry
                x2 = x1 + rw
                y2 = y1 + rh
                
        except:
            pass
        
        return [x1, y1, x2, y2]


class MissingPartsDetector:
    """缺件/漏件检测器"""
    
    def __init__(self):
        self.min_missing_area_ratio = 0.002  # 最小缺失区域比例
        self.max_extra_area_ratio = 0.001   # 最大额外区域比例
    
    def detect_missing_parts(self, ref_img: np.ndarray, 
                             target_img: np.ndarray,
                             bbox: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        检测缺件和漏件
        返回: (缺件列表, 漏件列表)
        """
        # 确保尺寸一致
        h, w = ref_img.shape[:2]
        if target_img.shape[:2] != (h, w):
            target_img = cv2.resize(target_img, (w, h))
        
        # 转换为灰度
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img.copy()
        
        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_img.copy()
        
        # 计算差异
        diff = cv2.absdiff(ref_gray, target_gray)
        
        # 阈值化
        _, diff_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)
        diff_binary = cv2.morphologyEx(diff_binary, cv2.MORPH_CLOSE, kernel)
        
        # 分离缺件和漏件区域
        ref_mask = self._create_object_mask(ref_gray)
        target_mask = self._create_object_mask(target_gray)
        
        # 缺件：参考图有物体，目标图没有
        missing_mask = cv2.bitwise_and(ref_mask, cv2.bitwise_not(target_mask))
        missing_mask = cv2.bitwise_and(missing_mask, diff_binary)
        
        # 漏件：目标图有物体，参考图没有
        extra_mask = cv2.bitwise_and(target_mask, cv2.bitwise_not(ref_mask))
        extra_mask = cv2.bitwise_and(extra_mask, diff_binary)
        
        # 查找缺件区域
        missing_parts = self._find_regions(missing_mask, "missing", bbox)
        
        # 查找漏件区域
        extra_parts = self._find_regions(extra_mask, "extra", bbox)
        
        return missing_parts, extra_parts
    
    def _create_object_mask(self, gray: np.ndarray) -> np.ndarray:
        """创建物体掩码"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return binary
    
    def _find_regions(self, mask: np.ndarray, region_type: str, 
                      bbox: List[int]) -> List[Dict]:
        """查找区域"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        total_area = mask.shape[0] * mask.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤小区域
            if area < total_area * self.min_missing_area_ratio:
                continue
            
            # 过滤细长区域（噪声）
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio > 10:
                continue
            
            # 计算中心点相对于边界框的位置
            cx = x + w // 2
            cy = y + h // 2
            
            regions.append({
                "type": region_type,
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "center": [cx, cy],
                "relative_pos": [
                    (cx - bbox[0]) / (bbox[2] - bbox[0] + 1e-6),
                    (cy - bbox[1]) / (bbox[3] - bbox[1] + 1e-6)
                ]
            })
        
        return regions
    
    def visualize_differences(self, ref_img: np.ndarray, 
                             target_img: np.ndarray,
                             missing_parts: List[Dict],
                             extra_parts: List[Dict]) -> np.ndarray:
        """可视化差异"""
        # 确保尺寸一致
        h, w = ref_img.shape[:2]
        if target_img.shape[:2] != (h, w):
            target_img = cv2.resize(target_img, (w, h))
        
        # 创建结果图
        result = target_img.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 绘制缺件区域（红色）
        for part in missing_parts:
            x1, y1, x2, y2 = part["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result, f"Missing: {part['area']:.0f}px", 
                       (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制漏件区域（蓝色）
        for part in extra_parts:
            x1, y1, x2, y2 = part["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result, f"Extra: {part['area']:.0f}px", 
                       (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result


class EnhancedVisionDetector:
    """增强版视觉检测器 - 主类"""
    
    def __init__(self, ref_folder: str, use_sift: bool = False):
        """
        初始化检测器
        
        Args:
            ref_folder: 参考图像文件夹路径
            use_sift: 是否使用SIFT（需要opencv-contrib）
        """
        self.ref_folder = os.path.normpath(ref_folder)
        
        # 初始化各模块
        self.feature_matcher = FeatureMatcher(use_sift=use_sift)
        self.shape_matcher = ShapeMatcher()
        self.localizer = ProductLocalizer()
        self.missing_detector = MissingPartsDetector()
        
        # 加载参考模板
        self.reference_templates = self._load_references()
        
        print(f"[EnhancedVision] 初始化完成，加载 {len(self.reference_templates)} 个参考模板")
    
    def _load_references(self) -> List[Dict]:
        """加载参考模板"""
        templates = []
        
        if not os.path.exists(self.ref_folder):
            print(f"[错误] 参考文件夹不存在: {self.ref_folder}")
            return templates
        
        # 递归遍历所有子文件夹
        for root, dirs, files in os.walk(self.ref_folder):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, filename)
                    
                    img = imread_safe(img_path)
                    if img is None:
                        continue
                    
                    # 提取特征
                    kp, des = self.feature_matcher.extract_features(img)
                    
                    # 提取轮廓
                    contours = self.shape_matcher.extract_contours(img)
                    
                    # 提取Hu矩
                    hu_moments = self.shape_matcher.compute_hu_moments(img)
                    
                    templates.append({
                        'path': img_path,
                        'name': filename,
                        'img': img,
                        'kp': kp,
                        'des': des,
                        'contours': contours,
                        'hu_moments': hu_moments,
                        'folder': os.path.basename(root)  # front/back/top/left/right
                    })
        
        return templates
    
    def inspect(self, target_input, min_feature_score: float = 0.3,
                min_shape_score: float = 0.7, min_localization_score: float = 0.25) -> DetectionResult:
        """
        执行完整检测
        
        Args:
            target_input: 图片路径(str) 或 图片数组(np.ndarray)
            min_feature_score: 最小特征得分阈值
            min_shape_score: 最小形状得分阈值
            min_localization_score: 最小定位得分阈值
            
        Returns:
            DetectionResult: 检测结果
        """
        result = DetectionResult(
            passed=False,
            status="FAIL",
            bbox=[0, 0, 0, 0],
            confidence=0.0,
            feature_score=0.0,
            shape_score=0.0,
            similarity=0.0,
            issues=[],
            warnings=[],
            missing_parts=[],
            extra_parts=[]
        )
        
        # 1. 加载目标图像
        if isinstance(target_input, str):
            target_img = imread_safe(os.path.normpath(target_input))
            if target_img is None:
                result.issues.append("无法加载目标图像")
                return result
        else:
            target_img = target_input.copy()
        
        h_img, w_img = target_img.shape[:2]
        
        if not self.reference_templates:
            result.issues.append("未加载参考模板")
            return result
        
        # 2. 多模板匹配，选择最佳参考图
        best_template = None
        best_loc_score = -1
        best_bbox = [0, 0, w_img, h_img]
        best_scale = 1.0
        
        for template in self.reference_templates:
            score, bbox, scale = self.localizer.multiscale_template_match(
                target_img, template['img']
            )
            
            if score > best_loc_score:
                best_loc_score = score
                best_bbox = bbox
                best_scale = scale
                best_template = template
        
        result.bbox = best_bbox
        result.confidence = best_loc_score
        
        if best_loc_score < min_localization_score:
            result.issues.append(f"产品定位置信度低 ({best_loc_score:.3f})，请检查拍摄角度和位置")
        
        # 3. 提取ROI
        x1, y1, x2, y2 = best_bbox
        roi = target_img[y1:y2, x1:x2]
        
        if roi.size == 0:
            result.issues.append("ROI区域无效")
            return result
        
        # 4. 特征匹配评分
        if best_template is not None:
            kp_roi, des_roi = self.feature_matcher.extract_features(roi)
            matches = self.feature_matcher.match_features(
                best_template['des'], des_roi
            )
            
            feature_score = self.feature_matcher.compute_match_score(
                matches, best_template['kp'], kp_roi, roi.shape
            )
            result.feature_score = feature_score
            
            if feature_score < min_feature_score:
                result.warnings.append(f"特征匹配得分偏低 ({feature_score:.3f})")
        
        # 5. 形状匹配评分
        if best_template is not None:
            # 调整参考图尺寸以匹配ROI
            ref_resized = cv2.resize(best_template['img'], (x2-x1, y2-y1))
            shape_score, diff_img = self.shape_matcher.match_shapes(ref_resized, roi)
            result.shape_score = shape_score
            
            if shape_score < min_shape_score:
                result.warnings.append(f"形状匹配得分偏低 ({shape_score:.3f})")
        
        # 6. 缺件/漏件检测
        if best_template is not None:
            ref_resized = cv2.resize(best_template['img'], (x2-x1, y2-y1))
            missing_parts, extra_parts = self.missing_detector.detect_missing_parts(
                ref_resized, roi, best_bbox
            )
            result.missing_parts = missing_parts
            result.extra_parts = extra_parts
            
            if len(missing_parts) > 0:
                result.issues.append(f"检测到 {len(missing_parts)} 处缺件")
            if len(extra_parts) > 0:
                result.issues.append(f"检测到 {len(extra_parts)} 处漏件")
        
        # 7. 计算综合相似度
        result.similarity = 0.4 * result.feature_score + 0.4 * result.shape_score + 0.2 * result.confidence
        
        # 8. 最终判定
        passed = (len(result.issues) == 0 and 
                  result.feature_score >= min_feature_score and
                  result.shape_score >= min_shape_score and
                  len(result.missing_parts) == 0 and
                  len(result.extra_parts) == 0)
        
        result.passed = passed
        result.status = "PASS" if passed else "FAIL"
        
        # 9. 生成标注图像
        result.annotated_image = self._draw_result(target_img, result)
        
        return result
    
    def _draw_result(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """绘制检测结果"""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # 绘制产品边界框
        x1, y1, x2, y2 = result.bbox
        box_color = (0, 255, 0) if result.passed else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(vis, "Product ROI", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # 绘制缺件区域
        for part in result.missing_parts:
            px1, py1, px2, py2 = part['bbox']
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(vis, "MISSING", (px1, max(15, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制漏件区域
        for part in result.extra_parts:
            px1, py1, px2, py2 = part['bbox']
            cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 165, 0), 2)
            cv2.putText(vis, "EXTRA", (px1, max(15, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # 绘制状态信息
        status_color = (0, 200, 0) if result.passed else (0, 0, 255)
        status_text = f"Status: {result.status}"
        cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        score_text = f"Feature: {result.feature_score:.2f} | Shape: {result.shape_score:.2f} | Similarity: {result.similarity:.2f}"
        cv2.putText(vis, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def inspect_batch(self, folder: str, output_folder: str = None) -> List[DetectionResult]:
        """批量检测"""
        results = []
        
        if not os.path.exists(folder):
            print(f"[错误] 文件夹不存在: {folder}")
            return results
        
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for filename in os.listdir(folder):
            if not filename.lower().endswith(valid_extensions):
                continue
            
            img_path = os.path.join(folder, filename)
            print(f"\n正在检测: {filename}")
            
            result = self.inspect(img_path)
            results.append(result)
            
            # 保存结果
            if output_folder and result.annotated_image is not None:
                out_path = os.path.join(output_folder, f"result_{filename}")
                try:
                    # 使用numpy保存以支持中文路径
                    _, ext = os.path.splitext(out_path)
                    cv2.imencode(ext, result.annotated_image)[1].tofile(out_path)
                except Exception as e:
                    print(f"保存失败: {e}")
        
        # 统计结果
        passed = sum(1 for r in results if r.passed)
        print(f"\n{'='*50}")
        print(f"批量检测完成: {passed}/{len(results)} 通过")
        print(f"{'='*50}")
        
        return results


# === 便捷函数 ===

def create_detector(ref_folder: str) -> EnhancedVisionDetector:
    """
    创建检测器的便捷函数
    """
    return EnhancedVisionDetector(ref_folder)


def quick_inspect(ref_folder: str, target_path: str) -> DetectionResult:
    """
    快速检测的便捷函数
    """
    detector = EnhancedVisionDetector(ref_folder)
    return detector.inspect(target_path)
