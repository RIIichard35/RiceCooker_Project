import cv2
import numpy as np
import os
from .config import MIN_MATCH_COUNT

# === 🛠️ 关键修复：支持中文路径的读取函数 ===
def imread_safe(file_path):
    """
    解决 OpenCV 在 Windows 上无法读取中文路径的问题
    """
    try:
        # 使用 numpy 读取文件流，再解码，绕过 OpenCV 的路径限制
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE) # 直接读为灰度
        return img
    except Exception as e:
        print(f"[读取失败] {file_path}: {e}")
        return None

class Stage1Detector:
    def __init__(self, ref_folder_path):
        """
        初始化检测器，加载指定文件夹下的所有参考图
        """
        self.reference_features = [] # 存储多张金样的特征
        
        # 路径标准化，防止斜杠问题
        ref_folder_path = os.path.normpath(ref_folder_path)

        if not os.path.exists(ref_folder_path):
            print(f"[Stage1] [错误] 找不到参考图文件夹: {ref_folder_path}")
            return

        # 1. 初始化 ORB
        self.orb = cv2.ORB_create(nfeatures=1000)

        # 2. 遍历文件夹读取所有图片
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        print(f"[Stage1] 正在加载多模板金样库: {ref_folder_path} ...")
        
        count = 0
        if os.path.isdir(ref_folder_path):
            files = os.listdir(ref_folder_path)
            for filename in files:
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(ref_folder_path, filename)
                    
                    # ⚠️ 使用修复后的读取函数
                    img = imread_safe(img_path)
                    
                    if img is not None:
                        # 计算特征
                        kp, des = self.orb.detectAndCompute(img, None)
                        if des is not None:
                            self.reference_features.append({
                                'kp': kp, 'des': des, 'img': img, 'name': filename
                            })
                            count += 1
                    else:
                        print(f"[警告] 无法读取金样图片 (可能是路径或格式问题): {filename}")
        else:
             print(f"[Stage1] [错误] 提供给 Stage1 的路径不是一个文件夹！")

        print(f"[Stage1] 初始化完成。成功加载 {count} 张参考标准图。")

    def check_integrity(self, target_img_path):
        """
        一级检测：多模板匹配逻辑
        """
        # 路径标准化
        target_img_path = os.path.normpath(target_img_path)

        if not self.reference_features:
            return False, "Init Failed (No Ref Images Loaded)", None

        # ⚠️ 使用修复后的读取函数
        target_img = imread_safe(target_img_path)
        
        if target_img is None:
            return False, "Image Load Error (Check Path/Chinese characters)", None

        # 2. 计算待测图特征
        kp_target, des_target = self.orb.detectAndCompute(target_img, None)
        if des_target is None or len(kp_target) == 0:
            return False, "No features in target", target_img

        # 3. 轮询匹配 (Multi-template Matching)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        best_match_count = -1
        best_ref_data = None
        best_matches_list = []

        # 遍历每一张金样
        for ref_data in self.reference_features:
            if ref_data['des'] is None: continue # 跳过无效特征
            
            matches = bf.match(ref_data['des'], des_target)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 取前 15% 优质点
            keep_percent = 0.15
            if len(matches) > 0:
                good_matches = matches[:int(len(matches) * keep_percent)]
                current_count = len(good_matches)
            else:
                current_count = 0
            
            if current_count > best_match_count:
                best_match_count = current_count
                best_ref_data = ref_data
                best_matches_list = good_matches

        # 4. 绘图
        debug_img = None
        if best_ref_data is not None and target_img is not None:
            debug_img = cv2.drawMatches(best_ref_data['img'], best_ref_data['kp'], 
                                        target_img, kp_target, 
                                        best_matches_list, None, flags=2)

        # 5. 判定
        matched_name = best_ref_data['name'] if best_ref_data else "None"
        print(f"   >>> [调试] 最佳匹配金样: {matched_name} | 匹配点数: {best_match_count}")
        
        if best_match_count < MIN_MATCH_COUNT:
            return False, f"FAIL: Mismatch (Best={best_match_count})", debug_img
        else:
            return True, f"PASS: Integrity OK (Best={best_match_count})", debug_img

    def _evaluate_roi_quality(self, roi_bgr):
        """
        评估 ROI 图像质量，用于判断覆膜反光/虚焦/纹理过弱等问题。
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 纹理复杂度：边缘占比过低往往意味着模糊、过曝、覆膜强反光或拍摄角度不佳
        edges = cv2.Canny(gray, 80, 180)
        texture_ratio = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        _ = h  # 避免未使用变量告警
        # 反光粗略指标：高亮且低饱和像素占比
        highlight_mask = ((v > 220) & (s < 45)).astype(np.uint8)
        reflection_ratio = float(np.count_nonzero(highlight_mask)) / float(highlight_mask.size + 1e-6)

        return {
            "sharpness": sharpness,
            "texture_ratio": texture_ratio,
            "reflection_ratio": reflection_ratio
        }

    def _simple_similarity(self, ref_gray, aligned_gray):
        """
        计算简化版结构相似度（0~1），避免依赖额外第三方库。
        """
        abs_diff = cv2.absdiff(ref_gray, aligned_gray)
        norm_mean = float(np.mean(abs_diff)) / 255.0
        return max(0.0, 1.0 - norm_mean)

    def _extract_product_mask(self, gray_img):
        """
        从图像中提取产品主体掩膜（最大连通域），用于材料缺失判定。
        """
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # OTSU 自动阈值，适应不同光照
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 兼容明暗背景：选择面积更大的前景解释
        fg_area = int(np.count_nonzero(bin_img))
        inv = cv2.bitwise_not(bin_img)
        inv_area = int(np.count_nonzero(inv))
        if inv_area > fg_area:
            bin_img = inv

        kernel = np.ones((5, 5), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray_img, dtype=np.uint8)

        max_cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.drawContours(mask, [max_cnt], -1, 255, thickness=-1)
        return mask

    def _clamp_bbox(self, x1, y1, x2, y2, w_img, h_img):
        x1 = max(0, min(int(x1), w_img - 1))
        y1 = max(0, min(int(y1), h_img - 1))
        x2 = max(x1 + 1, min(int(x2), w_img))
        y2 = max(y1 + 1, min(int(y2), h_img))
        return x1, y1, x2, y2

    def _multiscale_template_localize(self, target_gray):
        """
        多尺度模板匹配做粗定位，优先保证产品框稳定。
        返回:
        {
            "score": float,
            "ref_data": dict,
            "bbox": [x1, y1, x2, y2],
            "scale": float
        }
        """
        h_t, w_t = target_gray.shape[:2]
        target_blur = cv2.GaussianBlur(target_gray, (5, 5), 0)
        target_edge = cv2.Canny(target_blur, 50, 150)
        scales = [0.75, 0.85, 1.0, 1.15, 1.3]

        best = {"score": -1.0, "ref_data": None, "bbox": None, "scale": 1.0}
        for ref_data in self.reference_features:
            ref_gray = ref_data["img"]
            h_r0, w_r0 = ref_gray.shape[:2]
            for scale in scales:
                w_r = int(w_r0 * scale)
                h_r = int(h_r0 * scale)
                if w_r < 80 or h_r < 80:
                    continue
                if w_r >= w_t or h_r >= h_t:
                    continue

                ref_resized = cv2.resize(ref_gray, (w_r, h_r), interpolation=cv2.INTER_AREA)
                ref_edge = cv2.Canny(cv2.GaussianBlur(ref_resized, (5, 5), 0), 50, 150)
                match = cv2.matchTemplate(target_edge, ref_edge, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(match)

                if max_val > best["score"]:
                    x1, y1 = max_loc
                    x2, y2 = x1 + w_r, y1 + h_r
                    x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, w_t, h_t)
                    best = {
                        "score": float(max_val),
                        "ref_data": ref_data,
                        "bbox": [x1, y1, x2, y2],
                        "scale": scale
                    }
        return best

    def _detect_missing_regions(self, ref_gray, aligned_target_gray):
        """
        缺件定义：参考中存在材料，但当前样品对应区域材料缺失。
        通过主体掩膜差异 (ref_mask - target_mask) 定位缺失区域。
        """
        ref_mask = self._extract_product_mask(ref_gray)
        tgt_mask = self._extract_product_mask(aligned_target_gray)

        # 只保留“参考有、当前无”的区域 = 材料缺失候选
        binary = cv2.bitwise_and(ref_mask, cv2.bitwise_not(tgt_mask))

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        h, w = ref_gray.shape[:2]
        area_min = max(380, int(0.0018 * w * h))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < area_min:
                continue
            # 细长噪声过滤 + 极端大区域过滤
            ratio = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
            if ratio > 10:
                continue
            if area > int(0.35 * w * h):
                continue
            boxes.append((x, y, bw, bh))
        return boxes

    def inspect_with_localization(
        self,
        target_input,
        min_match_count=None,
        min_localization_score=0.20,
        min_similarity_fail=0.72,
        missing_regions_fail_count=2
    ):
        """
        新版 Stage1 初筛：
        1) 多模板匹配并选择最佳模板
        2) 通过单应性框出产品区域
        3) 给出初筛结论与问题列表
        4) 标注疑似缺件/少件区域

        :param target_input: 图片路径(str) 或 BGR图(np.ndarray)
        :param min_match_count: 可动态覆盖阈值
        :return: dict
        """
        threshold = int(min_match_count if min_match_count is not None else MIN_MATCH_COUNT)
        result = {
            "pass": False,
            "status": "FAIL",
            "score": 0,
            "threshold": threshold,
            "localization_score": 0.0,
            "best_ref_name": None,
            "bbox": None,
            "polygon": None,
            "similarity": 0.0,
            "quality": {},
            "issues": [],
            "warnings": [],
            "missing_regions": [],
            "annotated_image": None
        }

        if not self.reference_features:
            result["issues"].append("参考图未加载成功")
            return result

        # 读取目标图
        if isinstance(target_input, str):
            target_gray = imread_safe(os.path.normpath(target_input))
            if target_gray is None:
                result["issues"].append("待测图读取失败")
                return result
            target_bgr = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR)
        else:
            if target_input is None or not hasattr(target_input, "shape"):
                result["issues"].append("待测图无效")
                return result
            if len(target_input.shape) == 2:
                target_gray = target_input.copy()
                target_bgr = cv2.cvtColor(target_gray, cv2.COLOR_GRAY2BGR)
            else:
                target_bgr = target_input.copy()
                target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

        h_img, w_img = target_gray.shape[:2]
        vis = target_bgr.copy()

        # Step 1: 先用模板匹配稳定框产品，避免先前 homography 容易跑偏到背景
        coarse = self._multiscale_template_localize(target_gray)
        if coarse["ref_data"] is None or coarse["bbox"] is None:
            result["issues"].append("无法定位产品主体，请调整摆放和拍摄位置")
            result["annotated_image"] = vis
            return result

        x1, y1, x2, y2 = coarse["bbox"]
        result["bbox"] = [x1, y1, x2, y2]
        result["localization_score"] = float(coarse["score"])
        result["best_ref_name"] = coarse["ref_data"]["name"]
        result["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        # 只基于框内区域做后续判定
        roi_gray = target_gray[y1:y2, x1:x2]
        roi_bgr = target_bgr[y1:y2, x1:x2]
        if roi_gray.size == 0:
            result["issues"].append("产品框无效，请调整相机与产品位置")
            result["annotated_image"] = vis
            return result

        ref_gray = coarse["ref_data"]["img"]
        h_ref, w_ref = ref_gray.shape[:2]
        roi_resized = cv2.resize(roi_gray, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

        # Step 2: 仅 ROI 内做特征得分和相似度
        kp_roi, des_roi = self.orb.detectAndCompute(roi_gray, None)
        kp_ref, des_ref = self.orb.detectAndCompute(ref_gray, None)
        if des_roi is None or des_ref is None:
            result["score"] = 0
            result["warnings"].append("框内特征较少，建议补光并减少反光")
        else:
            bf_local = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_local.match(des_ref, des_roi)
            matches = sorted(matches, key=lambda m: m.distance)
            keep_n = max(8, int(len(matches) * 0.20))
            result["score"] = int(min(len(matches), keep_n))

        similarity = self._simple_similarity(ref_gray, roi_resized)
        result["similarity"] = float(similarity)

        # Step 3: 缺件/少件疑似区域 (仅框内比较)
        missing_boxes_ref = self._detect_missing_regions(ref_gray, roi_resized)
        for (mx, my, mw, mh) in missing_boxes_ref:
            # 从参考平面映射到 ROI，再映射回整图坐标
            rx1 = x1 + int(mx * (x2 - x1) / float(w_ref))
            ry1 = y1 + int(my * (y2 - y1) / float(h_ref))
            rx2 = x1 + int((mx + mw) * (x2 - x1) / float(w_ref))
            ry2 = y1 + int((my + mh) * (y2 - y1) / float(h_ref))
            rx1, ry1, rx2, ry2 = self._clamp_bbox(rx1, ry1, rx2, ry2, w_img, h_img)
            if (rx2 - rx1) * (ry2 - ry1) < 120:
                continue
            result["missing_regions"].append([rx1, ry1, rx2, ry2])
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(vis, "Missing/Suspect", (rx1, max(20, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Step 4: 质量评估（严格只看产品框内）
        quality = self._evaluate_roi_quality(roi_bgr)
        result["quality"] = quality

        # Step 5: 阻断项/告警项分离，降低误杀正常品
        if result["localization_score"] < float(min_localization_score):
            result["issues"].append(
                f"产品定位置信度低 ({result['localization_score']:.2f})，请调整摆放后重拍"
            )
        if result["score"] < max(8, int(threshold * 0.60)) and similarity < float(min_similarity_fail):
            result["issues"].append(
                f"框内匹配与相似度均偏低 (match={result['score']}, sim={similarity:.2f})"
            )
        if len(result["missing_regions"]) >= int(missing_regions_fail_count) and similarity < 0.80:
            result["issues"].append(f"发现 {len(result['missing_regions'])} 处疑似缺件/少件区域")

        if quality["sharpness"] < 22:
            result["warnings"].append(f"清晰度偏低 (sharpness={quality['sharpness']:.1f})")
        if quality["texture_ratio"] < 0.012:
            result["warnings"].append(
                f"纹理偏弱 (texture={quality['texture_ratio']:.3f})，疑似覆膜/过曝"
            )
        if quality["reflection_ratio"] > 0.35:
            result["warnings"].append(
                f"反光偏强 (reflection={quality['reflection_ratio']:.3f})，疑似塑料膜干扰"
            )
        if len(result["missing_regions"]) == 1 and similarity >= 0.80:
            result["warnings"].append("存在 1 处轻微差异区域，建议人工复核")

        result["pass"] = len(result["issues"]) == 0
        result["status"] = "PASS" if result["pass"] else "FAIL"
        status_color = (0, 180, 0) if result["pass"] else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(vis, "Product ROI", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.putText(
            vis,
            f"Stage1 {result['status']} | loc={result['localization_score']:.2f} | match={result['score']} | sim={result['similarity']:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2
        )
        result["annotated_image"] = vis
        return result