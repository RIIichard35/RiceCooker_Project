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