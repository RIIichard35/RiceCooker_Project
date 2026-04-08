# src/stage3_rules.py
from .config import PIXELS_PER_MM, MIN_DEFECT_SIZE_MM

class Stage3RuleEngine:
    def __init__(self):
        pass

    def calculate_iou_contains(self, defect_box, film_box):
        """判断缺陷是否在塑料膜内"""
        dx1, dy1, dx2, dy2 = defect_box
        fx1, fy1, fx2, fy2 = film_box

        ix1 = max(dx1, fx1); iy1 = max(dy1, fy1)
        ix2 = min(dx2, fx2); iy2 = min(dy2, fy2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        defect_area = (dx2 - dx1) * (dy2 - dy1)
        
        if defect_area == 0: return False
        return (inter_area / defect_area) > 0.6 # 60%以上在膜内就算被覆盖

    def apply_rules(self, detections):
        """
        修正后的逻辑：
        1. 必须是 scratch/dent 才算缺陷。
        2. 如果在 plastic_film 里，忽略。
        3. 如果太小，忽略。
        4. 剩下的如果数量 > 0，直接 NG。
        """
        
        # 1. 提取环境特征 (膜、面板)
        films = [d['box'] for d in detections if d['cls'] == 'plastic_film']
        
        processed_detections = []
        valid_defect_count = 0 # 有效划痕计数

        for item in detections:
            # 复制一份，以免修改原数据
            d = item.copy()
            
            # --- 初始化状态 ---
            d['status'] = 'NG' 
            d['color'] = (255, 0, 0) # 默认红色 (BGR格式: Blue, Green, Red)
            d['display_label'] = d['cls']

            # --- 过滤器 1: 非缺陷过滤 ---
            # 如果 AI 检出了膜、面板、或者之前的通用物体(人、杯子)，一律忽略
            if d['cls'] not in ['scratch', 'dent']:
                d['status'] = 'INFO'
                d['color'] = (255, 255, 0) # 黄色
                # 如果是膜，我们甚至不需要要在图上标出来干扰视线，或者标个浅色
                if d['cls'] == 'plastic_film':
                     d['display_label'] = "Film (Ignored)"
                processed_detections.append(d)
                continue

            # --- 过滤器 2: 覆膜过滤 (抗干扰) ---
            is_in_film = False
            for film_box in films:
                if self.calculate_iou_contains(d['box'], film_box):
                    is_in_film = True
                    break
            
            if is_in_film:
                d['status'] = 'IGNORED'
                d['display_label'] = f"{d['cls']} (In Film)"
                d['color'] = (255, 165, 0) # 橙色
                processed_detections.append(d)
                continue

            # --- 过滤器 3: 尺寸过滤 (去噪点) ---
            w = d['box'][2] - d['box'][0]
            h = d['box'][3] - d['box'][1]
            size_mm = max(w, h) / PIXELS_PER_MM
            
            if size_mm < MIN_DEFECT_SIZE_MM:
                d['status'] = 'IGNORED'
                d['display_label'] = f"{d['cls']} (<{MIN_DEFECT_SIZE_MM}mm)"
                d['color'] = (0, 255, 0) # 绿色
                processed_detections.append(d)
                continue

            # --- 判定为真缺陷 ---
            valid_defect_count += 1
            d['display_label'] = f"{d['cls']} {d['conf']:.2f}" # 像报告一样显示置信度
            processed_detections.append(d)

        # --- 最终大判决 ---
        if valid_defect_count > 0:
            final_verdict = "FAIL" # 只要有一个有效划痕，就是不良品
            final_desc = f"检测到 {valid_defect_count} 处有效瑕疵"
        else:
            final_verdict = "PASS"
            final_desc = "合格品 (无瑕疵或瑕疵已忽略)"

        return final_verdict, processed_detections, final_desc