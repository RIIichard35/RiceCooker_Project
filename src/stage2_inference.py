# 二级检测算法待实现
from ultralytics import YOLO
import cv2
import numpy as np
import os

class Stage2Detector:
    def __init__(self, model_path):
        """
        初始化 YOLO 模型
        """
        self.model = None
        
        # 检查模型是否存在
        if os.path.exists(model_path):
            print(f"[Stage2] 正在加载本地模型: {model_path} ...")
            self.model = YOLO(model_path)
        else:
            print(f"[Stage2] ⚠️ 未找到模型文件: {model_path}")
            print(f"[Stage2] 尝试自动下载官方 yolov8n.pt 模型...")
            try:
                self.model = YOLO("yolov8n.pt") # 如果没找到指定模型，就用官方原本的
            except Exception as e:
                print(f"[Stage2] ❌ 模型加载极其失败: {e}")

    def detect_defects(self, image_input):
        """
        执行推理
        :param image_input: 可以是图片路径(str)，也可以是内存中的图片(numpy array)
        :return: detections 列表
        """
        detections = []
        
        if self.model is None:
            return []

        # 运行推理 (conf=0.15 降低阈值，让它尽可能多检出东西以便观察)
        results = self.model.predict(image_input, conf=0.15, verbose=False)
        result = results[0]
        
        # 解析结果
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id] # 获取类别名称
            xyxy = box.xyxy[0].cpu().numpy().tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            
            detections.append({
                "cls": cls_name,
                "box": xyxy,
                "conf": conf,
                "status": "WAITING" # 等待三级判决
            })
            
        return detections