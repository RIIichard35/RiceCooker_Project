from ultralytics import YOLO
import cv2
import os

class Stage2Detector:
    def __init__(self, model_path):
        """在初始化时加载模型，整个生命周期只加载一次"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到 YOLO 模型: {model_path}")
        
        print("⏳ 正在加载 YOLOv8 模型到内存 (这可能需要几秒钟)...")
        # 如果在树莓派上内存吃紧，后续可以考虑在这里加入特定参数，或转为 ONNX/NCNN 格式
        self.model = YOLO(model_path)
        print("✅ 模型加载完毕！")

    def detect(self, img_path, conf_threshold=0.25, debug=False):
        """
        对单张图片进行缺陷检测
        :return: (has_defect, msg, result_image)
        """
        if not os.path.exists(img_path):
            return False, f"找不到待测图片: {img_path}", None

        # 运行推理 (禁止自动 save 和显示，由代码主动控制)
        results = self.model.predict(source=img_path, conf=conf_threshold, save=False, verbose=False)
        
        # 提取结果
        result_obj = results[0]
        boxes = result_obj.boxes
        
        has_defect = len(boxes) > 0  # 如果检测到任何框，说明有缺陷
        
        # 绘制带框的结果图
        res_plotted = result_obj.plot()

        if debug:
            cv2.imshow("Stage 2 Defect Detection", res_plotted)
            print("💡 提示: 按键盘任意键关闭 Stage 2 窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        msg = f"发现 {len(boxes)} 处缺陷" if has_defect else "未发现明显缺陷"
        return has_defect, msg, res_plotted