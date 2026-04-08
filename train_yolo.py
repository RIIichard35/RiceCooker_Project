from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. 确保目录正确
    # data.yaml 的路径是由 prepare_dataset.py 生成的
    yaml_path = os.path.abspath("training_workspace/data.yaml")
    
    # 检查一下配置文件在不在
    if not os.path.exists(yaml_path):
        print(f"❌ 错误：找不到文件 {yaml_path}")
        print("💡 提示：请先运行 python tools/prepare_dataset.py")
        exit()

    # 2. 加载预训练模型 (yolov8n.pt 是最轻量级的，适合笔记本跑)
    model = YOLO('yolov8n.pt') 

    print("🚀 准备开始训练...")
    
    # 3. 启动训练
    # epochs=100: 训练100轮
    # imgsz=640: 图片大小
    # batch=4: 显存小就设4，大就设8或16
    model.train(data=yaml_path, 
                epochs=100, 
                imgsz=640, 
                batch=4, 
                name='rice_cooker_defect_v1')
    
    print("✅ 训练完成！")