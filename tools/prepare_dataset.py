import os
import shutil
import random
import yaml

# ================= 配置区 =================
# 1. 你的原始图片在哪里？
SOURCE_IMG_DIR = r"assets/images_test"  # 或者你存放几百张训练图的绝对路径
# 2. LabelImg 生成的 xml/txt 标签在哪里？
SOURCE_LABEL_DIR = r"assets/labels_raw" # 你刚才保存标注的地方
# 3. 准备把整理好的数据放在哪？
OUTPUT_DIR = r"training_workspace"
# =========================================

def setup_yolo_structure():
    # 定义 YOLO 目录结构
    dirs = [
        f"{OUTPUT_DIR}/images/train",
        f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train",
        f"{OUTPUT_DIR}/labels/val"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ 创建目录: {d}")

    # 获取所有图片
    images = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    # 划分 80% 训练，20% 验证
    split_index = int(len(images) * 0.8)
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]
    
    print(f"📊 数据集划分: 训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张")

    def move_files(file_list, type_name):
        for img_name in file_list:
            base_name = os.path.splitext(img_name)[0]
            
            # 1. 复制图片
            src_img = os.path.join(SOURCE_IMG_DIR, img_name)
            dst_img = os.path.join(OUTPUT_DIR, "images", type_name, img_name)
            shutil.copy(src_img, dst_img)
            
            # 2. 复制对应的 txt 标签 (LabelImg生成的是 txt, 如果是xml需要转换，这里假设你选了YOLO格式)
            # 注意：在 LabelImg 里保存时，左侧按钮要选 "PascalVOC" -> 切换为 "YOLO"
            txt_name = base_name + ".txt"
            src_txt = os.path.join(SOURCE_LABEL_DIR, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, "labels", type_name, txt_name)
            
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            else:
                print(f"⚠️ 警告: 图片 {img_name} 没有对应的标注文件！")

    move_files(train_imgs, "train")
    move_files(val_imgs, "val")
    
    # 生成 data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['scratch', 'dent', 'plastic_film'] # 必须和你标注时的顺序一致！
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"🎉 数据集准备完毕！配置文件位于: {OUTPUT_DIR}/data.yaml")

if __name__ == "__main__":
    setup_yolo_structure()