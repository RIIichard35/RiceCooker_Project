import os
import cv2
import csv
import time
from datetime import datetime
from verify_stage1 import run_stage1
from verify_stage2 import Stage2Detector

def determine_ref_model(filename):
    """根据文件名动态推断需要使用的金样文件夹"""
    name_lower = filename.lower()
    if "front" in name_lower:
        return "front"
    elif "top" in name_lower:
        return "top"
    elif "right" in name_lower:
        return "right"
    elif "left" in name_lower:
        return "left"
    else:
        return None  # 无法识别特征词

def main():
    # 1. 基础路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "assets", "images_test")
    model_path = os.path.join(base_dir, "assets", "models", "best.pt")

    # --- 优化点 1: 动态创建带时间戳的输出文件夹，防止覆盖历史数据 ---
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "runs", f"batch_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    report_csv_path = os.path.join(output_dir, "inspection_report.csv")

    print("="*60)
    print(f"🚀 启动自动化批量检测流水线 | 批次: {current_time}")
    print("="*60)

    # 2. 获取所有待测图片
    if not os.path.exists(test_dir):
        print(f"❌ 找不到测试图文件夹: {test_dir}")
        return
        
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"⚠️ 文件夹 {test_dir} 中没有找到图片！")
        return

    # 3. 初始化 YOLO 模型
    try:
        yolo_detector = Stage2Detector(model_path)
    except Exception as e:
        print(f"\n[致命错误] 模型加载失败: {e}")
        return

    # 4. 统计数据与报告列表
    report_data = []
    stats = {"total": len(image_files), "ok": 0, "ng_stage1": 0, "ng_stage2": 0, "unrecognized": 0}
    start_time = time.time()

    # ------------------ 批量流水线开始 ------------------
    for idx, filename in enumerate(image_files, 1):
        test_image_path = os.path.join(test_dir, filename)
        print(f"\n[{idx}/{stats['total']}] 正在检测: {filename} ...")

        # 初始化单条数据记录（新增 Model_Type 字段）
        row_data = {
            "Filename": filename,
            "Model_Type": "Unknown",
            "Stage1_Result": "未执行",
            "Stage2_Result": "未执行",
            "Final_Decision": "NG",
            "Details": ""
        }

        # --- 优化点 2: 动态匹配型号 ---
        ref_subfolder_name = determine_ref_model(filename)
        if not ref_subfolder_name:
            print(f"  ├─ ⚠️ [跳过] 无法从文件名识别型号 (front/top/right/left)")
            row_data["Details"] = "文件名未包含型号特征词"
            stats["unrecognized"] += 1
            report_data.append(row_data)
            continue

        row_data["Model_Type"] = ref_subfolder_name
        ref_folder = os.path.join(base_dir, "assets", "images_ref", ref_subfolder_name)

        # [步骤 1] 运行防呆验证
        is_passed, msg1, img1 = run_stage1(test_image_path, ref_folder, debug=False)
        final_img = None
        
        if not is_passed:
            print(f"  ├─ ❌ [Stage 1 拦截] {msg1}")
            row_data["Stage1_Result"] = "Fail"
            row_data["Final_Decision"] = "NG"
            row_data["Details"] = f"防呆失败: {msg1}"
            stats["ng_stage1"] += 1
            final_img = img1
        else:
            print(f"  ├─ ✅ [Stage 1 通过] 型号 ({ref_subfolder_name}) 匹配成功")
            row_data["Stage1_Result"] = "Pass"
            
            # [步骤 2] 运行缺陷检测
            has_defect, msg2, img2 = yolo_detector.detect(test_image_path, conf_threshold=0.25, debug=False)
            
            if has_defect:
                print(f"  ├─ ⚠️ [Stage 2 警告] {msg2}")
                row_data["Stage2_Result"] = "Fail (Defect Found)"
                row_data["Final_Decision"] = "NG"
                row_data["Details"] = msg2
                stats["ng_stage2"] += 1
                final_img = img2
            else:
                print(f"  ├─ ✅ [Stage 2 通过] {msg2}")
                row_data["Stage2_Result"] = "Pass"
                row_data["Final_Decision"] = "OK"
                row_data["Details"] = "良品"
                stats["ok"] += 1

        # --- 优化点 3: 只保存 NG 图片，提升 I/O 速度 ---
        if row_data["Final_Decision"] == "NG" and final_img is not None:
            save_path = os.path.join(output_dir, f"NG_{filename}")
            cv2.imwrite(save_path, final_img)
            
        report_data.append(row_data)

    # ------------------ 生成测试报告 ------------------
    print("\n" + "="*60)
    print("📊 批量检测完成，正在生成报告...")
    with open(report_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ["Filename", "Model_Type", "Stage1_Result", "Stage2_Result", "Final_Decision", "Details"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in report_data:
            writer.writerow(data)
            
    total_time = time.time() - start_time
    fps = stats["total"] / total_time if total_time > 0 else 0

    print(f"📁 本次测试批次文件夹: {output_dir}")
    print("📌 [最终统计 summary]")
    print(f"  - 总图片数: {stats['total']} (无法识别跳过: {stats['unrecognized']})")
    print(f"  - 🟢 良品 (OK): {stats['ok']}")
    print(f"  - 🔴 防呆拦截 (NG): {stats['ng_stage1']}")
    print(f"  - 🔴 缺陷拦截 (NG): {stats['ng_stage2']}")
    
    valid_total = stats['total'] - stats['unrecognized']
    if valid_total > 0:
        yield_rate = (stats['ok'] / valid_total) * 100
        print(f"  - 良率 (Yield): {yield_rate:.2f}%")
        
    print(f"  - 平均速度: {fps:.2f} 帧/秒 (总耗时 {total_time:.2f}s)")
    print("="*60)

if __name__ == "__main__":
    main()