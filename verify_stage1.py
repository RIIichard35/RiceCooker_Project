import os
import cv2
from src.stage1_vision import Stage1Detector

def run_stage1(test_img_path, ref_folder_path, debug=False):
    """
    运行第一阶段检测 (模板匹配/防呆)
    :param test_img_path: 待测图片的完整路径
    :param ref_folder_path: 该型号对应的金样文件夹路径
    :param debug: 是否显示弹窗
    :return: (is_passed, msg, result_img)
    """
    if not os.path.exists(ref_folder_path):
        return False, f"找不到标准样文件夹: {ref_folder_path}", None

    if not os.path.exists(test_img_path):
        return False, f"找不到待测图片: {test_img_path}", None

    try:
        detector = Stage1Detector(ref_folder_path)
        is_passed, msg, result_img = detector.check_integrity(test_img_path)
        
        # 仅在调试模式下才弹窗，避免在自动化流水线中卡死程序
        if debug and result_img is not None:
            h, w = result_img.shape[:2]
            scale = 0.6
            resized_img = cv2.resize(result_img, (int(w*scale), int(h*scale)))
            cv2.imshow(f"Stage 1 Match Result", resized_img)
            print("💡 提示: 按键盘任意键关闭 Stage 1 窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return is_passed, msg, result_img
        
    except Exception as e:
        return False, f"Stage 1 初始化或检测异常: {e}", None