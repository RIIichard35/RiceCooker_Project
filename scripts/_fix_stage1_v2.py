"""修复 stage1_vision.py：annotated_image 改为产品裁剪图（黄框围整图）"""
path = 'src/stage1_vision.py'
lines = open(path, encoding='utf-8').readlines()

# 找 J 区块（annotated_image 赋值区块）
start = end = None
for i, l in enumerate(lines):
    if 'J. annotated_image' in l:
        start = i
    if start is not None and 'result["product_bbox"]' in l:
        end = i + 1
        break

print(f'J 区块: 行 {start+1} ~ {end}')

NEW_J = '''\
        # ── J. annotated_image：裁剪出产品区域，黄框围整个产品 ──────────
        # 直接裁剪到 bbox，产品充满画面，不再显示整幅灰色画布。
        # GUI 层叠加 Anomalib 异常框（坐标已对齐到裁剪图）。
        cutout_bgr_full = self._cutout_to_bgr_for_annotation(transparent_cutout)
        product_crop = cutout_bgr_full[y1:y2, x1:x2].copy()
        if product_crop.size == 0:
            product_crop = cutout_bgr_full.copy()
        ch, cw = product_crop.shape[:2]
        # 黄框围住整个裁剪图（即围住整个产品）
        cv2.rectangle(product_crop, (2, 2), (cw - 3, ch - 3), (0, 215, 255), 3)
        cv2.putText(
            product_crop, f"sim={result[\'similarity\']:.2f}",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2,
        )
        if pose_angle > 1.0:
            cv2.putText(
                product_crop, f"Skew:{pose_angle:.1f}deg",
                (cw - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 100, 255) if pose_angle >= POSE_SKEW_WARN_DEG else (180, 180, 0), 2,
            )
        result["annotated_image"] = product_crop
        result["product_bbox"] = [x1, y1, x2, y2]
        return result
'''

new_lines = lines[:start] + [NEW_J] + lines[end:]
open(path, 'w', encoding='utf-8').writelines(new_lines)
print(f'Done. Total lines: {len(new_lines)}')
