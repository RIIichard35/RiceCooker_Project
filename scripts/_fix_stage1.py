"""临时修复脚本：重写 stage1_vision.py 中 H/I/J 区块"""
path = 'src/stage1_vision.py'
lines = open(path, encoding='utf-8').readlines()

start = end = None
for i, l in enumerate(lines):
    if 'vis_crop = vis[y1:y2' in l:
        start = i - 1   # 包含上一行注释
    if 'result["annotated_image"] = vis_crop' in l:
        end = i + 1     # 包含该行

print(f'替换范围: 行 {start+1} ~ {end}')

NEW_BLOCK = '''\
        # ── H. 框坐标映射（只记录坐标，不画框）──────────────────────────
        def map_to_full(bx1, by1, bx2, by2):
            rx1 = x1 + int(bx1 * (x2 - x1) / max(w_ref, 1))
            ry1 = y1 + int(by1 * (y2 - y1) / max(h_ref, 1))
            rx2 = x1 + int(bx2 * (x2 - x1) / max(w_ref, 1))
            ry2 = y1 + int(by2 * (y2 - y1) / max(h_ref, 1))
            rx1, ry1, rx2, ry2 = self._clamp(rx1, ry1, rx2, ry2, w_img, h_img)
            if (rx2 - rx1) * (ry2 - ry1) < 50:
                return None
            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            if product_mask[cy, cx] == 0:
                return None
            return [rx1, ry1, rx2, ry2]

        for box in missing_boxes_roi:
            mapped = map_to_full(*box)
            if mapped is not None:
                result["missing_regions"].append(mapped)

        for box in extra_boxes_roi:
            mapped = map_to_full(*box)
            if mapped is not None:
                result["extra_regions"].append(mapped)

        # ── I. 综合判定 ────────────────────────────────────────────────
        if len(result["missing_regions"]) >= missing_thresh:
            result["issues"].append(
                f"发现 {len(result[\'missing_regions\'])} 处缺件区域（阈值={missing_thresh}）"
            )
        if len(result["extra_regions"]) > 0:
            result["issues"].append(
                f"发现 {len(result[\'extra_regions\'])} 处疑似多余区域"
            )
        if result["similarity"] < min_similarity_fail:
            result["warnings"].append(
                f"与标准件相似度偏低 ({result[\'similarity\']:.2f} < {min_similarity_fail})"
            )

        result["pass"]   = len(result["issues"]) == 0
        result["status"] = "PASS" if result["pass"] else "FAIL"

        # ── J. annotated_image：抠图底图 + 黄色产品大框 ──────────────
        # 以抠图（灰色背景）为底图，画产品定位框（黄色）。
        # 缺件/异常红框由 GUI 层叠加 Anomalib 结果后统一绘制，不在此处理。
        ann = self._cutout_to_bgr_for_annotation(transparent_cutout)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 215, 255), 3)
        cv2.putText(
            ann, f"Product  sim={result[\'similarity\']:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2,
        )
        if pose_angle > 1.0:
            cv2.putText(
                ann, f"Skew:{pose_angle:.1f}deg",
                (x2 - 130, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 100, 255) if pose_angle >= POSE_SKEW_WARN_DEG else (180, 180, 0),
                2,
            )
        result["annotated_image"] = ann
        result["product_bbox"] = [x1, y1, x2, y2]
        return result
'''

new_lines = lines[:start] + [NEW_BLOCK] + lines[end:]
open(path, 'w', encoding='utf-8').writelines(new_lines)
print('Done. Total lines:', len(new_lines))
