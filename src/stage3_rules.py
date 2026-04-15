from .config import MIN_DEFECT_SIZE_MM, PIXELS_PER_MM


class Stage3RuleEngine:
    """Business rules applied after ROI-only YOLO inference."""

    def __init__(self, valid_defect_classes=None, film_class="plastic_film"):
        self.valid_defect_classes = tuple(valid_defect_classes or ("scratch", "dent"))
        self.film_class = film_class

    def calculate_iou_contains(self, defect_box, film_box):
        """Return True when most of the defect box is covered by film."""
        dx1, dy1, dx2, dy2 = defect_box
        fx1, fy1, fx2, fy2 = film_box

        ix1 = max(dx1, fx1)
        iy1 = max(dy1, fy1)
        ix2 = min(dx2, fx2)
        iy2 = min(dy2, fy2)
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        defect_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)

        if defect_area <= 0:
            return False
        return (inter_area / defect_area) > 0.6

    def apply_rules(self, detections):
        films = [d["box"] for d in detections if d["cls"] == self.film_class]

        processed_detections = []
        valid_defect_count = 0

        for item in detections:
            d = item.copy()
            d["status"] = "NG"
            d["color"] = (255, 0, 0)
            d["display_label"] = d["cls"]

            if d["cls"] not in self.valid_defect_classes:
                d["status"] = "INFO"
                d["color"] = (255, 255, 0)
                if d["cls"] == self.film_class:
                    d["display_label"] = "Film (Ignored)"
                else:
                    d["display_label"] = f"{d['cls']} (Info)"
                processed_detections.append(d)
                continue

            is_in_film = any(self.calculate_iou_contains(d["box"], film_box) for film_box in films)
            if is_in_film:
                d["status"] = "IGNORED"
                d["display_label"] = f"{d['cls']} (In Film)"
                d["color"] = (255, 165, 0)
                processed_detections.append(d)
                continue

            w = max(0.0, d["box"][2] - d["box"][0])
            h = max(0.0, d["box"][3] - d["box"][1])
            size_mm = max(w, h) / float(PIXELS_PER_MM)
            d["size_mm"] = size_mm

            if size_mm < float(MIN_DEFECT_SIZE_MM):
                d["status"] = "IGNORED"
                d["display_label"] = f"{d['cls']} (<{MIN_DEFECT_SIZE_MM}mm)"
                d["color"] = (0, 255, 0)
                processed_detections.append(d)
                continue

            valid_defect_count += 1
            d["display_label"] = f"{d['cls']} {d['conf']:.2f}"
            processed_detections.append(d)

        if valid_defect_count > 0:
            final_verdict = "FAIL"
            final_desc = f"Detected {valid_defect_count} valid defect(s) inside product ROI"
        else:
            final_verdict = "PASS"
            final_desc = "No valid defect inside product ROI"

        return final_verdict, processed_detections, final_desc
