"""
scripts/eval_anomalib.py — 阈值校准与 ROC 评估
================================================

用于在训练完成后，用实测样本评估 Anomalib 异常分，
输出 ROC 曲线并推荐最优阈值（最大化 Youden's J）。

用法：
    python scripts/eval_anomalib.py \\
        --view front \\
        --normal  path/to/normal_images/ \\
        --abnormal path/to/defect_images/ \\
        [--output eval_results/]

说明：
    --normal   : 放置正常产品图（与训练时同视角，尚未见过的验证集）
    --abnormal : 放置已知缺件/漏件产品图
    输入图片可以是原始图片或已抠图的 RGBA PNG，脚本会先运行 Stage1 抠图。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ANOMALIB_MODEL_DIR, ANOMALIB_SCORE_THRESHOLD, STANDARDS_DIR
from src.stage2_anomalib import AnomalibDetector


# ─────────────────────────────────────────────────────────────────────────
# 图像加载
# ─────────────────────────────────────────────────────────────────────────

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_images(folder: str) -> list[np.ndarray]:
    """加载文件夹下所有图片，返回 BGRA 列表（不足4通道的自动补 alpha=255）。"""
    imgs = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in VALID_EXTS:
            continue
        arr = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [警告] 无法读取: {p.name}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img   = np.dstack([img, alpha])
        imgs.append(img)
    return imgs


# ─────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────

def collect_scores(
    detector: AnomalibDetector,
    folder: str,
    label: int,  # 0=正常, 1=异常
) -> tuple[list[float], list[int]]:
    imgs = load_images(folder)
    if not imgs:
        print(f"  [警告] {folder} 下未读取到图片")
        return [], []
    scores, labels = [], []
    for img in imgs:
        r = detector.inspect(img)
        scores.append(r["anomaly_score"])
        labels.append(label)
    return scores, labels


def roc_curve_manual(
    y_true: list[int], y_score: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """手动计算 ROC 曲线（FPR, TPR, thresholds），避免 sklearn 依赖。"""
    y_true  = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs, fprs = [], []
    P = int(y_true.sum())
    N = int(len(y_true) - P)
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / max(P, 1))
        fprs.append(fp / max(N, 1))
    return np.array(fprs), np.array(tprs), thresholds


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def find_best_threshold(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float, float]:
    """Youden's J = TPR - FPR，最大化处为最优阈值。"""
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


# ─────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────

def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_val: float,
    best_thr: float,
    best_tpr: float,
    best_fpr: float,
    current_thr: float,
    view: str,
    out_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过绘图] 未安装 matplotlib，请执行: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"PatchCore ROC (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(
        best_fpr, best_tpr, s=120, zorder=5,
        color="red", label=f"最优阈值 {best_thr:.3f}（TPR={best_tpr:.2f}, FPR={best_fpr:.2f}）",
    )
    ax.axvline(x=best_fpr, color="red",   linestyle="--", lw=0.8, alpha=0.5)
    ax.axhline(y=best_tpr, color="red",   linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR / 召回率)")
    ax.set_title(f"Anomalib PatchCore — 视角: {view}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"roc_{view}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC 图已保存: {save_path}")


def plot_score_dist(
    normal_scores: list[float],
    abnormal_scores: list[float],
    best_thr: float,
    view: str,
    out_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_scores,   bins=15, alpha=0.7, label="正常品",   color="steelblue")
    ax.hist(abnormal_scores, bins=15, alpha=0.7, label="缺件/漏件", color="tomato")
    ax.axvline(best_thr, color="black", linestyle="--", lw=1.5,
               label=f"最优阈值 {best_thr:.3f}")
    ax.set_xlabel("异常分 (Anomaly Score)")
    ax.set_ylabel("图片数")
    ax.set_title(f"异常分分布 — 视角: {view}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"score_dist_{view}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  分布图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Anomalib 阈值评估 & ROC 曲线")
    ap.add_argument("--view",     required=True, help="视角（front/back/left/right/top）")
    ap.add_argument("--normal",   required=True, help="正常品图片文件夹")
    ap.add_argument("--abnormal", required=True, help="缺件/漏件图片文件夹")
    ap.add_argument("--output",   default="eval_results", help="结果输出目录")
    args = ap.parse_args()

    print(f"\n{'='*56}")
    print(f"  评估视角: {args.view}")
    print(f"  当前配置阈值: {ANOMALIB_SCORE_THRESHOLD}")
    print(f"{'='*56}")

    det = AnomalibDetector(view=args.view, model_dir=ANOMALIB_MODEL_DIR)
    if not det.is_ready:
        print("[错误] 模型未就绪，请先运行 scripts/train_anomalib.py 训练。")
        sys.exit(1)

    print(f"\n加载正常品图片（{args.normal}）…")
    normal_scores,   normal_labels   = collect_scores(det, args.normal,   label=0)
    print(f"加载缺件/漏件图片（{args.abnormal}）…")
    abnormal_scores, abnormal_labels = collect_scores(det, args.abnormal, label=1)

    if not normal_scores or not abnormal_scores:
        print("[错误] 正常或异常样本为空，请检查路径。")
        sys.exit(1)

    all_scores = normal_scores + abnormal_scores
    all_labels = normal_labels + abnormal_labels

    fpr, tpr, thresholds = roc_curve_manual(all_labels, all_scores)
    auc_val = auc_trapz(fpr, tpr)
    best_thr, best_tpr, best_fpr = find_best_threshold(fpr, tpr, thresholds)

    print(f"\n{'─'*40}")
    print(f"  AUC          : {auc_val:.4f}")
    print(f"  最优阈值      : {best_thr:.4f}  (Youden's J 最大化)")
    print(f"  最优 TPR      : {best_tpr:.4f}")
    print(f"  最优 FPR      : {best_fpr:.4f}")
    print(f"  正常品均值    : {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
    print(f"  异常品均值    : {np.mean(abnormal_scores):.4f} ± {np.std(abnormal_scores):.4f}")
    print(f"{'─'*40}")
    print(f"\n  建议将 config.py 中 ANOMALIB_SCORE_THRESHOLD 设置为: {best_thr:.3f}")

    plot_roc(fpr, tpr, auc_val, best_thr, best_tpr, best_fpr,
             ANOMALIB_SCORE_THRESHOLD, args.view, args.output)
    plot_score_dist(normal_scores, abnormal_scores, best_thr, args.view, args.output)

    # 保存数值结果
    out_txt = os.path.join(args.output, f"eval_{args.view}.txt")
    os.makedirs(args.output, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"view={args.view}\n")
        f.write(f"auc={auc_val:.6f}\n")
        f.write(f"best_threshold={best_thr:.6f}\n")
        f.write(f"best_tpr={best_tpr:.6f}\n")
        f.write(f"best_fpr={best_fpr:.6f}\n")
        f.write(f"normal_mean={np.mean(normal_scores):.6f}\n")
        f.write(f"abnormal_mean={np.mean(abnormal_scores):.6f}\n")
        f.write("normal_scores=" + ",".join(f"{s:.6f}" for s in normal_scores) + "\n")
        f.write("abnormal_scores=" + ",".join(f"{s:.6f}" for s in abnormal_scores) + "\n")
    print(f"  数值结果已保存: {out_txt}")


if __name__ == "__main__":
    main()
