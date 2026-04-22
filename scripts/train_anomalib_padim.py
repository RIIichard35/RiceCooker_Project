import argparse
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim


def collect_normal_dirs(standards_root: Path) -> list[str]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    dirs: list[str] = []
    for child in sorted(standards_root.iterdir()):
        if not child.is_dir():
            continue
        has_images = any(p.suffix.lower() in image_exts for p in child.iterdir() if p.is_file())
        if has_images:
            dirs.append(str(child))
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomalib Padim using standards images.")
    parser.add_argument(
        "--standards-root",
        default="assets/standards",
        help="Path to standards root with view subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default="output_logs/anomalib_padim",
        help="Directory for anomalib outputs/checkpoints.",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Image resize side length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Train/val batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers on Windows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    args = parser.parse_args()

    standards_root = Path(args.standards_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not standards_root.exists():
        raise FileNotFoundError(f"Standards root not found: {standards_root}")

    normal_dirs = collect_normal_dirs(standards_root)
    if not normal_dirs:
        raise RuntimeError(f"No image folders found under: {standards_root}")

    print("[Anomalib] Normal folders:")
    for d in normal_dirs:
        print(" -", d)

    datamodule = Folder(
        name="ricecooker_stage1",
        root=None,
        normal_dir=normal_dirs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_split_mode="none",
        val_split_mode="from_train",
        val_split_ratio=0.2,
        seed=args.seed,
    )

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )

    engine = Engine(
        default_root_dir=str(output_root),
        max_epochs=1,
        accelerator="auto",
        devices=1,
        deterministic=True,
        enable_progress_bar=False,
    )

    print("[Anomalib] Start training Padim...")
    engine.fit(model=model, datamodule=datamodule)
    print("[Anomalib] Training completed.")
    print(f"[Anomalib] Outputs at: {output_root}")


if __name__ == "__main__":
    main()
