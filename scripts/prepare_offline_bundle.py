from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare offline wheel bundle for Raspberry Pi.")
    parser.add_argument(
        "--requirements",
        default="requirements-pi.txt",
        help="Requirements file used for offline download.",
    )
    parser.add_argument(
        "--wheelhouse",
        default="offline_bundle/wheelhouse",
        help="Output wheelhouse directory.",
    )
    parser.add_argument(
        "--python-version",
        default="311",
        help="Target python version for pip download (e.g. 311).",
    )
    parser.add_argument(
        "--modnet-model",
        default="assets/models/modnet_photographic_portrait_matting.onnx",
        help="Path to local MODNet ONNX model to copy into offline bundle.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    req = (root / args.requirements).resolve()
    wheelhouse = (root / args.wheelhouse).resolve()
    bundle_root = wheelhouse.parent
    models_dir = bundle_root / "models"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not req.exists():
        raise FileNotFoundError(f"requirements file not found: {req}")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(req),
        "-d",
        str(wheelhouse),
        "--platform",
        "linux_aarch64",
        "--implementation",
        "cp",
        "--python-version",
        args.python_version,
        "--only-binary=:all:",
    ]
    print("Running:", " ".join(cmd))
    run(cmd)

    model_path = (root / args.modnet_model).resolve()
    if model_path.exists():
        dst = models_dir / model_path.name
        shutil.copy2(model_path, dst)
        print(f"Copied MODNet model: {dst}")
    else:
        print(f"WARNING: MODNet model not found, skip copy: {model_path}")

    print(f"Offline bundle ready: {wheelhouse}")
    print("Copy offline_bundle and project files to Raspberry Pi, then run install_offline.sh")


if __name__ == "__main__":
    main()
