"""預先下載 ONNX 權重到 models/（權重不入庫）。

    python scripts/download_models.py                       # configs/default.yaml 使用的模型
    python scripts/download_models.py --mode lightweight    # CPU 原型建議先用 lightweight 確認流程
    python scripts/download_models.py --model all --mode all

網址取自已安裝的 rtmlib（YOLOX / RTMPose / RTMO，Apache-2.0），主站失敗時 rtmlib 會自動改用
Hugging Face 鏡像。下載後 src/detection/rtmlib_backend.py 會優先使用 models/ 中的檔案，可離線執行。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG, POSE_MODELS, POSE_MODES, REPO_ROOT, load_config  # noqa: E402
from src.detection.rtmlib_backend import local_model_path, model_specs  # noqa: E402


def main() -> int:
    defaults = load_config([])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=(*POSE_MODELS, "all"), default=defaults.detection.model)
    ap.add_argument("--mode", choices=(*POSE_MODES, "all"), default=defaults.detection.mode)
    ap.add_argument("--dir", type=Path, default=defaults.models_dir, help="download directory (default: models/)")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    from rtmlib.tools.file import download_checkpoint

    models = POSE_MODELS if args.model == "all" else (args.model,)
    modes = POSE_MODES if args.mode == "all" else (args.mode,)
    urls: dict[str, str] = {}
    for model in models:
        for mode in modes:
            for role, (url, _) in model_specs(model, mode).items():
                urls.setdefault(url, f"{model}/{mode}/{role}")

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"defaults from {DEFAULT_CONFIG.relative_to(REPO_ROOT)}; saving to {args.dir}")
    failed = 0
    for url, label in urls.items():
        dst = local_model_path(url, args.dir)
        if dst.exists() and not args.force:
            print(f"[skip] {label:<28} {dst.name} (already exists)")
            continue
        if dst.exists():
            dst.unlink()
        print(f"[get ] {label:<28} {url}")
        try:
            path = Path(download_checkpoint(url, dst_dir=str(args.dir)))
        except Exception as e:  # 網路錯誤時繼續下載其他模型
            print(f"[fail] {label}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        if path.resolve() != dst.resolve():
            path.replace(dst)
        print(f"[ ok ] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
