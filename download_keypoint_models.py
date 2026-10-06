from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models" / "keypoints"


@dataclass(frozen=True)
class YoloPoseModel:
    key: str
    file_name: str
    url: str


YOLO_POSE_MODELS: dict[str, YoloPoseModel] = {
    "n": YoloPoseModel(
        key="n",
        file_name="yolov8n-pose.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
    ),
    "s": YoloPoseModel(
        key="s",
        file_name="yolov8s-pose.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt",
    ),
    "m": YoloPoseModel(
        key="m",
        file_name="yolov8m-pose.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pobieranie modelu YOLO-Pose do ekstrakcji keypointow."
    )
    parser.add_argument(
        "--model",
        default="n",
        choices=sorted(YOLO_POSE_MODELS.keys()),
        help="Rozmiar modelu YOLO-Pose: n (najszybszy), s, m.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Folder docelowy dla pliku modelu.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz plik, jesli juz istnieje.",
    )
    return parser


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temp_file.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    temp_file.replace(destination)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selected_model = YOLO_POSE_MODELS[args.model]
    destination = args.output_dir / selected_model.file_name

    if destination.exists() and not args.force:
        print(f"[skip] Model juz istnieje: {destination}")
        return 0

    print(f"[download] {selected_model.file_name}")
    print(f"[source]   {selected_model.url}")
    print(f"[target]   {destination}")

    try:
        download_file(selected_model.url, destination)
    except urllib.error.URLError as exc:
        print(f"[error] Nie udalo sie pobrac modelu: {exc}")
        return 1

    print(f"[ok] Zapisano model: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
