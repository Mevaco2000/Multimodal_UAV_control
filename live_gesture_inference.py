from __future__ import annotations

import argparse
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from train_lstm_classifier import LSTMClassifier, humanize_label, select_feature_indices


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LSTM_MODEL_PATH = SCRIPT_DIR / "artifacts" / "moja_wersja" / "gestures_v5_cl_arm.pt"
DEFAULT_POSE_MODEL_PATH = SCRIPT_DIR / "pose_landmarker_lite.task"
DEFAULT_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
]


def get_device(device_name: str) -> torch.device:
    normalized_name = device_name.strip().lower()
    if normalized_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA nie jest dostepna. Przelaczam model LSTM na CPU.", RuntimeWarning, stacklevel=2)
            return torch.device("cpu")
        return torch.device("cuda")
    if normalized_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lstm_model(
    model_path: Path,
    device_name: str,
) -> tuple[
    torch.nn.Module,
    torch.device,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    tuple[int, ...] | None,
]:
    device = get_device(device_name)
    checkpoint = torch.load(model_path, map_location=device)

    class_names = list(checkpoint["class_names"])
    normalization_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    normalization_std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    sequence_length = int(checkpoint["sequence_length"])
    raw_feature_indices = checkpoint.get("feature_indices")
    feature_indices = None if raw_feature_indices is None else tuple(int(index) for index in raw_feature_indices)

    model = LSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        num_classes=len(class_names),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        device,
        normalization_mean,
        normalization_std,
        class_names,
        sequence_length,
        feature_indices,
    )


def create_pose_landmarker(
    model_path: Path,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
) -> Any:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(detection_result: Any, landmark_indices: list[int]) -> np.ndarray:
    if not detection_result.pose_landmarks:
        return np.zeros((len(landmark_indices), 4), dtype=np.float32)

    landmarks = detection_result.pose_landmarks[0]
    return np.array(
        [
            [
                landmarks[index].x,
                landmarks[index].y,
                getattr(landmarks[index], "z", 0.0),
                getattr(landmarks[index], "visibility", 1.0),
            ]
            for index in landmark_indices
        ],
        dtype=np.float32,
    )


def smooth_landmarks(
    current_landmarks: np.ndarray,
    previous_landmarks: np.ndarray | None,
    smoothing_alpha: float,
) -> np.ndarray:
    if previous_landmarks is None:
        return current_landmarks
    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    return alpha * current_landmarks + (1.0 - alpha) * previous_landmarks


def apply_visibility_threshold(landmarks: np.ndarray, visibility_threshold: float) -> np.ndarray:
    filtered_landmarks = landmarks.copy()
    if visibility_threshold <= 0:
        return filtered_landmarks

    low_visibility_mask = filtered_landmarks[:, 3] < visibility_threshold
    filtered_landmarks[low_visibility_mask] = 0.0
    return filtered_landmarks


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def draw_pose_overlay(frame: np.ndarray, landmarks: np.ndarray) -> None:
    frame_height, frame_width, _ = frame.shape

    for start_index, end_index in SKELETON_CONNECTIONS:
        start_point = landmarks[start_index]
        end_point = landmarks[end_index]
        if start_point[3] <= 0.0 or end_point[3] <= 0.0:
            continue

        cv2.line(
            frame,
            (int(start_point[0] * frame_width), int(start_point[1] * frame_height)),
            (int(end_point[0] * frame_width), int(end_point[1] * frame_height)),
            (255, 180, 0),
            2,
        )

    for point in landmarks:
        if point[3] <= 0.0:
            continue
        cv2.circle(
            frame,
            (int(point[0] * frame_width), int(point[1] * frame_height)),
            4,
            (0, 255, 0),
            -1,
        )


def prepare_sequence_tensor(
    sequence: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    device: torch.device,
    feature_indices: tuple[int, ...] | None,
) -> torch.Tensor:
    selected_sequence = select_feature_indices(sequence, feature_indices)
    flattened_sequence = selected_sequence.reshape(selected_sequence.shape[0], -1).astype(np.float32)
    normalized_sequence = (flattened_sequence - normalization_mean[None, :]) / normalization_std[None, :]
    return torch.from_numpy(normalized_sequence[None, :, :]).to(device)


def predict_gesture(
    model: torch.nn.Module,
    sequence: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    class_names: list[str],
    device: torch.device,
    feature_indices: tuple[int, ...] | None,
) -> tuple[str, float, list[tuple[str, float]]]:
    input_tensor = prepare_sequence_tensor(
        sequence,
        normalization_mean,
        normalization_std,
        device,
        feature_indices,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    ranked_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [(class_names[index], float(probabilities[index])) for index in ranked_indices]
    return class_names[predicted_index], float(probabilities[predicted_index]), top_predictions


def draw_prediction_overlay(
    frame: np.ndarray,
    predicted_label: str | None,
    confidence: float | None,
    top_predictions: list[tuple[str, float]],
    buffered_frames: int,
    sequence_length: int,
    using_camera: bool,
) -> np.ndarray:
    if predicted_label is None or confidence is None:
        status_line = f"Gesture: collecting sequence... ({buffered_frames}/{sequence_length})"
    else:
        status_line = f"Gesture: {humanize_label(predicted_label)} ({confidence:.1%})"

    output_frame = cv2.copyMakeBorder(frame, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(
        output_frame,
        status_line,
        (12, frame.shape[0] + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output_frame


def run_gesture_inference(
    video_path: Path | None = None,
    camera_index: int = 0,
    model_path: Path = DEFAULT_LSTM_MODEL_PATH,
    pose_model_path: Path = DEFAULT_POSE_MODEL_PATH,
    device_name: str = "auto",
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    landmark_smoothing_alpha: float = 0.6,
    visibility_threshold: float = 0.4,
    enhance_contrast: bool = False,
    window_name: str = "Live gesture inference",
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu LSTM: {model_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu pozy MediaPipe: {pose_model_path}")

    (
        model,
        device,
        normalization_mean,
        normalization_std,
        class_names,
        sequence_length,
        feature_indices,
    ) = load_lstm_model(
        model_path=model_path,
        device_name=device_name,
    )
    sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    previous_landmarks: np.ndarray | None = None
    using_camera = video_path is None

    capture_source = camera_index if using_camera else str(video_path)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc zrodla obrazu: {capture_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 30.0

    with create_pose_landmarker(
        model_path=pose_model_path,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as landmarker:
        frame_index = 0
        camera_start_time = time.perf_counter()

        while True:
            success, frame = cap.read()
            if not success:
                break

            if using_camera:
                frame = cv2.flip(frame, 1)
            if enhance_contrast:
                frame = enhance_frame(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            if using_camera:
                timestamp_ms = int((time.perf_counter() - camera_start_time) * 1000)
            else:
                timestamp_ms = int(round(frame_index * (1000.0 / fps)))
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            landmarks = extract_landmarks(detection_result, DEFAULT_LANDMARK_INDICES)
            landmarks = smooth_landmarks(landmarks, previous_landmarks, landmark_smoothing_alpha)
            landmarks = apply_visibility_threshold(landmarks, visibility_threshold)
            previous_landmarks = landmarks.copy()
            sequence_buffer.append(landmarks)

            draw_pose_overlay(frame, landmarks)

            predicted_label = None
            confidence = None
            top_predictions: list[tuple[str, float]] = []
            if len(sequence_buffer) == sequence_length:
                predicted_label, confidence, top_predictions = predict_gesture(
                    model=model,
                    sequence=np.stack(sequence_buffer).astype(np.float32),
                    normalization_mean=normalization_mean,
                    normalization_std=normalization_std,
                    class_names=class_names,
                    device=device,
                    feature_indices=feature_indices,
                )

            output_frame = draw_prediction_overlay(
                frame=frame,
                predicted_label=predicted_label,
                confidence=confidence,
                top_predictions=top_predictions,
                buffered_frames=len(sequence_buffer),
                sequence_length=sequence_length,
                using_camera=using_camera,
            )

            cv2.imshow(window_name, output_frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in {27, ord("q")}:
                break

    cap.release()
    cv2.destroyAllWindows()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live gesture inference na bazie MediaPipe + LSTM.")
    parser.add_argument("--video", type=Path, default=None, help="Sciezka do pliku wideo. Domyslnie uzywana jest kamera.")
    parser.add_argument("--camera-index", type=int, default=0, help="Indeks kamery OpenCV, gdy nie podano --video.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LSTM_MODEL_PATH, help="Sciezka do modelu LSTM.")
    parser.add_argument("--pose-model-path", type=Path, default=DEFAULT_POSE_MODEL_PATH, help="Sciezka do modelu pozy MediaPipe (.task).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Urzadzenie dla modelu LSTM.")
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5, help="Minimalna pewnosc detekcji pozy.")
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5, help="Minimalna pewnosc obecnosci pozy.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5, help="Minimalna pewnosc trackingu pozy.")
    parser.add_argument("--landmark-smoothing-alpha", type=float, default=0.6, help="Wspolczynnik wygladzania landmarkow z zakresu 0-1.")
    parser.add_argument("--visibility-threshold", type=float, default=0.4, help="Prog widocznosci, ponizej ktorego landmark jest zerowany.")
    parser.add_argument("--enhance-contrast", action="store_true", help="Wlacz lokalne podbicie kontrastu kazdej klatki.")
    parser.add_argument("--window-name", default="Live gesture inference", help="Nazwa okna OpenCV.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_gesture_inference(
        video_path=args.video,
        camera_index=args.camera_index,
        model_path=args.model_path,
        pose_model_path=args.pose_model_path,
        device_name=args.device,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        landmark_smoothing_alpha=args.landmark_smoothing_alpha,
        visibility_threshold=args.visibility_threshold,
        enhance_contrast=args.enhance_contrast,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
