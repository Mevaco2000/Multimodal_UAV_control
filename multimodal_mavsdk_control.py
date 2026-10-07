from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty as ThreadQueueEmpty
from queue import Queue

import cv2
import mediapipe as mp
import numpy as np
from mavsdk import System
try:
    from mavsdk.plugins.offboard import OffboardError, VelocityBodyYawspeed
except ImportError:
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

SCRIPT_DIR = Path(__file__).resolve().parent
VOICE_MODULE_PATH = SCRIPT_DIR / "realtime_predict.py"

from live_gesture_inference import (
    DEFAULT_LANDMARK_INDICES,
    apply_visibility_threshold,
    create_pose_landmarker,
    draw_pose_overlay,
    draw_prediction_overlay,
    enhance_frame,
    extract_landmarks,
    load_lstm_model,
    predict_gesture,
    smooth_landmarks,
)


def load_voice_predictor_module():
    spec = importlib.util.spec_from_file_location("voice_realtime_predict", VOICE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie mozna zaladowac modulu glosowego z: {VOICE_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


voice_predictor = load_voice_predictor_module()


DEFAULT_GESTURE_MODEL_PATH = SCRIPT_DIR / "gestures_v5_cl_arm.pt"
DEFAULT_POSE_MODEL_PATH = SCRIPT_DIR / "pose_landmarker_lite.task"
DEFAULT_CONNECTION_URL = "udpin://127.0.0.1:14540"


@dataclass(frozen=True)
class RuntimeConfig:
    connection_url: str = DEFAULT_CONNECTION_URL
    gesture_model_path: Path = DEFAULT_GESTURE_MODEL_PATH
    pose_model_path: Path = DEFAULT_POSE_MODEL_PATH
    camera_index: int = 0
    device: str = "auto"
    voice_confidence_threshold: float = 0.55
    gesture_confidence_threshold: float = 0.85
    voice_repeat_count: int = 2
    gesture_repeat_count: int = 5
    voice_priority_window: float = 2.0
    linear_speed: float = 0.2
    vertical_speed: float = 0.12
    yaw_rate: float = 6.0
    slow_down_factor: float = 0.8
    min_linear_speed: float = 0.2
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    landmark_smoothing_alpha: float = 0.6
    visibility_threshold: float = 0.4
    enhance_contrast: bool = False
    hide_window: bool = False


CONFIG = RuntimeConfig()


@dataclass(frozen=True)
class CommandEvent:
    source: str
    label: str
    confidence: float


class CommandStatus:
    def __init__(self) -> None:
        self._label = "hover"
        self._source = "system"
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def set(self, label: str, source: str) -> None:
        with self._lock:
            self._label = label
            self._source = source
            self._updated_at = time.monotonic()

    def get(self) -> tuple[str, str, float]:
        with self._lock:
            return self._label, self._source, self._updated_at


class ControlMode:
    def __init__(self) -> None:
        self._gesture_enabled = False
        self._voice_enabled = False
        self._lock = threading.Lock()

    def toggle_for_source(self, source: str) -> tuple[bool, bool]:
        with self._lock:
            if source == "gesture":
                self._gesture_enabled = not self._gesture_enabled
            elif source == "voice":
                self._voice_enabled = not self._voice_enabled
            return self._gesture_enabled, self._voice_enabled

    def is_enabled(self, source: str) -> bool:
        with self._lock:
            if source == "gesture":
                return self._gesture_enabled
            if source == "voice":
                return self._voice_enabled
            return True

    def snapshot(self) -> tuple[bool, bool]:
        with self._lock:
            return self._gesture_enabled, self._voice_enabled


class InputHistory:
    def __init__(self) -> None:
        self._gesture_labels: deque[str] = deque(maxlen=5)
        self._voice_labels: deque[str] = deque(maxlen=2)
        self._lock = threading.Lock()

    def add(self, source: str, label: str) -> None:
        with self._lock:
            if source == "gesture":
                self._gesture_labels.append(label)
            elif source == "voice":
                self._voice_labels.append(label)

    def snapshot(self) -> tuple[list[str], list[str]]:
        with self._lock:
            return list(self._gesture_labels), list(self._voice_labels)


def humanize_command_label(label: str) -> str:
    return label.replace("_", " ")


def draw_active_command_overlay(
    frame: np.ndarray,
    command_status: CommandStatus,
    control_mode: ControlMode,
    input_history: InputHistory,
) -> np.ndarray:
    active_label, active_source, _ = command_status.get()
    display_label = humanize_command_label(active_label)
    gesture_enabled, voice_enabled = control_mode.snapshot()
    gesture_history, voice_history = input_history.snapshot()
    status_line = f"Active command: {display_label} [{active_source}]"
    mode_line = (
        f"Control modes: gesture={'on' if gesture_enabled else 'off'} | "
        f"voice={'on' if voice_enabled else 'off'}"
    )
    gesture_line = "Last 5 gestures: " + (
        " -> ".join(humanize_command_label(label) for label in gesture_history)
        if gesture_history
        else "-"
    )
    voice_line = "Last 2 voice: " + (
        " -> ".join(humanize_command_label(label) for label in voice_history)
        if voice_history
        else "-"
    )

    output_frame = cv2.copyMakeBorder(frame, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(
        output_frame,
        status_line,
        (12, frame.shape[0] + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (90, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        mode_line,
        (12, frame.shape[0] + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 240, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        gesture_line,
        (12, frame.shape[0] + 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 210, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        voice_line,
        (12, frame.shape[0] + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return output_frame


class ConfirmationGate:
    def __init__(self, required_repeats: int, confidence_threshold: float) -> None:
        self.required_repeats = max(1, required_repeats)
        self.confidence_threshold = confidence_threshold
        self.last_label: str | None = None
        self.repeat_count = 0
        self.locked_label: str | None = None
        self._lock = threading.Lock()

    def ingest(self, label: str | None, confidence: float | None) -> str | None:
        with self._lock:
            if label is None or confidence is None or confidence < self.confidence_threshold:
                self.last_label = None
                self.repeat_count = 0
                self.locked_label = None
                return None

            if label != self.last_label:
                self.last_label = label
                self.repeat_count = 1
                self.locked_label = None
            else:
                self.repeat_count += 1

            if self.locked_label == label:
                return None

            if self.repeat_count >= self.required_repeats:
                self.locked_label = label
                return label

            return None


class DroneController:
    def __init__(
        self,
        connection_url: str,
        linear_speed: float,
        vertical_speed: float,
        yaw_rate_deg_s: float,
        slow_down_factor: float,
        min_linear_speed: float,
    ) -> None:
        self.drone = None
        self._native_mode = False
        self._native_mavsdk = None
        self._native_system = None
        self._native_action = None
        self._native_offboard = None
        self._native_connection_handle = None
        try:
            self.drone = System()
        except TypeError:
            from mavsdk.asyncio import ComponentType, Configuration, Mavsdk

            self._native_mode = True
            configuration = Configuration.create_with_component_type(ComponentType.GROUND_STATION)
            self._native_mavsdk = Mavsdk(configuration)
        self.connection_url = connection_url
        self.linear_speed = linear_speed
        self.vertical_speed = vertical_speed
        self.yaw_rate_deg_s = yaw_rate_deg_s
        self.slow_down_factor = slow_down_factor
        self.min_linear_speed = min_linear_speed
        self.current_linear_speed = linear_speed
        self.offboard_started = False
        self.active_motion = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

    @property
    def action(self):
        if self._native_mode:
            if self._native_action is None:
                raise RuntimeError("Plugin Action nie jest gotowy. Wywolaj najpierw connect().")
            return self._native_action
        if self.drone is None:
            raise RuntimeError("System MAVSDK nie zostal zainicjalizowany.")
        return self.drone.action

    @property
    def offboard(self):
        if self._native_mode:
            if self._native_offboard is None:
                raise RuntimeError("Plugin Offboard nie jest gotowy. Wywolaj najpierw connect().")
            return self._native_offboard
        if self.drone is None:
            raise RuntimeError("System MAVSDK nie zostal zainicjalizowany.")
        return self.drone.offboard

    async def connect(self) -> None:
        print(f"[mavsdk] Laczenie z {self.connection_url}...")
        if self._native_mode:
            from mavsdk.asyncio.plugins.action import ActionAsync
            from mavsdk.asyncio.plugins.offboard import OffboardAsync

            if self._native_mavsdk is None:
                raise RuntimeError("Brak instancji natywnego klienta MAVSDK.")

            self._native_connection_handle = await self._native_mavsdk.add_any_connection(self.connection_url)
            self._native_system = await self._native_mavsdk.first_autopilot(timeout_s=10.0)
            if self._native_system is None:
                raise RuntimeError(
                    f"Nie wykryto autopilota dla polaczenia: {self.connection_url}. "
                    "Sprawdz URL i czy endpoint MAVSDK/SITL dziala."
                )

            self._native_action = ActionAsync(self._native_system)
            self._native_offboard = OffboardAsync(self._native_system)
            print("[mavsdk] Polaczono z systemem (native MAVSDK v4).")
            return

        if self.drone is None:
            raise RuntimeError("System MAVSDK nie zostal zainicjalizowany.")

        await self.drone.connect(system_address=self.connection_url)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[mavsdk] Polaczono z systemem.")
                return

    async def stop(self) -> None:
        if self.offboard_started:
            try:
                await self.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass

            try:
                await self.offboard.stop()
            except OffboardError:
                pass
            self.offboard_started = False

    async def shutdown(self) -> None:
        if not self._native_mode or self._native_mavsdk is None:
            return

        if self._native_connection_handle is not None:
            try:
                await self._native_mavsdk.remove_connection(self._native_connection_handle)
            except Exception:
                pass
            self._native_connection_handle = None

        self._native_mavsdk.destroy()
        self._native_mavsdk = None
        self._native_system = None
        self._native_action = None
        self._native_offboard = None

    async def ensure_offboard(self) -> None:
        if self.offboard_started:
            return

        await self.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        try:
            await self.offboard.start()
        except OffboardError as exc:
            raise RuntimeError(f"Nie udalo sie uruchomic offboard: {exc}") from exc
        self.offboard_started = True

    async def hover(self) -> None:
        await self.ensure_offboard()
        self.active_motion = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        await self.offboard.set_velocity_body(self.active_motion)

    async def set_continuous_velocity(
        self,
        forward_m_s: float = 0.0,
        right_m_s: float = 0.0,
        down_m_s: float = 0.0,
        yawspeed_deg_s: float = 0.0,
    ) -> None:
        await self.ensure_offboard()
        self.active_motion = VelocityBodyYawspeed(
            forward_m_s,
            right_m_s,
            down_m_s,
            yawspeed_deg_s,
        )
        await self.offboard.set_velocity_body(self.active_motion)

    async def refresh_active_motion(self) -> None:
        motion = self.active_motion
        if abs(motion.forward_m_s) > 0.0:
            motion = VelocityBodyYawspeed(
                self.current_linear_speed if motion.forward_m_s > 0.0 else -self.current_linear_speed,
                motion.right_m_s,
                motion.down_m_s,
                motion.yawspeed_deg_s,
            )
        if abs(motion.right_m_s) > 0.0:
            motion = VelocityBodyYawspeed(
                motion.forward_m_s,
                self.current_linear_speed if motion.right_m_s > 0.0 else -self.current_linear_speed,
                motion.down_m_s,
                motion.yawspeed_deg_s,
            )

        self.active_motion = motion
        if self.offboard_started:
            await self.offboard.set_velocity_body(self.active_motion)

    async def execute_command(self, label: str, source: str) -> None:
        print(f"[cmd] {source}: {label}")
        try:
            if label == "arm":
                await self.action.arm()
                return

            if label == "disarm":
                await self.stop()
                await self.action.disarm()
                return

            if label == "dispatch":
                await self.hover()
                return

            if label == "land":
                await self.stop()
                await self.action.land()
                return

            if label == "hover":
                await self.hover()
                return

            if label == "slow_down":
                self.current_linear_speed = max(
                    self.min_linear_speed,
                    self.current_linear_speed * self.slow_down_factor,
                )
                print(f"[cmd] Nowa predkosc liniowa: {self.current_linear_speed:.2f} m/s")
                await self.refresh_active_motion()
                return

            if label == "move_ahead":
                await self.set_continuous_velocity(forward_m_s=self.current_linear_speed)
                return

            if label == "backward":
                await self.set_continuous_velocity(forward_m_s=-self.current_linear_speed)
                return

            if label == "move_left":
                await self.set_continuous_velocity(right_m_s=-self.current_linear_speed)
                return

            if label == "move_right":
                await self.set_continuous_velocity(right_m_s=self.current_linear_speed)
                return

            if label == "move_upward":
                await self.set_continuous_velocity(down_m_s=-self.vertical_speed)
                return

            if label == "move_down":
                await self.set_continuous_velocity(down_m_s=self.vertical_speed)
                return

            if label == "turn_left":
                await self.set_continuous_velocity(yawspeed_deg_s=-self.yaw_rate_deg_s)
                return

            if label == "turn_right":
                await self.set_continuous_velocity(yawspeed_deg_s=self.yaw_rate_deg_s)
                return

            print(f"[cmd] Pomijam nieobslugiwana komende: {label}")
        except Exception as exc:
            print(f"[cmd] Blad podczas wykonywania '{label}': {exc}")


class VoiceRecognizer(threading.Thread):
    def __init__(self, stop_event: threading.Event, on_prediction) -> None:
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.on_prediction = on_prediction
        self.pre_roll: list[np.ndarray] = []
        self.recording: list[np.ndarray] = []
        self.silence_count = 0
        self.is_recording = False
        self.lock = threading.Lock()

        self.pre_roll_blocks = max(
            1,
            int(
                voice_predictor.PRE_ROLL_MS
                * voice_predictor.SAMPLE_RATE
                / 1000
                / voice_predictor.BLOCKSIZE
            ),
        )
        self.silence_blocks = max(
            1,
            int(
                voice_predictor.SILENCE_AFTER_MS
                * voice_predictor.SAMPLE_RATE
                / 1000
                / voice_predictor.BLOCKSIZE
            ),
        )
        self.max_blocks = int(
            voice_predictor.MAX_RECORD_MS
            * voice_predictor.SAMPLE_RATE
            / 1000
            / voice_predictor.BLOCKSIZE
        )

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[voice] status: {status}")

        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk**2)))

        with self.lock:
            if not self.is_recording:
                self.pre_roll.append(chunk)
                if len(self.pre_roll) > self.pre_roll_blocks:
                    self.pre_roll.pop(0)

                if rms > voice_predictor.VAD_THRESHOLD:
                    self.is_recording = True
                    self.silence_count = 0
                    self.recording.clear()
                    self.recording.extend(self.pre_roll)
                    self.recording.append(chunk)
            else:
                self.recording.append(chunk)
                if rms < voice_predictor.VAD_THRESHOLD:
                    self.silence_count += 1
                else:
                    self.silence_count = 0

                if self.silence_count >= self.silence_blocks or len(self.recording) >= self.max_blocks:
                    audio_clip = np.concatenate(self.recording).astype(np.float32)
                    self.is_recording = False
                    self.recording.clear()
                    self.pre_roll.clear()
                    threading.Thread(target=self.run_prediction, args=(audio_clip,), daemon=True).start()

    def run_prediction(self, audio: np.ndarray) -> None:
        label, confidence = voice_predictor.predict(audio)
        print(f"[voice] {label} ({confidence:.1%})")
        self.on_prediction("voice", label, confidence)

    def run(self) -> None:
        if voice_predictor.sd is None:
            print(
                "[voice] Nie mozna uruchomic nasluchu audio bez PortAudio. "
                f"Szczegoly: {voice_predictor.SOUNDDEVICE_IMPORT_ERROR}"
            )
            return

        print("[voice] Nasluch uruchomiony.")
        with voice_predictor.sd.InputStream(
            samplerate=voice_predictor.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=voice_predictor.BLOCKSIZE,
            callback=self.audio_callback,
        ):
            while not self.stop_event.is_set():
                voice_predictor.sd.sleep(100)


class GestureRecognizer(threading.Thread):
    def __init__(
        self,
        stop_event: threading.Event,
        on_prediction,
        command_status: CommandStatus,
        control_mode: ControlMode,
        input_history: InputHistory,
        camera_index: int,
        model_path: Path,
        pose_model_path: Path,
        device_name: str,
        confidence_threshold: float,
        show_window: bool,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float,
        min_tracking_confidence: float,
        landmark_smoothing_alpha: float,
        visibility_threshold: float,
        enhance_contrast_frames: bool,
    ) -> None:
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.on_prediction = on_prediction
        self.command_status = command_status
        self.control_mode = control_mode
        self.input_history = input_history
        self.camera_index = camera_index
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        self.device_name = device_name
        self.confidence_threshold = confidence_threshold
        self.show_window = show_window
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_smoothing_alpha = landmark_smoothing_alpha
        self.visibility_threshold = visibility_threshold
        self.enhance_contrast_frames = enhance_contrast_frames

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Nie znaleziono modelu gestow: {self.model_path}")
            if not self.pose_model_path.exists():
                raise FileNotFoundError(f"Nie znaleziono modelu pozy: {self.pose_model_path}")

            (
                model,
                device,
                normalization_mean,
                normalization_std,
                class_names,
                sequence_length,
                feature_indices,
            ) = load_lstm_model(self.model_path, self.device_name)

            sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
            previous_landmarks: np.ndarray | None = None
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Nie mozna otworzyc kamery: {self.camera_index}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1.0:
                fps = 30.0

            print("[gesture] Inferencja uruchomiona.")
            with create_pose_landmarker(
                self.pose_model_path,
                self.min_pose_detection_confidence,
                self.min_pose_presence_confidence,
                self.min_tracking_confidence,
            ) as landmarker:
                frame_index = 0
                start_time = time.perf_counter()

                while not self.stop_event.is_set():
                    success, frame = cap.read()
                    if not success:
                        print("[gesture] Nie udalo sie odczytac klatki z kamery. Zatrzymuje inferencje.")
                        self.stop_event.set()
                        break

                    frame = cv2.flip(frame, 1)
                    if self.enhance_contrast_frames:
                        frame = enhance_frame(frame)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                    if not timestamp_ms:
                        timestamp_ms = int(round(frame_index * (1000.0 / fps)))
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame_index += 1

                    landmarks = extract_landmarks(detection_result, DEFAULT_LANDMARK_INDICES)
                    landmarks = smooth_landmarks(landmarks, previous_landmarks, self.landmark_smoothing_alpha)
                    landmarks = apply_visibility_threshold(landmarks, self.visibility_threshold)
                    previous_landmarks = landmarks.copy()
                    sequence_buffer.append(landmarks)

                    has_pose = bool(np.any(landmarks[:, 3] > 0.0))
                    predicted_label = None
                    confidence = None
                    top_predictions: list[tuple[str, float]] = []

                    if has_pose and len(sequence_buffer) == sequence_length:
                        predicted_label, confidence, top_predictions = predict_gesture(
                            model=model,
                            sequence=np.stack(sequence_buffer).astype(np.float32),
                            normalization_mean=normalization_mean,
                            normalization_std=normalization_std,
                            class_names=class_names,
                            device=device,
                            feature_indices=feature_indices,
                        )
                        if confidence is not None:
                            print(f"[gesture] {predicted_label} ({confidence:.1%})")
                    else:
                        self.on_prediction("gesture", None, None)

                    if predicted_label is not None and confidence is not None:
                        if confidence >= self.confidence_threshold:
                            self.on_prediction("gesture", predicted_label, confidence)
                        else:
                            self.on_prediction("gesture", None, None)

                    if self.show_window:
                        draw_pose_overlay(frame, landmarks)
                        output_frame = draw_prediction_overlay(
                            frame=frame,
                            predicted_label=predicted_label,
                            confidence=confidence,
                            top_predictions=top_predictions,
                            buffered_frames=len(sequence_buffer),
                            sequence_length=sequence_length,
                            using_camera=True,
                        )
                        output_frame = draw_active_command_overlay(
                            output_frame,
                            self.command_status,
                            self.control_mode,
                            self.input_history,
                        )
                        cv2.imshow("Multimodal MAVSDK control", output_frame)
                        pressed_key = cv2.waitKey(1) & 0xFF
                        if pressed_key in {27, ord("q")}:
                            self.stop_event.set()
                            break
        except Exception:
            print("[gesture] Nieoczekiwany blad w watku inferencji:")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            if cap is not None:
                cap.release()
            if self.show_window:
                cv2.destroyAllWindows()


async def async_main(config: RuntimeConfig) -> None:
    stop_event = threading.Event()
    command_status = CommandStatus()
    control_mode = ControlMode()
    input_history = InputHistory()
    voice_queue: Queue[CommandEvent] = Queue()
    gesture_queue: Queue[CommandEvent] = Queue()
    voice_gate = ConfirmationGate(config.voice_repeat_count, config.voice_confidence_threshold)
    gesture_gate = ConfirmationGate(config.gesture_repeat_count, config.gesture_confidence_threshold)
    last_voice_command_at = 0.0

    def is_voice_priority_active(now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        return (current_time - last_voice_command_at) < config.voice_priority_window

    def handle_prediction(source: str, label: str | None, confidence: float | None) -> None:
        nonlocal last_voice_command_at
        gate = voice_gate if source == "voice" else gesture_gate
        confirmed_label = gate.ingest(label, confidence)
        if confirmed_label is None or confidence is None:
            return

        input_history.add(source, confirmed_label)

        if confirmed_label != "dispatch" and not control_mode.is_enabled(source):
            print(f"[mode] Pomijam komende {source} '{confirmed_label}', bo to sterowanie jest wylaczone.")
            return

        event = CommandEvent(source=source, label=confirmed_label, confidence=confidence)
        print(f"[confirm] {source}: {confirmed_label} ({confidence:.1%})")
        if source == "voice":
            last_voice_command_at = time.monotonic()
            voice_queue.put(event)
            return

        if confirmed_label != "dispatch" and is_voice_priority_active():
            print(f"[priority] Pomijam gest '{confirmed_label}', bo glos ma priorytet.")
            return

        gesture_queue.put(event)

    controller = DroneController(
        connection_url=config.connection_url,
        linear_speed=config.linear_speed,
        vertical_speed=config.vertical_speed,
        yaw_rate_deg_s=config.yaw_rate,
        slow_down_factor=config.slow_down_factor,
        min_linear_speed=config.min_linear_speed,
    )
    await controller.connect()

    voice_thread = VoiceRecognizer(stop_event=stop_event, on_prediction=handle_prediction)
    gesture_thread = GestureRecognizer(
        stop_event=stop_event,
        on_prediction=handle_prediction,
        command_status=command_status,
        control_mode=control_mode,
        input_history=input_history,
        camera_index=config.camera_index,
        model_path=config.gesture_model_path,
        pose_model_path=config.pose_model_path,
        device_name=config.device,
        confidence_threshold=config.gesture_confidence_threshold,
        show_window=not config.hide_window,
        min_pose_detection_confidence=config.min_pose_detection_confidence,
        min_pose_presence_confidence=config.min_pose_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        landmark_smoothing_alpha=config.landmark_smoothing_alpha,
        visibility_threshold=config.visibility_threshold,
        enhance_contrast_frames=config.enhance_contrast,
    )

    voice_thread.start()
    gesture_thread.start()

    try:
        while not stop_event.is_set():
            try:
                event = voice_queue.get_nowait()
            except ThreadQueueEmpty:
                try:
                    event = gesture_queue.get_nowait()
                except ThreadQueueEmpty:
                    await asyncio.sleep(0.05)
                    continue

            if event.source == "gesture" and event.label != "dispatch" and is_voice_priority_active():
                print(f"[priority] Odrzucam oczekujacy gest '{event.label}', bo glos ma priorytet.")
                continue

            if event.label == "dispatch":
                gesture_enabled, voice_enabled = control_mode.toggle_for_source(event.source)
                mode_label = (
                    f"gesture_control_{'on' if gesture_enabled else 'off'}"
                    if event.source == "gesture"
                    else f"voice_control_{'on' if voice_enabled else 'off'}"
                )
                command_status.set(mode_label, event.source)
                print(
                    "[mode] "
                    f"gesture={'on' if gesture_enabled else 'off'}, "
                    f"voice={'on' if voice_enabled else 'off'}"
                )
                continue

            command_status.set(event.label, event.source)
            await controller.execute_command(event.label, event.source)
    finally:
        stop_event.set()
        await controller.stop()
        await controller.shutdown()
        voice_thread.join(timeout=1.0)
        gesture_thread.join(timeout=1.0)


def main() -> None:
    config = CONFIG
    print("Sterowanie multimodalne uruchomione.")
    print(f"  MAVSDK:   {config.connection_url}")
    print(f"  Gesty:    {config.gesture_model_path}")
    print(f"  Glos:     {voice_predictor.MODEL_PATH}")
    print(
        f"  Potwierdzenia: glos={config.voice_repeat_count}x, gest={config.gesture_repeat_count}x"
    )
    print("Zatrzymanie: Ctrl+C lub q w oknie OpenCV.")
    try:
        asyncio.run(async_main(config))
    except KeyboardInterrupt:
        print("\nZatrzymano sterowanie multimodalne.")


if __name__ == "__main__":
    main()