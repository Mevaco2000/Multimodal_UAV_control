from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except Exception as exc:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_ASSET_DIR = SCRIPT_DIR.parent / "sterowanie_glosem"


def resolve_asset_path(file_name: str) -> Path:
    local_path = SCRIPT_DIR / file_name
    if local_path.exists():
        return local_path

    legacy_path = LEGACY_ASSET_DIR / file_name
    if legacy_path.exists():
        return legacy_path

    return local_path


MODEL_PATH = resolve_asset_path("model_wav2vec2_commands.torchscript.pt")
CHECKPOINT_PATH = resolve_asset_path("model_wav2vec2_commands.pt")
SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000
TARGET_RMS = 0.08

VAD_THRESHOLD = 0.015
SILENCE_AFTER_MS = 600
PRE_ROLL_MS = 150
MAX_RECORD_MS = 3000

CONFIDENCE_THRESHOLD = 0.55
BLOCKSIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_raw = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
CLASSES: list[str] = checkpoint_raw["classes"]
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()


def normalize_rms(waveform: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < 1e-6:
        return waveform
    gain = target_rms / rms
    return np.clip(waveform * gain, -1.0, 1.0)


def preprocess(audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    audio = audio.astype(np.float32, copy=False)
    sample_count = len(audio)
    if sample_count >= CLIP_SAMPLES:
        audio = audio[:CLIP_SAMPLES]
    else:
        pad = CLIP_SAMPLES - sample_count
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    audio = audio - float(audio.mean())
    audio = normalize_rms(audio)

    waveforms = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([CLIP_SAMPLES], dtype=torch.long, device=DEVICE)
    return waveforms, lengths


@torch.no_grad()
def predict(audio: np.ndarray) -> tuple[str, float]:
    waveforms, lengths = preprocess(audio)
    logits = model(waveforms, lengths)
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(probabilities.argmax())
    return CLASSES[predicted_index], float(probabilities[predicted_index])


PRE_ROLL_BLOCKS = max(1, int(PRE_ROLL_MS * SAMPLE_RATE / 1000 / BLOCKSIZE))
SILENCE_BLOCKS = max(1, int(SILENCE_AFTER_MS * SAMPLE_RATE / 1000 / BLOCKSIZE))
MAX_BLOCKS = int(MAX_RECORD_MS * SAMPLE_RATE / 1000 / BLOCKSIZE)

pre_roll: list[np.ndarray] = []
recording: list[np.ndarray] = []
silence_count = [0]
is_recording = [False]
lock = threading.Lock()


def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    del frames, time_info, status
    chunk = indata[:, 0].copy()
    rms = float(np.sqrt(np.mean(chunk**2)))

    with lock:
        if not is_recording[0]:
            pre_roll.append(chunk)
            if len(pre_roll) > PRE_ROLL_BLOCKS:
                pre_roll.pop(0)

            if rms > VAD_THRESHOLD:
                is_recording[0] = True
                silence_count[0] = 0
                recording.clear()
                recording.extend(pre_roll)
                recording.append(chunk)
        else:
            recording.append(chunk)

            if rms < VAD_THRESHOLD:
                silence_count[0] += 1
            else:
                silence_count[0] = 0

            if silence_count[0] >= SILENCE_BLOCKS or len(recording) >= MAX_BLOCKS:
                audio_clip = np.concatenate(recording).astype(np.float32)
                is_recording[0] = False
                recording.clear()
                pre_roll.clear()
                threading.Thread(target=run_prediction, args=(audio_clip,), daemon=True).start()


def run_prediction(audio: np.ndarray) -> None:
    label, confidence = predict(audio)
    bar = "█" * int(confidence * 20)
    if confidence >= CONFIDENCE_THRESHOLD:
        line = f"  >>> {label:<15}  {confidence * 100:5.1f}%  {bar}"
    else:
        line = f"  ??? (niepewny)         {confidence * 100:5.1f}%  {bar}"
    print(f"\r{line:<65}", flush=True)
    print("", flush=True)

    if sd is not None:
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    if sd is None:
        raise RuntimeError(f"Nie mozna uruchomic audio: {SOUNDDEVICE_IMPORT_ERROR}")

    print(f"Model: {MODEL_PATH}")
    print(f"Checkpoint z klasami: {CHECKPOINT_PATH}")
    print(f"Klasy: {', '.join(CLASSES)}")
    print(f"Urzadzenie: {DEVICE}")
    print(f"Prog VAD: RMS > {VAD_THRESHOLD}")
    print(f"Cisza po komendzie: {SILENCE_AFTER_MS} ms")
    print(f"Prog pewnosci: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print("\nMow komende... (Ctrl+C aby zakonczyc)\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\nZatrzymano.")