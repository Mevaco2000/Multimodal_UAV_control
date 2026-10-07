# Multimodal MAVSDK Control

Projekt umożliwia **sterowanie dronem przez MAVSDK** na podstawie dwóch źródeł wejścia:

- komend głosowych (model audio z `realtime_predict.py`),
- gestów ciała (MediaPipe Pose + klasyfikator LSTM z `live_gesture_inference.py`).

Główny punkt wejścia to skrypt `multimodal_mavsdk_control.py`, który łączy oba kanały sterowania, filtruje predykcje i przekłada je na komendy lotu.

---

## Co robi ten projekt

Skrypt:

1. łączy się z dronem/autopilotem przez MAVSDK,
2. uruchamia dwa wątki:
   - rozpoznawanie głosu,
   - rozpoznawanie gestów z kamery,
3. potwierdza komendy przez mechanizm powtórzeń (żeby ograniczyć przypadkowe aktywacje),
4. utrzymuje aktywną komendę i pokazuje overlay w oknie OpenCV,
5. wykonuje komendy ruchu/offboard/uzbrojenia/lądowania.

---

## Struktura folderu

- `multimodal_mavsdk_control.py` – główna logika multimodalna i sterowanie MAVSDK.
- `live_gesture_inference.py` – inferencja gestów (MediaPipe + LSTM), funkcje pomocnicze do landmarków i overlay.
- `realtime_predict.py` – inferencja komend głosowych (VAD + model TorchScript).
- `requirements.txt` – zależności Pythona.

Oczekiwane pliki modeli:

- `artifacts/moja_wersja/gestures_v5_cl_arm.pt` (domyślny model gestów),
- `pose_landmarker_lite.task` (model MediaPipe Pose),
- `model_wav2vec2_commands.torchscript.pt` (model głosowy).

> Modele są przechowywane w Git LFS. Po klonowaniu wykonaj `git lfs pull`, żeby pobrać pełne pliki binarne (`.pt`, `.task`).

---

## Wymagania

- Python 3.10+ (zalecane 3.10/3.11),
- kamera (dla gestów),
- mikrofon (dla głosu),
- działający endpoint MAVSDK/SITL/autopilot,
- biblioteka PortAudio (wymagana przez `sounddevice`).

Zależności z `requirements.txt`:

- `numpy`
- `opencv-python`
- `mediapipe`
- `torch`
- `mavsdk`
- `sounddevice`

---

## Instalacja Git i Git LFS

Jeżeli nie masz jeszcze Git/Git LFS, wykonaj te kroki:

1. Na Raspberry Pi OS zainstaluj Git:

```bash
sudo apt update
sudo apt install -y git
```

2. Sprawdź instalację:

```bash
git --version
```

3. Zainstaluj **Git LFS**:

```bash
sudo apt install -y git-lfs
```

4. Sprawdź instalację:

```bash
git lfs version
```

5. Jednorazowo aktywuj Git LFS:

```bash
git lfs install
```

6. Sklonuj repozytorium i pobierz pliki LFS:

```bash
git clone <URL_REPOZYTORIUM>
cd <FOLDER_REPOZYTORIUM>
git lfs pull
```

Po tym kroku pliki modeli śledzone przez LFS powinny być dostępne lokalnie.

---

## Instalacja

W folderze projektu:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jeżeli `sounddevice` zgłasza błąd inicjalizacji, doinstaluj/napraw PortAudio w systemie.

---

## Uruchomienie

Podstawowe uruchomienie:

```bash
python multimodal_mavsdk_control.py
```

Zatrzymanie:

- `Ctrl+C` w terminalu lub
- `q` / `Esc` w oknie OpenCV.

---

## Konfiguracja (bez parsera CLI)

Projekt nie korzysta z argumentów wiersza poleceń.
Konfiguracja jest ustawiana w kodzie, w `multimodal_mavsdk_control.py`, w strukturze `RuntimeConfig` oraz stałej `CONFIG`.

Najważniejsze pola:

- `connection_url` – adres systemu MAVSDK (domyślnie `udpin://127.0.0.1:14540`),
- `gesture_model_path` – ścieżka do modelu gestów,
- `pose_model_path` – ścieżka do modelu pozy MediaPipe,
- `camera_index` – indeks kamery,
- `device` (`auto`/`cpu`/`cuda`) – urządzenie dla modelu gestów.

### Potwierdzanie komend i priorytety

- `voice_confidence_threshold` – minimalna pewność komendy głosowej,
- `gesture_confidence_threshold` – minimalna pewność gestu,
- `voice_repeat_count` – liczba kolejnych identycznych predykcji głosu do akceptacji,
- `gesture_repeat_count` – liczba kolejnych identycznych predykcji gestu do akceptacji,
- `voice_priority_window` – czas (s), w którym głos ma priorytet nad gestem.

### Parametry ruchu drona

- `linear_speed` – prędkość ruchu poziomego (m/s),
- `vertical_speed` – prędkość pionowa (m/s),
- `yaw_rate` – prędkość obrotu yaw (deg/s),
- `slow_down_factor` – mnożnik dla komendy `slow_down`,
- `min_linear_speed` – minimalna prędkość liniowa po spowolnieniach.

### Parametry wizji

- `min_pose_detection_confidence`
- `min_pose_presence_confidence`
- `min_tracking_confidence`
- `landmark_smoothing_alpha`
- `visibility_threshold`
- `enhance_contrast` – lokalne podbicie kontrastu klatek,
- `hide_window` – uruchomienie bez podglądu OpenCV.

---

## Obsługiwane komendy lotu (warstwa wykonawcza)

Warstwa wykonawcza `DroneController.execute_command` rozpoznaje m.in.:

- `arm`, `disarm`, `land`, `hover`,
- `move_ahead`, `backward`,
- `move_left`, `move_right`,
- `move_upward`, `move_down`,
- `turn_left`, `turn_right`,
- `slow_down`,
- `dispatch` (specjalna komenda przełączająca tryb sterowania źródłem).

`dispatch` **nie wykonuje manewru lotu** — przełącza włączanie/wyłączanie sterowania:

- wypowiedziane `dispatch` przełącza sterowanie głosowe,
- pokazane `dispatch` (gest) przełącza sterowanie gestami.

---

## Jak działa logika priorytetów

1. Każdy kanał (głos/gest) ma osobną bramkę potwierdzania (`ConfirmationGate`).
2. Po zaakceptowaniu komendy głosowej uruchamiane jest okno priorytetu (`voice_priority_window`).
3. W tym oknie czasowym komendy z gestów (poza `dispatch`) są ignorowane.
4. Aktywna komenda, tryby sterowania i historia wejścia są rysowane w overlayu.

---

## Typowe problemy

### 1) `Nie znaleziono modelu gestow` / `Nie znaleziono modelu pozy`

Sprawdź obecność plików modelu i poprawność pól `gesture_model_path` oraz `pose_model_path` w `RuntimeConfig`.

### 2) Błąd audio / brak PortAudio

Skrypt głosowy wymaga `sounddevice` + PortAudio. Jeżeli PortAudio nie jest dostępne, wątek głosu nie wystartuje.

### 3) Brak połączenia z dronem

Zweryfikuj `connection_url` i to, czy endpoint MAVSDK/SITL działa oraz nasłuchuje na odpowiednim porcie.

### 4) Brak obrazu z kamery

Ustaw właściwy indeks `camera_index` (np. `0`, `1`, `2`) i upewnij się, że kamera nie jest zajęta przez inny proces.

---

## Uwagi bezpieczeństwa

To jest kod sterowania ruchem. Testuj najpierw:

1. na symulatorze (SITL),
2. z bardzo zachowawczymi prędkościami,
3. w kontrolowanym środowisku.

Przed testami na realnym dronie upewnij się, że masz fizyczny i programowy mechanizm awaryjnego przejęcia kontroli.
