from __future__ import annotations

import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Brakuje pakietu 'torch'. Zainstaluj go poleceniem: pip install torch"
    ) from exc


DEFAULT_FILE_GLOB = "*_predicted.npy"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "nagrania_keypoints"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "lstm_gesture"
DEFAULT_FEATURE_INDICES: tuple[int, ...] | None = None
DEFAULT_AUGMENTATION_CONFIG = {
    "enabled": False,
    "jitter_std": 0.01,
    "shift_std": 0.02,
    "scale_std": 0.03,
    "frame_dropout_prob": 0.03,
    "keypoint_dropout_prob": 0.05,
}
DEFAULT_GESTURES = [
    "arm",
    "backward",
    "disarm",
    "dispatch",
    "hover",
    "land",
    "move_ahead",
    "move_down",
    "move_left",
    "move_right",
    "move_upward",
    "slow_down",
    "turn_left",
    "turn_right",
]


def humanize_label(label: str) -> str:
    return label.replace("_", " ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_sequence(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim < 2:
        raise ValueError(f"Oczekiwano tablicy co najmniej 2D, otrzymano shape={sequence.shape}.")

    time_steps = sequence.shape[0]
    feature_dim = int(np.prod(sequence.shape[1:]))
    return sequence.reshape(time_steps, feature_dim).astype(np.float32)


def select_feature_indices(
    sequence: np.ndarray,
    feature_indices: tuple[int, ...] | None,
) -> np.ndarray:
    if feature_indices is None:
        return sequence
    if sequence.ndim < 3:
        raise ValueError(
            "feature_indices mozna ustawic tylko dla danych o ksztalcie co najmniej 3D, "
            f"otrzymano shape={sequence.shape}."
        )

    feature_count = sequence.shape[-1]
    invalid_indices = [index for index in feature_indices if index < 0 or index >= feature_count]
    if invalid_indices:
        raise ValueError(
            f"Niepoprawne feature_indices={invalid_indices}. Ostatni wymiar ma rozmiar {feature_count}."
        )

    return sequence[..., list(feature_indices)].astype(np.float32)


def load_samples(
    data_dir: Path,
    file_glob: str,
    gestures: list[str] | None,
    feature_indices: tuple[int, ...] | None = DEFAULT_FEATURE_INDICES,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu z danymi: {data_dir}")

    if gestures:
        missing_directories = [gesture for gesture in gestures if not (data_dir / gesture).is_dir()]
        if missing_directories:
            raise FileNotFoundError(
                "Brakuje katalogow klas w zbiorze danych: " + ", ".join(missing_directories)
            )

        files: list[Path] = []
        for gesture in gestures:
            files.extend(sorted((data_dir / gesture).glob(file_glob)))
    else:
        files = sorted(path for path in data_dir.rglob(file_glob) if path.is_file())

    if not files:
        raise FileNotFoundError(
            f"Nie znaleziono plikow pasujacych do wzorca '{file_glob}' w katalogu {data_dir}"
        )

    sequences: list[np.ndarray] = []
    label_names: list[str] = []
    sample_paths: list[str] = []
    expected_shape: tuple[int, int] | None = None

    for path in files:
        label_name = path.parent.name
        array = np.load(path)
        array = select_feature_indices(array, feature_indices)
        flattened = flatten_sequence(array)

        sequence_shape = (flattened.shape[0], flattened.shape[1])
        if expected_shape is None:
            expected_shape = sequence_shape
        elif sequence_shape != expected_shape:
            raise ValueError(
                "Wszystkie probki musza miec ten sam ksztalt po splaszczeniu. "
                f"Pierwszy ksztalt: {expected_shape}, plik {path} ma {sequence_shape}."
            )

        sequences.append(flattened)
        label_names.append(label_name)
        sample_paths.append(str(path))

    if gestures:
        class_names = [gesture for gesture in gestures if gesture in set(label_names)]
    else:
        class_names = sorted(set(label_names))
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    labels = np.array([class_to_index[name] for name in label_names], dtype=np.int64)
    features = np.stack(sequences).astype(np.float32)
    return features, labels, class_names, sample_paths


def truncate_sequence_length(
    features: np.ndarray,
    sequence_length: int | None,
) -> tuple[np.ndarray, int]:
    available_sequence_length = int(features.shape[1])
    if sequence_length is None:
        return features, available_sequence_length
    if sequence_length <= 0:
        raise ValueError("sequence_length musi byc wieksze od 0 albo None.")
    if sequence_length > available_sequence_length:
        raise ValueError(
            "sequence_length nie moze byc wieksze od dlugosci probek. "
            f"Otrzymano {sequence_length}, ale probki maja {available_sequence_length} krokow."
        )
    if sequence_length == available_sequence_length:
        return features, available_sequence_length
    return features[:, :sequence_length, :].astype(np.float32), available_sequence_length


def split_indices(labels: np.ndarray, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split musi byc w zakresie (0, 1).")

    rng = random.Random(seed)
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped_indices[label].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for label, indices in grouped_indices.items():
        rng.shuffle(indices)
        if len(indices) == 1:
            raise ValueError(
                f"Klasa o indeksie {label} ma tylko 1 probke. Potrzebne sa co najmniej 2 probki na klase."
            )

        val_count = max(1, int(math.ceil(len(indices) * val_split)))
        if val_count >= len(indices):
            val_count = len(indices) - 1

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def resolve_selected_feature_dim(
    feature_indices: tuple[int, ...] | None,
    raw_input_size: int,
) -> int | None:
    if feature_indices is not None:
        return len(feature_indices)
    for candidate in (2, 3, 4):
        if raw_input_size % candidate == 0:
            return candidate
    return None


class SequenceAugmentor:
    def __init__(
        self,
        selected_feature_dim: int | None,
        jitter_std: float = 0.01,
        shift_std: float = 0.02,
        scale_std: float = 0.03,
        frame_dropout_prob: float = 0.03,
        keypoint_dropout_prob: float = 0.05,
    ):
        self.selected_feature_dim = selected_feature_dim
        self.jitter_std = float(jitter_std)
        self.shift_std = float(shift_std)
        self.scale_std = float(scale_std)
        self.frame_dropout_prob = float(frame_dropout_prob)
        self.keypoint_dropout_prob = float(keypoint_dropout_prob)

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        augmented = sequence.clone()
        if augmented.ndim != 2 or augmented.numel() == 0:
            return augmented

        if self.jitter_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.jitter_std

        if self.frame_dropout_prob > 0:
            frame_mask = torch.rand(augmented.shape[0], device=augmented.device) < self.frame_dropout_prob
            if bool(frame_mask.any()):
                augmented[frame_mask] = 0.0

        if (
            self.selected_feature_dim is None
            or self.selected_feature_dim <= 0
            or augmented.shape[1] % self.selected_feature_dim != 0
        ):
            return augmented

        keypoint_count = augmented.shape[1] // self.selected_feature_dim
        structured = augmented.view(augmented.shape[0], keypoint_count, self.selected_feature_dim)
        coordinate_dims = min(2, self.selected_feature_dim)

        if coordinate_dims > 0 and self.shift_std > 0:
            shift = torch.randn(1, 1, coordinate_dims, device=structured.device) * self.shift_std
            structured[:, :, :coordinate_dims] = structured[:, :, :coordinate_dims] + shift

        if coordinate_dims > 0 and self.scale_std > 0:
            scale = 1.0 + torch.randn(1, 1, 1, device=structured.device) * self.scale_std
            structured[:, :, :coordinate_dims] = structured[:, :, :coordinate_dims] * scale

        if self.keypoint_dropout_prob > 0:
            keypoint_mask = (
                torch.rand(structured.shape[0], structured.shape[1], 1, device=structured.device)
                < self.keypoint_dropout_prob
            )
            if bool(keypoint_mask.any()):
                structured = structured.masked_fill(keypoint_mask, 0.0)

        return structured.view_as(augmented)


def resolve_augmentation_config(
    augmentation: bool | dict[str, float | bool] | None,
) -> dict[str, float | bool]:
    if augmentation is None:
        return dict(DEFAULT_AUGMENTATION_CONFIG)
    if isinstance(augmentation, bool):
        config = dict(DEFAULT_AUGMENTATION_CONFIG)
        config["enabled"] = augmentation
        return config
    config = dict(DEFAULT_AUGMENTATION_CONFIG)
    config.update(augmentation)
    return config


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: SequenceAugmentor | None = None,
    ):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.transform is not None:
            features = self.transform(features)
        return features, self.labels[index]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        last_output = outputs[:, -1, :]
        return self.classifier(last_output)


def compute_normalization(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def get_device(device_name: str) -> torch.device:
    normalized_name = device_name.strip().lower()
    if normalized_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA nie jest dostepna. Trening zostanie uruchomiony na CPU.", RuntimeWarning, stacklevel=2)
            return torch.device("cpu")
        return torch.device("cuda")
    if normalized_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

            total_loss += float(loss.item()) * batch_labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == batch_labels).sum().item())
            total_examples += int(batch_labels.size(0))

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def evaluate_per_class(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    model.eval()
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            predictions = logits.argmax(dim=1)

            for label_index, predicted_index in zip(batch_labels.tolist(), predictions.tolist()):
                class_name = class_names[int(label_index)]
                class_total[class_name] += 1
                if int(predicted_index) == int(label_index):
                    class_correct[class_name] += 1

    class_accuracy = {
        class_name: class_correct[class_name] / max(class_total[class_name], 1)
        for class_name in class_names
    }
    return class_accuracy, class_correct, class_total


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == batch_labels).sum().item())
        total_examples += int(batch_labels.size(0))

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def build_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    transform: SequenceAugmentor | None = None,
) -> DataLoader:
    dataset = SequenceDataset(features, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def resolve_output_paths(
    output_dir: Path | None,
    model_path: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if model_path is not None:
        resolved_model_path = Path(model_path)
        resolved_output_dir = resolved_model_path.parent
    else:
        resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        resolved_model_path = resolved_output_dir / "gesture_lstm.pt"

    history_path = resolved_output_dir / "training_history.json"
    metadata_path = resolved_output_dir / "training_metadata.json"
    labels_path = resolved_output_dir / "labels.json"
    return resolved_model_path, metadata_path, history_path, labels_path


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    model_path: Path | None = None,
    file_glob: str = DEFAULT_FILE_GLOB,
    feature_indices: tuple[int, ...] | None = DEFAULT_FEATURE_INDICES,
    sequence_length: int | None = None,
    epochs: int = 2000,
    batch_size: int = 16,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    device_name: str = "auto",
    gestures: list[str] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    reduce_lr_patience: int | None = 5,
    reduce_lr_factor: float = 0.5,
    reduce_lr_threshold: float = 1e-3,
    reduce_lr_min_lr: float = 1e-6,
    augmentation: bool | dict[str, float | bool] | None = None,
    use_class_weights: bool = True,
) -> dict[str, object]:
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience musi byc wieksze od 0 albo None.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta nie moze byc ujemne.")
    if reduce_lr_patience is not None and reduce_lr_patience <= 0:
        raise ValueError("reduce_lr_patience musi byc wieksze od 0 albo None.")
    if not 0.0 < reduce_lr_factor < 1.0:
        raise ValueError("reduce_lr_factor musi byc w zakresie (0, 1).")
    if reduce_lr_threshold < 0:
        raise ValueError("reduce_lr_threshold nie moze byc ujemne.")
    if reduce_lr_min_lr < 0:
        raise ValueError("reduce_lr_min_lr nie moze byc ujemne.")

    augmentation_config = resolve_augmentation_config(augmentation)
    for numeric_key in (
        "jitter_std",
        "shift_std",
        "scale_std",
        "frame_dropout_prob",
        "keypoint_dropout_prob",
    ):
        numeric_value = float(augmentation_config[numeric_key])
        if numeric_value < 0:
            raise ValueError(f"{numeric_key} nie moze byc ujemne.")
        augmentation_config[numeric_key] = numeric_value

    set_seed(seed)

    resolved_gestures = gestures if gestures is not None else DEFAULT_GESTURES
    features, labels, class_names, sample_paths = load_samples(
        data_dir,
        file_glob,
        resolved_gestures,
        feature_indices=feature_indices,
    )
    features, source_sequence_length = truncate_sequence_length(features, sequence_length)
    train_indices, val_indices = split_indices(labels, val_split, seed)

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]

    mean, std = compute_normalization(train_features)
    train_features = normalize_features(train_features, mean, std)
    val_features = normalize_features(val_features, mean, std)

    train_transform = None
    if bool(augmentation_config["enabled"]):
        train_transform = SequenceAugmentor(
            selected_feature_dim=resolve_selected_feature_dim(feature_indices, int(train_features.shape[-1])),
            jitter_std=float(augmentation_config["jitter_std"]),
            shift_std=float(augmentation_config["shift_std"]),
            scale_std=float(augmentation_config["scale_std"]),
            frame_dropout_prob=float(augmentation_config["frame_dropout_prob"]),
            keypoint_dropout_prob=float(augmentation_config["keypoint_dropout_prob"]),
        )

    train_loader = build_data_loader(
        train_features,
        train_labels,
        batch_size,
        shuffle=True,
        transform=train_transform,
    )
    val_loader = build_data_loader(val_features, val_labels, batch_size, shuffle=False)

    device = get_device(device_name)
    model = LSTMClassifier(
        input_size=train_features.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(class_names),
        dropout=dropout,
    ).to(device)

    class_weights_list: list[float] | None = None
    if use_class_weights:
        class_counts = Counter(train_labels.tolist())
        class_weights_list = [
            len(train_labels) / (len(class_names) * class_counts[index])
            for index in range(len(class_names))
        ]
        class_weights = torch.tensor(
            class_weights_list,
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if reduce_lr_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            threshold=reduce_lr_threshold,
            min_lr=reduce_lr_min_lr,
        )

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    best_history_entry: dict[str, float | int] | None = None
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={current_lr:.6g}"
        )

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            best_history_entry = dict(history[-1])
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                "Early stopping: brak poprawy val_loss przez "
                f"{early_stopping_patience} epok. Zatrzymano na epoce {epoch}."
            )
            break

    if best_state_dict is None or best_history_entry is None:
        raise RuntimeError("Trening nie zapisal zadnego stanu modelu.")

    model.load_state_dict(best_state_dict)
    val_class_accuracy, val_class_correct, val_class_total = evaluate_per_class(
        model,
        val_loader,
        device,
        class_names,
    )
    worst_val_class = min(val_class_accuracy, key=val_class_accuracy.get)
    worst_val_class_accuracy = float(val_class_accuracy[worst_val_class])

    model_path, metadata_path, history_path, labels_path = resolve_output_paths(output_dir, model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input_size": int(train_features.shape[-1]),
            "sequence_length": int(train_features.shape[1]),
            "feature_indices": None if feature_indices is None else list(feature_indices),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "class_names": class_names,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        },
        model_path,
    )

    final_history_entry = dict(history[-1]) if history else None

    metadata = {
        "data_dir": str(data_dir),
        "file_glob": file_glob,
        "gestures": class_names,
        "display_names": {class_name: humanize_label(class_name) for class_name in class_names},
        "num_samples": int(features.shape[0]),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "source_sequence_length": source_sequence_length,
        "sequence_length": int(features.shape[1]),
        "input_size": int(features.shape[2]),
        "feature_indices": None if feature_indices is None else list(feature_indices),
        "classes": class_names,
        "class_distribution": {
            class_names[index]: int((labels == index).sum()) for index in range(len(class_names))
        },
        "train_class_distribution": {
            class_names[index]: int((train_labels == index).sum()) for index in range(len(class_names))
        },
        "val_class_distribution": {
            class_names[index]: int((val_labels == index).sum()) for index in range(len(class_names))
        },
        "val_class_accuracy": val_class_accuracy,
        "val_class_correct": val_class_correct,
        "val_class_total": val_class_total,
        "worst_val_class": worst_val_class,
        "worst_val_class_accuracy": worst_val_class_accuracy,
        "train_paths": [sample_paths[index] for index in train_indices],
        "val_paths": [sample_paths[index] for index in val_indices],
        "best_epoch": best_epoch,
        "best_epoch_metrics": best_history_entry,
        "final_epoch_metrics": final_history_entry,
        "best_val_loss": float(best_history_entry["val_loss"]),
        "best_val_accuracy": float(best_history_entry["val_accuracy"]),
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "reduce_lr_patience": reduce_lr_patience,
        "reduce_lr_factor": reduce_lr_factor,
        "reduce_lr_threshold": reduce_lr_threshold,
        "reduce_lr_min_lr": reduce_lr_min_lr,
        "augmentation": augmentation_config,
        "use_class_weights": use_class_weights,
        "class_weights": class_weights_list,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "device": str(device),
    }

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print(f"Model zapisano w: {model_path}")
    print(f"Metadane zapisano w: {metadata_path}")
    print(f"Historie treningu zapisano w: {history_path}")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "history_path": str(history_path),
        "labels_path": str(labels_path),
        "num_samples": int(features.shape[0]),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "source_sequence_length": source_sequence_length,
        "class_distribution": {
            class_names[index]: int((labels == index).sum()) for index in range(len(class_names))
        },
        "train_class_distribution": {
            class_names[index]: int((train_labels == index).sum()) for index in range(len(class_names))
        },
        "val_class_distribution": {
            class_names[index]: int((val_labels == index).sum()) for index in range(len(class_names))
        },
        "val_class_accuracy": val_class_accuracy,
        "val_class_correct": val_class_correct,
        "val_class_total": val_class_total,
        "worst_val_class": worst_val_class,
        "worst_val_class_accuracy": worst_val_class_accuracy,
        "best_epoch": best_epoch,
        "best_epoch_metrics": best_history_entry,
        "final_epoch_metrics": final_history_entry,
        "best_val_loss": float(best_history_entry["val_loss"]),
        "best_val_accuracy": float(best_history_entry["val_accuracy"]),
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "augmentation": augmentation_config,
        "use_class_weights": use_class_weights,
        "class_weights": class_weights_list,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "classes": class_names,
    }