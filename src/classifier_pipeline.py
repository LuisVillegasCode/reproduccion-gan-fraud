"""Clasificador baseline reutilizable para fraude con tarjetas de crédito.

No contiene rutas, persistencia ni dependencias de Google Colab. Recibe datos y
configuración explícitos y devuelve resultados estructurados.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FraudClassifier(nn.Module):
    """Arquitectura del paper: 30 ReLU -> 30 Sigmoid -> 2 logits."""

    def __init__(
        self,
        input_dim: int = 30,
        first_hidden_dim: int = 30,
        second_hidden_dim: int = 30,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if min(input_dim, first_hidden_dim, second_hidden_dim) <= 0:
            raise ValueError("Las dimensiones del clasificador deben ser positivas.")
        if output_dim != 2:
            raise ValueError("La reproducción binaria requiere output_dim=2.")

        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.ReLU(),
            nn.Linear(first_hidden_dim, second_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(second_hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    """Restablece las fuentes de aleatoriedad usadas por el entrenamiento."""
    if seed < 0:
        raise ValueError("La semilla debe ser no negativa.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device

    name = device.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible.")
    if name not in {"cpu", "cuda"}:
        raise ValueError("device debe ser 'auto', 'cpu' o 'cuda'.")
    return torch.device(name)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X e y deben ser arreglos NumPy.")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X debe ser 2D e y debe ser 1D.")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X e y deben contener el mismo número de muestras no vacío.")
    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o valores infinitos.")
    labels = np.unique(y)
    if labels.size != 2 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y debe contener ambas clases binarias 0 y 1.")


def initialize_model(model: nn.Module, config: Mapping[str, Any]) -> None:
    """Inicializa pesos y sesgos con la distribución indicada."""
    method = str(config.get("method", "uniform")).lower()

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        if method == "uniform":
            minimum = float(config.get("min_value", -0.5))
            maximum = float(config.get("max_value", 0.5))
            if minimum >= maximum:
                raise ValueError("min_value debe ser menor que max_value.")
            nn.init.uniform_(module.weight, minimum, maximum)
            if module.bias is not None:
                nn.init.uniform_(module.bias, minimum, maximum)

        elif method == "normal":
            mean = float(config.get("mean", 0.0))
            std = float(config.get("std", 1.0))
            if std <= 0:
                raise ValueError("std debe ser mayor que cero.")
            nn.init.normal_(module.weight, mean, std)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean, std)

        else:
            raise ValueError("La inicialización debe ser 'uniform' o 'normal'.")


def build_model(
    model_config: Mapping[str, Any],
    seed: int,
    device: str | torch.device = "auto",
) -> FraudClassifier:
    """Crea desde cero el modelo configurado y lo mueve al dispositivo."""
    hidden_layers = model_config.get("hidden_layers", [])
    if len(hidden_layers) != 2:
        raise ValueError("classifier_model.hidden_layers debe contener dos capas.")

    activations = [str(layer["activation"]).lower() for layer in hidden_layers]
    if activations != ["relu", "sigmoid"]:
        raise ValueError("La arquitectura requiere activaciones ReLU y Sigmoid.")

    set_seed(seed)
    model = FraudClassifier(
        input_dim=int(model_config.get("input_dim", 30)),
        first_hidden_dim=int(hidden_layers[0]["units"]),
        second_hidden_dim=int(hidden_layers[1]["units"]),
        output_dim=int(model_config.get("output_dim", 2)),
    )
    initialize_model(model, model_config.get("initialization", {}))
    return model.to(resolve_device(device))


def split_train_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserva validación estratificada exclusivamente desde entrenamiento."""
    _validate_xy(X_train, y_train)
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio debe estar entre 0 y 1.")

    split = train_test_split(
        X_train,
        y_train,
        test_size=validation_ratio,
        random_state=seed,
        stratify=y_train,
    )
    X_subtrain, X_val, y_subtrain, y_val = split

    if np.unique(y_subtrain).size != 2 or np.unique(y_val).size != 2:
        raise ValueError("Subentrenamiento y validación deben contener ambas clases.")
    return X_subtrain, X_val, y_subtrain, y_val


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: str | torch.device = "auto",
) -> DataLoader:
    """Construye un DataLoader reproducible con tipos compatibles con PyTorch."""
    _validate_xy(X, y)
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo.")
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo.")

    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )
    use_pin_memory = pin_memory and resolve_device(device).type == "cuda"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=generator if shuffle else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )


def calculate_momentum(
    epoch_index: int,
    initial: float = 0.5,
    final: float = 0.99,
    warmup_epochs: int = 10,
) -> float:
    """Incrementa linealmente el momentum de 0.5 a 0.99 en 10 épocas."""
    if epoch_index < 0 or warmup_epochs <= 0:
        raise ValueError("epoch_index debe ser no negativo y warmup_epochs positivo.")
    if not 0.0 <= initial <= final < 1.0:
        raise ValueError("Se requiere 0 <= initial <= final < 1.")
    if epoch_index >= warmup_epochs:
        return final

    progress = epoch_index / max(warmup_epochs - 1, 1)
    return initial + progress * (final - initial)


def _validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float | int]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("classification_threshold debe estar entre 0 y 1.")

    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device, non_blocking=True))
            scores = torch.softmax(logits, dim=1)[:, 1]
            predictions = (scores >= threshold).to(torch.int64)
            true_parts.append(y_batch.numpy())
            pred_parts.append(predictions.cpu().numpy())
            score_parts.append(scores.cpu().numpy())

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    y_score = np.concatenate(score_parts)

    if np.unique(y_true).size != 2:
        raise ValueError("La validación debe contener ambas clases.")
    if not np.isfinite(y_score).all():
        raise RuntimeError("El modelo produjo probabilidades no finitas.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fraud_scores = y_score[y_true == 1]

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positives": int((y_pred == 1).sum()),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "max_p_fraud": float(y_score.max()),
        "fraud_p50": float(np.quantile(fraud_scores, 0.50)),
        "fraud_p90": float(np.quantile(fraud_scores, 0.90)),
    }


def _training_components(
    model: nn.Module,
    training_config: Mapping[str, Any],
) -> tuple[nn.CrossEntropyLoss, torch.optim.SGD]:
    if str(training_config.get("loss", "cross_entropy")).lower() != "cross_entropy":
        raise ValueError("La reproducción utiliza CrossEntropyLoss sin ponderación.")

    optimizer_config = training_config.get("optimizer", {})
    if str(optimizer_config.get("name", "sgd")).lower() != "sgd":
        raise ValueError("La reproducción utiliza SGD.")

    learning_rate = float(training_config["learning_rate"])
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo.")

    initial_momentum = float(training_config.get("momentum", {}).get("initial", 0.5))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=initial_momentum,
        nesterov=bool(optimizer_config.get("nesterov", True)),
    )
    return nn.CrossEntropyLoss(), optimizer


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_batch), y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Pérdida no finita: {loss.item()}.")

        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("El DataLoader de entrenamiento no produjo muestras.")
    return total_loss / total_samples


def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    training_config: Mapping[str, Any],
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Selecciona la mejor época por PR-AUC y desempata por ROC-AUC."""
    device = resolve_device(device)
    model.to(device)

    max_epochs = int(training_config.get("max_epochs", 100))
    min_epochs = int(training_config.get("min_epochs", 20))
    patience = int(training_config.get("patience", 15))
    min_delta = float(training_config.get("min_delta", 1e-5))
    threshold = float(training_config.get("classification_threshold", 0.5))

    if max_epochs <= 0 or min_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs, min_epochs y patience deben ser positivos.")
    if min_epochs > max_epochs or min_delta < 0:
        raise ValueError("Configuración inválida de early stopping.")

    selection_config = training_config.get("model_selection", {})
    if str(selection_config.get("primary_metric", "pr_auc")).lower() != "pr_auc":
        raise ValueError("La métrica principal debe ser PR-AUC.")
    if str(selection_config.get("tie_break_metric", "roc_auc")).lower() != "roc_auc":
        raise ValueError("La métrica de desempate debe ser ROC-AUC.")

    momentum_config = training_config.get("momentum", {})
    initial = float(momentum_config.get("initial", 0.5))
    final = float(momentum_config.get("final", 0.99))
    warmup = int(momentum_config.get("warmup_epochs", 10))
    criterion, optimizer = _training_components(model, training_config)

    best_pr = -np.inf
    best_roc = -np.inf
    best_epoch = 0
    best_state = None
    best_metrics: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    stop_reason = "max_epochs_reached"

    for epoch_index in range(max_epochs):
        start = time.perf_counter()
        momentum = calculate_momentum(epoch_index, initial, final, warmup)
        for group in optimizer.param_groups:
            group["momentum"] = momentum

        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _validation_metrics(model, validation_loader, device, threshold)
        duration = time.perf_counter() - start

        history.append({
            "epoch": epoch_index + 1,
            "train_loss": float(train_loss),
            "momentum": float(momentum),
            **{f"val_{key}": value for key, value in metrics.items()},
            "duration_seconds": float(duration),
        })

        if verbose:
            print(
                f"Epoch {epoch_index + 1:03d} | loss={train_loss:.6f} | "
                f"PR-AUC={metrics['pr_auc']:.6f} | "
                f"ROC-AUC={metrics['roc_auc']:.6f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"pred+={metrics['predicted_positives']} | {duration:.1f}s"
            )

        current_pr = float(metrics["pr_auc"])
        current_roc = float(metrics["roc_auc"])
        improves_pr = current_pr > best_pr + min_delta
        improves_roc_on_tie = (
            abs(current_pr - best_pr) <= min_delta
            and current_roc > best_roc + min_delta
        )

        if improves_pr or improves_roc_on_tie:
            best_pr = current_pr
            best_roc = current_roc
            best_epoch = epoch_index + 1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch_index + 1 >= min_epochs and epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            break

    if best_state is None:
        raise RuntimeError("No se pudo seleccionar un modelo válido.")
    if bool(selection_config.get("restore_best_weights", True)):
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "stop_reason": stop_reason,
    }


def train_final_model(
    model: nn.Module,
    train_loader: DataLoader,
    training_config: Mapping[str, Any],
    epochs: int,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena desde cero sobre todo el conjunto de entrenamiento."""
    if epochs <= 0:
        raise ValueError("epochs debe ser positivo.")

    device = resolve_device(device)
    model.to(device)
    momentum_config = training_config.get("momentum", {})
    initial = float(momentum_config.get("initial", 0.5))
    final = float(momentum_config.get("final", 0.99))
    warmup = int(momentum_config.get("warmup_epochs", 10))
    criterion, optimizer = _training_components(model, training_config)
    history: list[dict[str, float | int]] = []

    for epoch_index in range(epochs):
        start = time.perf_counter()
        momentum = calculate_momentum(epoch_index, initial, final, warmup)
        for group in optimizer.param_groups:
            group["momentum"] = momentum

        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        duration = time.perf_counter() - start
        history.append({
            "epoch": epoch_index + 1,
            "train_loss": float(train_loss),
            "momentum": float(momentum),
            "duration_seconds": float(duration),
        })

        if verbose:
            print(
                f"Epoch {epoch_index + 1:03d}/{epochs:03d} | "
                f"loss={train_loss:.6f} | momentum={momentum:.4f} | "
                f"{duration:.1f}s"
            )

    return {"model": model, "epochs": epochs, "history": history}


def run_classifier_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Mapping[str, Any],
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Interfaz principal: selecciona la época y entrena el modelo final."""
    _validate_xy(X_train, y_train)

    seed = int(config["project"]["seed"])
    model_config = config["classifier_model"]
    training_config = config["classifier_training"]
    loader_config = config.get("data_loader", {})
    validation_ratio = float(config["preprocessing"]["split"]["validation_ratio"])
    device = resolve_device(device)

    expected_features = int(model_config.get("input_dim", 30))
    if X_train.shape[1] != expected_features:
        raise ValueError(
            f"Se esperaban {expected_features} características y se recibieron "
            f"{X_train.shape[1]}."
        )

    X_subtrain, X_val, y_subtrain, y_val = split_train_validation(
        X_train, y_train, validation_ratio, seed
    )

    loader_args = {
        "batch_size": int(training_config["batch_size"]),
        "seed": seed,
        "num_workers": int(loader_config.get("num_workers", 0)),
        "pin_memory": bool(loader_config.get("pin_memory", True)),
        "device": device,
    }
    subtrain_loader = create_dataloader(
        X_subtrain, y_subtrain, shuffle=True, **loader_args
    )
    validation_loader = create_dataloader(
        X_val, y_val, shuffle=False, **loader_args
    )

    selection_model = build_model(model_config, seed, device)
    selection = train_with_validation(
        selection_model,
        subtrain_loader,
        validation_loader,
        training_config,
        device,
        verbose,
    )

    full_train_loader = create_dataloader(
        X_train, y_train, shuffle=True, **loader_args
    )
    final_model = build_model(model_config, seed, device)
    final = train_final_model(
        final_model,
        full_train_loader,
        training_config,
        selection["best_epoch"],
        device,
        verbose,
    )

    return {
        "final_model": final["model"],
        "best_epoch": selection["best_epoch"],
        "best_validation_metrics": selection["best_metrics"],
        "selection_history": selection["history"],
        "final_history": final["history"],
        "stop_reason": selection["stop_reason"],
        "split_summary": {
            "subtrain_samples": int(len(y_subtrain)),
            "subtrain_fraud": int(np.sum(y_subtrain == 1)),
            "validation_samples": int(len(y_val)),
            "validation_fraud": int(np.sum(y_val == 1)),
        },
        "device": str(device),
    }


__all__ = [
    "FraudClassifier",
    "build_model",
    "calculate_momentum",
    "create_dataloader",
    "initialize_model",
    "resolve_device",
    "run_classifier_pipeline",
    "set_seed",
    "split_train_validation",
    "train_final_model",
    "train_with_validation",
]
