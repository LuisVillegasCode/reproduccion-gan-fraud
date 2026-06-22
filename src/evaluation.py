"""Evaluación reutilizable para clasificadores binarios de fraude.

Este módulo concentra la inferencia, el cálculo de métricas y la comparación
con resultados de referencia. No contiene rutas, persistencia, gráficos ni
dependencias específicas de Google Colab.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


PAPER_COMPARABLE_METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "accuracy",
)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resuelve el dispositivo sin asumir que CUDA está disponible."""
    if isinstance(device, torch.device):
        resolved = device
    else:
        name = str(device).lower().strip()
        if name == "auto":
            resolved = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif name in {"cpu", "cuda"}:
            resolved = torch.device(name)
        else:
            raise ValueError(
                "device debe ser 'auto', 'cpu', 'cuda' o torch.device."
            )

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible.")

    return resolved


def _as_one_dimensional_array(
    values: Any,
    name: str,
) -> np.ndarray:
    """Convierte una entrada compatible a un vector NumPy unidimensional."""
    array = np.asarray(values)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)

    if array.ndim != 1:
        raise ValueError(f"{name} debe ser un vector unidimensional.")

    if array.size == 0:
        raise ValueError(f"{name} no puede estar vacío.")

    return array


def _validate_binary_labels(
    labels: np.ndarray,
    name: str,
    require_both_classes: bool,
) -> np.ndarray:
    """Valida etiquetas binarias y devuelve int64."""
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"{name} debe contener valores numéricos.")

    if not np.isfinite(labels).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos.")

    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{name} debe contener etiquetas enteras.")

    labels = labels.astype(np.int64, copy=False)
    unique_labels = np.unique(labels)

    if not np.isin(unique_labels, [0, 1]).all():
        raise ValueError(f"{name} debe contener únicamente las etiquetas 0 y 1.")

    if require_both_classes and unique_labels.size != 2:
        raise ValueError(
            f"{name} debe contener ambas clases para calcular ROC-AUC y PR-AUC."
        )

    return labels


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    """Valida probabilidades de la clase fraude."""
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("y_score debe contener valores numéricos.")

    scores = scores.astype(np.float64, copy=False)

    if not np.isfinite(scores).all():
        raise ValueError("y_score contiene NaN o valores infinitos.")

    tolerance = 1e-7
    if scores.min() < -tolerance or scores.max() > 1.0 + tolerance:
        raise ValueError("y_score debe permanecer en el intervalo [0, 1].")

    return np.clip(scores, 0.0, 1.0)


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "auto",
) -> dict[str, np.ndarray]:
    """Ejecuta inferencia y devuelve etiquetas, predicciones y scores.

    La predicción oficial utiliza ``argmax`` sobre dos logits. ``y_score``
    corresponde a la probabilidad Softmax de la clase fraude (índice 1).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model debe ser una instancia de torch.nn.Module.")

    resolved_device = resolve_device(device)
    model.to(resolved_device)

    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    previous_training_state = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(data_loader):
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        "Cada lote debe contener al menos características y etiquetas."
                    )

                features, labels = batch[0], batch[1]

                if not isinstance(features, torch.Tensor):
                    raise TypeError(
                        f"Las características del lote {batch_index} "
                        "deben ser torch.Tensor."
                    )
                if not isinstance(labels, torch.Tensor):
                    raise TypeError(
                        f"Las etiquetas del lote {batch_index} "
                        "deben ser torch.Tensor."
                    )

                features = features.to(
                    resolved_device,
                    non_blocking=True,
                )

                logits = model(features)

                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("El modelo no devolvió un tensor de logits.")

                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise RuntimeError(
                        "El modelo debe devolver una matriz de forma "
                        "(n_muestras, 2)."
                    )

                if logits.shape[0] != labels.shape[0]:
                    raise RuntimeError(
                        "La cantidad de logits no coincide con las etiquetas del lote."
                    )

                if not torch.isfinite(logits).all():
                    raise RuntimeError("El modelo produjo logits no finitos.")

                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                true_parts.append(
                    labels.detach().cpu().numpy().reshape(-1)
                )
                predicted_parts.append(
                    predictions.detach().cpu().numpy().reshape(-1)
                )
                score_parts.append(
                    probabilities[:, 1].detach().cpu().numpy().reshape(-1)
                )
    finally:
        model.train(previous_training_state)

    if not true_parts:
        raise ValueError("El DataLoader no produjo ningún lote.")

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(predicted_parts)
    y_score = np.concatenate(score_parts)

    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise RuntimeError(
            "La inferencia produjo cantidades incompatibles de resultados."
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def compute_probability_diagnostics(
    y_true: Any,
    y_score: Any,
) -> dict[str, float]:
    """Calcula diagnósticos opcionales compatibles con el notebook."""
    true_labels = _as_one_dimensional_array(y_true, "y_true")
    scores = _as_one_dimensional_array(y_score, "y_score")

    if len(true_labels) != len(scores):
        raise ValueError("y_true e y_score deben tener la misma longitud.")

    true_labels = _validate_binary_labels(
        true_labels,
        "y_true",
        require_both_classes=True,
    )
    scores = _validate_scores(scores)

    fraud_scores = scores[true_labels == 1]
    legitimate_scores = scores[true_labels == 0]

    return {
        "max_p_fraud": float(np.max(scores)),
        "fraud_p25": float(np.quantile(fraud_scores, 0.25)),
        "fraud_p50": float(np.quantile(fraud_scores, 0.50)),
        "fraud_p75": float(np.quantile(fraud_scores, 0.75)),
        "fraud_p90": float(np.quantile(fraud_scores, 0.90)),
        "legitimate_p99": float(np.quantile(legitimate_scores, 0.99)),
    }


def compute_binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    *,
    require_both_classes: bool = True,
    include_probability_diagnostics: bool = False,
) -> dict[str, int | float | None]:
    """Calcula métricas binarias usando fraude (1) como clase positiva."""
    true_labels = _as_one_dimensional_array(y_true, "y_true")
    predicted_labels = _as_one_dimensional_array(y_pred, "y_pred")
    scores = _as_one_dimensional_array(y_score, "y_score")

    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(scores)
    ):
        raise ValueError(
            "y_true, y_pred e y_score deben tener la misma longitud."
        )

    true_labels = _validate_binary_labels(
        true_labels,
        "y_true",
        require_both_classes=require_both_classes,
    )
    predicted_labels = _validate_binary_labels(
        predicted_labels,
        "y_pred",
        require_both_classes=False,
    )
    scores = _validate_scores(scores)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    actual_positives = int(tp + fn)
    actual_negatives = int(tn + fp)
    predicted_positives = int(tp + fp)
    predicted_negatives = int(tn + fn)
    total_samples = int(len(true_labels))

    sensitivity = (
        float(tp / actual_positives)
        if actual_positives > 0
        else 0.0
    )
    specificity = (
        float(tn / actual_negatives)
        if actual_negatives > 0
        else 0.0
    )
    fpr = (
        float(fp / actual_negatives)
        if actual_negatives > 0
        else 0.0
    )
    fnr = (
        float(fn / actual_positives)
        if actual_positives > 0
        else 0.0
    )

    unique_true = np.unique(true_labels)
    if unique_true.size == 2:
        roc_auc: float | None = float(
            roc_auc_score(true_labels, scores)
        )
        pr_auc: float | None = float(
            average_precision_score(true_labels, scores)
        )
    else:
        if require_both_classes:
            raise ValueError(
                "Se requieren ambas clases para calcular ROC-AUC y PR-AUC."
            )
        roc_auc = None
        pr_auc = None

    metrics: dict[str, int | float | None] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": total_samples,
        "actual_positives": actual_positives,
        "actual_negatives": actual_negatives,
        "predicted_positives": predicted_positives,
        "predicted_negatives": predicted_negatives,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    if include_probability_diagnostics:
        metrics.update(
            compute_probability_diagnostics(
                true_labels,
                scores,
            )
        )

    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "auto",
    *,
    require_both_classes: bool = True,
    include_probability_diagnostics: bool = False,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evalúa un modelo y devuelve métricas estructuradas.

    Con ``return_predictions=False`` devuelve directamente el diccionario de
    métricas. Si se activa, devuelve ``{"metrics": ..., "predictions": ...}``.
    """
    predictions = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    metrics = compute_binary_metrics(
        y_true=predictions["y_true"],
        y_pred=predictions["y_pred"],
        y_score=predictions["y_score"],
        require_both_classes=require_both_classes,
        include_probability_diagnostics=include_probability_diagnostics,
    )

    if return_predictions:
        return {
            "metrics": metrics,
            "predictions": predictions,
        }

    return metrics


def compare_with_reference(
    reproduction_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    metrics: Sequence[str] = PAPER_COMPARABLE_METRICS,
) -> list[dict[str, float | str]]:
    """Compara una reproducción con métricas externas de referencia."""
    if not metrics:
        raise ValueError("Debe indicarse al menos una métrica para comparar.")

    comparison: list[dict[str, float | str]] = []

    for metric in metrics:
        if metric not in reproduction_metrics:
            raise KeyError(
                f"La reproducción no contiene la métrica '{metric}'."
            )
        if metric not in reference_metrics:
            raise KeyError(
                f"La referencia no contiene la métrica '{metric}'."
            )

        reproduction_value = reproduction_metrics[metric]
        reference_value = reference_metrics[metric]

        if (
            isinstance(reproduction_value, bool)
            or not isinstance(reproduction_value, Real)
        ):
            raise TypeError(
                f"El valor reproducido de '{metric}' debe ser numérico."
            )
        if (
            isinstance(reference_value, bool)
            or not isinstance(reference_value, Real)
        ):
            raise TypeError(
                f"El valor de referencia de '{metric}' debe ser numérico."
            )

        reproduction_float = float(reproduction_value)
        reference_float = float(reference_value)

        if not (
            np.isfinite(reproduction_float)
            and np.isfinite(reference_float)
        ):
            raise ValueError(
                f"Los valores de '{metric}' deben ser finitos."
            )

        difference = reproduction_float - reference_float

        comparison.append({
            "metric": str(metric),
            "paper": reference_float,
            "reproduction": reproduction_float,
            "difference": float(difference),
            "absolute_difference": float(abs(difference)),
        })

    return comparison


__all__ = [
    "PAPER_COMPARABLE_METRICS",
    "collect_predictions",
    "compare_with_reference",
    "compute_binary_metrics",
    "compute_probability_diagnostics",
    "evaluate_model",
    "resolve_device",
]
