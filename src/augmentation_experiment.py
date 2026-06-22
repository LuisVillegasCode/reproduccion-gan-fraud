"""Orquestación de experimentos de aumento con GAN y SMOTE.

Este módulo combina muestras sintéticas con el entrenamiento original, entrena
un clasificador nuevo para cada combinación ``(método, N_g)`` y lo evalúa sobre
la misma partición de prueba. Reutiliza ``sample_generation.py``,
``classifier_pipeline.py`` y ``evaluation.py``; no contiene rutas, persistencia
ni dependencias específicas de Google Colab.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .classifier_pipeline import (
        build_model,
        create_dataloader,
        resolve_device,
        set_seed,
        train_final_model,
    )
    from .evaluation import evaluate_model
    from .sample_generation import (
        SUPPORTED_METHODS,
        generate_samples,
        take_sample_prefix,
        validate_synthetic_samples,
    )
except ImportError:  # Permite importar cuando ``src`` está directamente en sys.path.
    from classifier_pipeline import (
        build_model,
        create_dataloader,
        resolve_device,
        set_seed,
        train_final_model,
    )
    from evaluation import evaluate_model
    from sample_generation import (
        SUPPORTED_METHODS,
        generate_samples,
        take_sample_prefix,
        validate_synthetic_samples,
    )


REQUIRED_METRICS: tuple[str, ...] = (
    "tn",
    "fp",
    "fn",
    "tp",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
)

DELTA_METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "accuracy",
    "fp",
    "fn",
)


def _required(config: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in config:
        raise KeyError(f"Falta el parámetro obligatorio '{section}.{key}'.")
    return config[key]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} debe ser un entero.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} debe ser mayor que cero.")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} debe ser un entero.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} no puede ser negativo.")
    return value


def _validate_binary_dataset(
    X: Any,
    y: Any,
    *,
    name: str,
    expected_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida y normaliza una partición binaria ya preprocesada."""
    features = np.asarray(X)
    labels = np.asarray(y)

    if features.ndim != 2:
        raise ValueError(f"{name}.X debe ser una matriz bidimensional.")
    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError(f"{name}.y debe ser un vector unidimensional.")
    if len(features) == 0 or len(features) != len(labels):
        raise ValueError(
            f"{name}.X y {name}.y deben tener la misma longitud no vacía."
        )
    if features.shape[1] != expected_features:
        raise ValueError(
            f"{name}.X debe tener {expected_features} características; "
            f"se recibieron {features.shape[1]}."
        )
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(f"{name}.X debe contener valores numéricos.")
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError(f"{name}.y debe contener valores numéricos.")

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos.")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{name}.y debe contener etiquetas enteras.")

    labels = labels.astype(np.int64, copy=False)
    unique_labels = np.unique(labels)
    if unique_labels.size != 2 or not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"{name}.y debe contener ambas clases binarias 0 y 1.")

    tolerance = 1e-7
    if features.min() < -tolerance or features.max() > 1.0 + tolerance:
        raise ValueError(f"{name}.X debe estar escalado al intervalo [0, 1].")

    return (
        np.ascontiguousarray(np.clip(features, 0.0, 1.0), dtype=np.float32),
        np.ascontiguousarray(labels, dtype=np.int64),
    )


def _normalize_methods(methods: Sequence[str]) -> list[str]:
    if not methods:
        raise ValueError("Debe indicarse al menos un método de aumento.")

    normalized = [str(method).strip().lower() for method in methods]
    if len(normalized) != len(set(normalized)):
        raise ValueError("La lista de métodos contiene valores repetidos.")

    invalid = [method for method in normalized if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            "Métodos no admitidos: " + ", ".join(invalid) + "."
        )
    return normalized


def _normalize_sample_counts(values: Sequence[Any]) -> list[int]:
    if not values:
        raise ValueError("Debe indicarse al menos una cantidad N_g.")

    counts = [_non_negative_integer(value, "generated_sample_counts") for value in values]
    if len(counts) != len(set(counts)):
        raise ValueError("generated_sample_counts contiene valores repetidos.")

    # N_g=0 corresponde al baseline y no se repite para GAN o SMOTE.
    positive_counts = sorted(value for value in counts if value > 0)
    if not positive_counts:
        raise ValueError("Debe existir al menos una cantidad N_g mayor que cero.")
    return positive_counts


def _normalize_completed_experiments(
    values: Iterable[tuple[str, int, int]] | None,
) -> set[tuple[str, int, int]]:
    completed: set[tuple[str, int, int]] = set()
    if values is None:
        return completed

    for value in values:
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(
                "Cada experimento completado debe ser una tupla "
                "(method, generated_samples, seed)."
            )
        method = str(value[0]).strip().lower()
        generated_samples = _non_negative_integer(value[1], "generated_samples")
        seed = _non_negative_integer(value[2], "seed")
        completed.add((method, generated_samples, seed))
    return completed


def build_augmented_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatena entrenamiento y fraudes sintéticos y mezcla determinísticamente."""
    seed = _non_negative_integer(seed, "seed")
    if X_train.ndim != 2 or y_train.ndim != 1:
        raise ValueError("X_train debe ser 2D e y_train debe ser 1D.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener la misma longitud.")

    synthetic_features, synthetic_labels = validate_synthetic_samples(
        X_synthetic,
        expected_count=len(X_synthetic),
        data_dim=X_train.shape[1],
    )
    provided_labels = np.asarray(y_synthetic)
    if provided_labels.ndim != 1 or len(provided_labels) != len(synthetic_features):
        raise ValueError("y_synthetic debe ser 1D y corresponder a X_synthetic.")
    if len(provided_labels) and not np.all(provided_labels == 1):
        raise ValueError("Todas las muestras sintéticas deben tener etiqueta 1.")
    if not np.array_equal(provided_labels.astype(np.int64), synthetic_labels):
        raise ValueError("y_synthetic contiene etiquetas incompatibles.")

    augmented_X = np.concatenate(
        [np.asarray(X_train, dtype=np.float32), synthetic_features],
        axis=0,
    )
    augmented_y = np.concatenate(
        [np.asarray(y_train, dtype=np.int64), synthetic_labels],
        axis=0,
    )

    permutation = np.random.default_rng(seed).permutation(len(augmented_y))
    augmented_X = np.ascontiguousarray(augmented_X[permutation], dtype=np.float32)
    augmented_y = np.ascontiguousarray(augmented_y[permutation], dtype=np.int64)

    expected_fraud = int(np.count_nonzero(y_train == 1) + len(synthetic_labels))
    expected_legitimate = int(np.count_nonzero(y_train == 0))
    if int(np.count_nonzero(augmented_y == 1)) != expected_fraud:
        raise RuntimeError("El conjunto aumentado perdió muestras fraudulentas.")
    if int(np.count_nonzero(augmented_y == 0)) != expected_legitimate:
        raise RuntimeError("El conjunto aumentado alteró la clase legítima.")

    return augmented_X, augmented_y


def _create_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    shuffle: bool,
    device: str | torch.device,
):
    training_config = _required(config, "classifier_training", "config")
    loader_config = config.get("data_loader", {})
    return create_dataloader(
        X,
        y,
        batch_size=int(_required(training_config, "batch_size", "classifier_training")),
        shuffle=shuffle,
        seed=seed,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        device=device,
    )


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: Mapping[str, Any],
    epochs: int,
    seed: int,
    device: str | torch.device,
    include_probability_diagnostics: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Entrena un clasificador nuevo y lo evalúa una sola vez sobre test."""
    epochs = _positive_integer(epochs, "epochs")
    resolved_device = resolve_device(device)
    model_config = _required(config, "classifier_model", "config")
    training_config = _required(config, "classifier_training", "config")

    set_seed(seed)
    train_loader = _create_loader(
        X_train,
        y_train,
        config=config,
        seed=seed,
        shuffle=True,
        device=resolved_device,
    )
    model = build_model(model_config, seed=seed, device=resolved_device)

    training_started = time.perf_counter()
    training_result = train_final_model(
        model,
        train_loader,
        training_config,
        epochs=epochs,
        device=resolved_device,
        verbose=verbose,
    )
    training_duration = time.perf_counter() - training_started

    test_loader = _create_loader(
        X_test,
        y_test,
        config=config,
        seed=seed,
        shuffle=False,
        device=resolved_device,
    )
    evaluation_started = time.perf_counter()
    metrics = evaluate_model(
        training_result["model"],
        test_loader,
        device=resolved_device,
        require_both_classes=True,
        include_probability_diagnostics=include_probability_diagnostics,
        return_predictions=False,
    )
    evaluation_duration = time.perf_counter() - evaluation_started

    for metric_name in REQUIRED_METRICS:
        if metric_name not in metrics:
            raise RuntimeError(f"La evaluación no devolvió '{metric_name}'.")
        value = metrics[metric_name]
        if isinstance(value, Real) and not np.isfinite(float(value)):
            raise RuntimeError(f"La métrica '{metric_name}' no es finita.")

    return {
        "model": training_result["model"],
        "history": training_result["history"],
        "metrics": metrics,
        "training_duration_seconds": float(training_duration),
        "evaluation_duration_seconds": float(evaluation_duration),
    }


def _base_result_row(
    *,
    method: str,
    generated_samples: int,
    seed: int,
    epochs: int,
    original_legitimate: int,
    original_fraud: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "generated_samples": int(generated_samples),
        "minority_original": int(original_fraud),
        "minority_augmented": int(original_fraud + generated_samples),
        "majority_samples": int(original_legitimate),
        "total_training_samples": int(
            original_legitimate + original_fraud + generated_samples
        ),
        "epochs": int(epochs),
        "seed": int(seed),
    }


def compute_baseline_deltas(
    row: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Añade diferencias con signo respecto al baseline."""
    output = dict(row)
    for metric in DELTA_METRICS:
        if metric not in row or metric not in baseline_metrics:
            output[f"delta_{metric}"] = None
            continue
        current = row[metric]
        baseline = baseline_metrics[metric]
        if current is None or baseline is None:
            output[f"delta_{metric}"] = None
        else:
            output[f"delta_{metric}"] = float(current) - float(baseline)
    return output


def _baseline_row_from_metrics(
    metrics: Mapping[str, Any],
    *,
    epochs: int,
    seed: int,
    original_legitimate: int,
    original_fraud: int,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise KeyError(
            "Las métricas baseline no contienen: " + ", ".join(missing) + "."
        )

    row = _base_result_row(
        method="baseline",
        generated_samples=0,
        seed=seed,
        epochs=epochs,
        original_legitimate=original_legitimate,
        original_fraud=original_fraud,
    )
    row.update({key: metrics[key] for key in metrics})
    row.update({
        "generation_duration_seconds": 0.0,
        "training_duration_seconds": None,
        "evaluation_duration_seconds": None,
        "total_duration_seconds": None,
        "synthetic_unique_ratio": None,
        "status": "completed",
        "error_message": None,
    })
    return compute_baseline_deltas(row, metrics)


def run_single_augmentation_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    synthetic_result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    epochs: int,
    seed: int,
    device: str | torch.device = "auto",
    baseline_metrics: Mapping[str, Any] | None = None,
    include_probability_diagnostics: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta entrenamiento y evaluación para una reserva sintética concreta."""
    required = {"method", "X_synthetic", "y_synthetic", "generated_samples"}
    missing = required.difference(synthetic_result)
    if missing:
        raise KeyError(
            "Faltan claves en synthetic_result: " + ", ".join(sorted(missing))
        )

    method = str(synthetic_result["method"]).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Método sintético no admitido: '{method}'.")

    generated_samples = _non_negative_integer(
        synthetic_result["generated_samples"],
        "generated_samples",
    )
    X_synthetic = np.asarray(synthetic_result["X_synthetic"])
    y_synthetic = np.asarray(synthetic_result["y_synthetic"])
    if len(X_synthetic) != generated_samples:
        raise ValueError("generated_samples no coincide con X_synthetic.")

    experiment_started = time.perf_counter()
    augmented_X, augmented_y = build_augmented_dataset(
        X_train,
        y_train,
        X_synthetic,
        y_synthetic,
        seed=seed,
    )
    trained = _train_and_evaluate(
        augmented_X,
        augmented_y,
        X_test,
        y_test,
        config=config,
        epochs=epochs,
        seed=seed,
        device=device,
        include_probability_diagnostics=include_probability_diagnostics,
        verbose=verbose,
    )
    total_duration = time.perf_counter() - experiment_started

    original_legitimate = int(np.count_nonzero(y_train == 0))
    original_fraud = int(np.count_nonzero(y_train == 1))
    row = _base_result_row(
        method=method,
        generated_samples=generated_samples,
        seed=seed,
        epochs=epochs,
        original_legitimate=original_legitimate,
        original_fraud=original_fraud,
    )
    row.update(trained["metrics"])
    diagnostics = synthetic_result.get("diagnostics", {})
    row.update({
        "generation_duration_seconds": float(
            synthetic_result.get("generation_duration_seconds", 0.0)
        ),
        "training_duration_seconds": trained["training_duration_seconds"],
        "evaluation_duration_seconds": trained["evaluation_duration_seconds"],
        "total_duration_seconds": float(total_duration),
        "synthetic_unique_ratio": diagnostics.get("unique_ratio"),
        "status": "completed",
        "error_message": None,
    })

    if baseline_metrics is not None:
        row = compute_baseline_deltas(row, baseline_metrics)

    return {
        "row": row,
        "history": trained["history"],
        "model": trained["model"],
    }


def _error_row(
    *,
    method: str,
    generated_samples: int,
    seed: int,
    epochs: int,
    original_legitimate: int,
    original_fraud: int,
    error: Exception,
) -> dict[str, Any]:
    row = _base_result_row(
        method=method,
        generated_samples=generated_samples,
        seed=seed,
        epochs=epochs,
        original_legitimate=original_legitimate,
        original_fraud=original_fraud,
    )
    for metric in REQUIRED_METRICS:
        row[metric] = None
    row.update({
        "generation_duration_seconds": None,
        "training_duration_seconds": None,
        "evaluation_duration_seconds": None,
        "total_duration_seconds": None,
        "synthetic_unique_ratio": None,
        "status": "failed",
        "error_message": f"{type(error).__name__}: {error}",
    })
    for metric in DELTA_METRICS:
        row[f"delta_{metric}"] = None
    return row


def _release_model(model: nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_experiment_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Resume los mejores experimentos completados según F1."""
    completed = [
        dict(row)
        for row in rows
        if row.get("status") == "completed"
        and row.get("f1") is not None
        and np.isfinite(float(row["f1"]))
    ]

    def best_for(method: str) -> dict[str, Any] | None:
        candidates = [row for row in completed if row.get("method") == method]
        return max(candidates, key=lambda row: float(row["f1"])) if candidates else None

    augmented = [row for row in completed if row.get("method") in SUPPORTED_METHODS]
    return {
        "completed_experiments": len(completed),
        "failed_experiments": sum(row.get("status") == "failed" for row in rows),
        "baseline": best_for("baseline"),
        "best_gan_by_f1": best_for("gan"),
        "best_smote_by_f1": best_for("smote"),
        "best_augmented_by_f1": (
            max(augmented, key=lambda row: float(row["f1"])) if augmented else None
        ),
    }


def run_augmentation_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Mapping[str, Any],
    *,
    baseline_epochs: int,
    generator: nn.Module | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
    methods: Sequence[str] | None = None,
    sample_counts: Sequence[int] | None = None,
    device: str | torch.device = "auto",
    include_baseline: bool = True,
    include_probability_diagnostics: bool = False,
    keep_histories: bool = True,
    keep_models: bool = False,
    completed_experiments: Iterable[tuple[str, int, int]] | None = None,
    on_experiment_end: Callable[[Mapping[str, Any]], None] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta baseline, GAN y SMOTE para las cantidades configuradas.

    ``baseline_epochs`` debe proceder de la selección realizada por el baseline.
    Cuando ``baseline_metrics`` se proporciona, se reutiliza sin volver a entrenar
    el baseline. De forma predeterminada solo se mantiene un modelo activo por vez.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config debe ser un mapeo.")
    baseline_epochs = _positive_integer(baseline_epochs, "baseline_epochs")

    project_config = _required(config, "project", "config")
    classifier_model = _required(config, "classifier_model", "config")
    augmentation_config = _required(config, "augmentation", "config")
    seed = _non_negative_integer(_required(project_config, "seed", "project"), "seed")
    data_dim = _positive_integer(
        classifier_model.get("input_dim", 30),
        "classifier_model.input_dim",
    )

    X_train, y_train = _validate_binary_dataset(
        X_train,
        y_train,
        name="train",
        expected_features=data_dim,
    )
    X_test, y_test = _validate_binary_dataset(
        X_test,
        y_test,
        name="test",
        expected_features=data_dim,
    )

    selected_methods = _normalize_methods(
        methods if methods is not None else _required(
            augmentation_config,
            "methods",
            "augmentation",
        )
    )
    selected_counts = _normalize_sample_counts(
        sample_counts if sample_counts is not None else _required(
            augmentation_config,
            "generated_sample_counts",
            "augmentation",
        )
    )
    completed = _normalize_completed_experiments(completed_experiments)

    reuse_pool = bool(augmentation_config.get("reuse_maximum_sample_pool", True))
    generation_batch_size = _positive_integer(
        augmentation_config.get("gan_generation_batch_size", 1024),
        "augmentation.gan_generation_batch_size",
    )
    smote_k_neighbors = _positive_integer(
        augmentation_config.get("smote_k_neighbors", 5),
        "augmentation.smote_k_neighbors",
    )
    fail_fast = bool(augmentation_config.get("fail_fast", True))

    if "gan" in selected_methods:
        if generator is None:
            raise ValueError("generator es obligatorio para los experimentos GAN.")
        gan_model_config = _required(config, "gan_model", "config")
        gan_training_config = _required(config, "gan_training", "config")
        noise_config = _required(gan_training_config, "noise", "gan_training")
        noise_dim = _positive_integer(
            gan_model_config.get("noise_dim", 100),
            "gan_model.noise_dim",
        )
    else:
        noise_config = None
        noise_dim = None

    resolved_device = resolve_device(device)
    original_legitimate = int(np.count_nonzero(y_train == 0))
    original_fraud = int(np.count_nonzero(y_train == 1))

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    models: dict[str, nn.Module] = {}
    pools: dict[str, Mapping[str, Any]] = {}

    effective_baseline_metrics: Mapping[str, Any] | None = baseline_metrics

    # El baseline se evalúa una sola vez y siempre antes de los experimentos aumentados.
    baseline_key = ("baseline", 0, seed)
    if include_baseline and baseline_key not in completed:
        if baseline_metrics is not None:
            baseline_row = _baseline_row_from_metrics(
                baseline_metrics,
                epochs=baseline_epochs,
                seed=seed,
                original_legitimate=original_legitimate,
                original_fraud=original_fraud,
            )
            rows.append(baseline_row)
            if on_experiment_end is not None:
                on_experiment_end({"row": baseline_row, "history": None, "model": None})
        else:
            baseline_started = time.perf_counter()
            try:
                trained = _train_and_evaluate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    config=config,
                    epochs=baseline_epochs,
                    seed=seed,
                    device=resolved_device,
                    include_probability_diagnostics=include_probability_diagnostics,
                    verbose=verbose,
                )
            except Exception as error:
                if fail_fast:
                    raise
                row = _error_row(
                    method="baseline",
                    generated_samples=0,
                    seed=seed,
                    epochs=baseline_epochs,
                    original_legitimate=original_legitimate,
                    original_fraud=original_fraud,
                    error=error,
                )
                rows.append(row)
                if on_experiment_end is not None:
                    on_experiment_end({"row": row, "history": None, "model": None})
            else:
                effective_baseline_metrics = trained["metrics"]
                baseline_row = _base_result_row(
                    method="baseline",
                    generated_samples=0,
                    seed=seed,
                    epochs=baseline_epochs,
                    original_legitimate=original_legitimate,
                    original_fraud=original_fraud,
                )
                baseline_row.update(trained["metrics"])
                baseline_row.update({
                    "generation_duration_seconds": 0.0,
                    "training_duration_seconds": trained["training_duration_seconds"],
                    "evaluation_duration_seconds": trained["evaluation_duration_seconds"],
                    "total_duration_seconds": float(time.perf_counter() - baseline_started),
                    "synthetic_unique_ratio": None,
                    "status": "completed",
                    "error_message": None,
                })
                baseline_row = compute_baseline_deltas(
                    baseline_row,
                    effective_baseline_metrics,
                )
                rows.append(baseline_row)
                key = f"baseline:0:{seed}"
                if keep_histories:
                    histories[key] = trained["history"]
                if keep_models:
                    models[key] = trained["model"]
                if on_experiment_end is not None:
                    on_experiment_end({
                        "row": baseline_row,
                        "history": trained["history"],
                        "model": trained["model"],
                    })
                if not keep_models:
                    _release_model(trained["model"])

    # Si se reutilizarán prefijos, se crea una sola reserva máxima por método.
    if reuse_pool:
        maximum_count = max(selected_counts)
        for method in selected_methods:
            pending = [
                count
                for count in selected_counts
                if (method, count, seed) not in completed
            ]
            if not pending:
                continue
            try:
                generation_started = time.perf_counter()
                pool = generate_samples(
                    method,
                    n_samples=maximum_count,
                    seed=seed,
                    generator=generator,
                    X_train=X_train,
                    y_train=y_train,
                    noise_config=noise_config,
                    generation_batch_size=generation_batch_size,
                    noise_dim=noise_dim,
                    data_dim=data_dim,
                    device=resolved_device,
                    smote_k_neighbors=smote_k_neighbors,
                    include_diagnostics=True,
                )
                pool = dict(pool)
                pool["generation_duration_seconds"] = float(
                    time.perf_counter() - generation_started
                )
                pools[method] = pool
            except Exception as error:
                if fail_fast:
                    raise
                for count in pending:
                    row = _error_row(
                        method=method,
                        generated_samples=count,
                        seed=seed,
                        epochs=baseline_epochs,
                        original_legitimate=original_legitimate,
                        original_fraud=original_fraud,
                        error=error,
                    )
                    rows.append(row)
                    if on_experiment_end is not None:
                        on_experiment_end({"row": row, "history": None, "model": None})

    for method in selected_methods:
        if reuse_pool and method not in pools:
            continue

        for count in selected_counts:
            experiment_key = (method, count, seed)
            if experiment_key in completed:
                continue

            model: nn.Module | None = None
            try:
                if reuse_pool:
                    synthetic_result = take_sample_prefix(
                        pools[method],
                        n_samples=count,
                        include_diagnostics=True,
                    )
                    synthetic_result = dict(synthetic_result)
                    # La reserva se genera una sola vez; seleccionar un prefijo
                    # no añade un coste de generación relevante por experimento.
                    synthetic_result["generation_duration_seconds"] = 0.0
                else:
                    generation_started = time.perf_counter()
                    synthetic_result = generate_samples(
                        method,
                        n_samples=count,
                        seed=seed,
                        generator=generator,
                        X_train=X_train,
                        y_train=y_train,
                        noise_config=noise_config,
                        generation_batch_size=generation_batch_size,
                        noise_dim=noise_dim,
                        data_dim=data_dim,
                        device=resolved_device,
                        smote_k_neighbors=smote_k_neighbors,
                        include_diagnostics=True,
                    )
                    synthetic_result = dict(synthetic_result)
                    synthetic_result["generation_duration_seconds"] = float(
                        time.perf_counter() - generation_started
                    )

                result = run_single_augmentation_experiment(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    synthetic_result,
                    config=config,
                    epochs=baseline_epochs,
                    seed=seed,
                    device=resolved_device,
                    baseline_metrics=effective_baseline_metrics,
                    include_probability_diagnostics=include_probability_diagnostics,
                    verbose=verbose,
                )
            except Exception as error:
                if model is not None and not keep_models:
                    _release_model(model)
                if fail_fast:
                    raise
                row = _error_row(
                    method=method,
                    generated_samples=count,
                    seed=seed,
                    epochs=baseline_epochs,
                    original_legitimate=original_legitimate,
                    original_fraud=original_fraud,
                    error=error,
                )
                rows.append(row)
                if on_experiment_end is not None:
                    on_experiment_end({"row": row, "history": None, "model": None})
            else:
                row = result["row"]
                model = result["model"]
                rows.append(row)

                key = f"{method}:{count}:{seed}"
                if keep_histories:
                    histories[key] = result["history"]
                if keep_models:
                    models[key] = model
                if on_experiment_end is not None:
                    on_experiment_end(result)
                if not keep_models:
                    _release_model(model)
                    model = None

    return {
        "results": rows,
        "histories": histories,
        "models": models,
        "summary": summarize_experiment_results(rows),
        "metadata": {
            "seed": seed,
            "baseline_epochs": baseline_epochs,
            "methods": selected_methods,
            "generated_sample_counts": selected_counts,
            "reuse_maximum_sample_pool": reuse_pool,
            "same_test_partition": True,
            "classifier_reinitialized_for_each_experiment": True,
        },
    }


__all__ = [
    "build_augmented_dataset",
    "compute_baseline_deltas",
    "run_augmentation_experiments",
    "run_single_augmentation_experiment",
    "summarize_experiment_results",
]
