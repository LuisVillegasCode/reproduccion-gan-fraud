"""Generación de muestras sintéticas fraudulentas mediante GAN y SMOTE.

El módulo ofrece una interfaz común para producir exactamente ``N_g`` muestras
fraudulentas sintéticas. No entrena modelos, no combina conjuntos, no evalúa
clasificadores y no contiene rutas ni dependencias específicas de Google Colab.

La generación GAN reutiliza el generador entrenado por ``gan_pipeline.py`` y la
misma configuración de ruido utilizada durante su entrenamiento. La generación
SMOTE utiliza la implementación estándar de ``imbalanced-learn``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

try:
    from .gan_pipeline import resolve_device, sample_noise
except ImportError:  # Permite importar el módulo cuando ``src`` está en sys.path.
    from gan_pipeline import resolve_device, sample_noise


SUPPORTED_METHODS: tuple[str, ...] = ("gan", "smote")


def _validate_non_negative_integer(value: Any, name: str) -> int:
    """Valida un entero no negativo, rechazando booleanos."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} debe ser un número entero.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} debe ser mayor o igual que cero.")
    return value


def _validate_positive_integer(value: Any, name: str) -> int:
    """Valida un entero estrictamente positivo."""
    value = _validate_non_negative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} debe ser mayor que cero.")
    return value


def _validate_seed(seed: Any) -> int:
    seed = _validate_non_negative_integer(seed, "seed")
    if seed > 2**63 - 1:
        raise ValueError("seed excede el rango admitido por PyTorch.")
    return seed


def _validate_feature_matrix(
    features: Any,
    *,
    name: str,
    data_dim: int,
    allow_empty: bool = False,
) -> np.ndarray:
    """Valida y normaliza una matriz de características escaladas."""
    data_dim = _validate_positive_integer(data_dim, "data_dim")
    array = np.asarray(features)

    if array.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz bidimensional.")
    if array.shape[1] != data_dim:
        raise ValueError(
            f"{name} debe contener {data_dim} características; "
            f"se recibieron {array.shape[1]}."
        )
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{name} no puede estar vacío.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} debe contener valores numéricos.")

    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos.")

    tolerance = 1e-7
    if array.size and (
        float(array.min()) < -tolerance
        or float(array.max()) > 1.0 + tolerance
    ):
        raise ValueError(f"{name} debe estar escalado al intervalo [0, 1].")

    return np.ascontiguousarray(
        np.clip(array, 0.0, 1.0),
        dtype=np.float32,
    )


def _validate_training_data(
    X_train: Any,
    y_train: Any,
    *,
    data_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida el conjunto completo requerido por SMOTE."""
    features = _validate_feature_matrix(
        X_train,
        name="X_train",
        data_dim=data_dim,
    )
    labels = np.asarray(y_train)

    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError("y_train debe ser un vector unidimensional.")
    if len(labels) != len(features):
        raise ValueError("X_train e y_train deben tener la misma longitud.")
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError("y_train debe contener valores numéricos.")

    labels = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("y_train contiene NaN o valores infinitos.")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("y_train debe contener etiquetas enteras.")

    labels = labels.astype(np.int64, copy=False)
    unique_labels = np.unique(labels)
    if not np.isin(unique_labels, [0, 1]).all():
        raise ValueError("y_train debe contener únicamente las etiquetas 0 y 1.")
    if unique_labels.size != 2:
        raise ValueError("y_train debe contener ambas clases para aplicar SMOTE.")

    return features, np.ascontiguousarray(labels, dtype=np.int64)


def validate_synthetic_samples(
    samples: Any,
    *,
    expected_count: int,
    data_dim: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida muestras sintéticas y crea sus etiquetas fraudulentas.

    Devuelve una matriz ``float32`` y un vector ``int64`` compuesto por unos.
    Solo se recortan desviaciones numéricas mínimas alrededor de ``[0, 1]``.
    """
    expected_count = _validate_non_negative_integer(
        expected_count,
        "expected_count",
    )
    data_dim = _validate_positive_integer(data_dim, "data_dim")

    features = _validate_feature_matrix(
        samples,
        name="samples",
        data_dim=data_dim,
        allow_empty=True,
    )

    if features.shape[0] != expected_count:
        raise ValueError(
            f"Se solicitaron {expected_count} muestras, pero se obtuvieron "
            f"{features.shape[0]}."
        )

    labels = np.ones(expected_count, dtype=np.int64)
    return features, labels


def summarize_synthetic_samples(
    samples: Any,
    *,
    data_dim: int = 30,
) -> dict[str, int | float | None]:
    """Calcula diagnósticos descriptivos sin filtrar ni modificar muestras."""
    features = _validate_feature_matrix(
        samples,
        name="samples",
        data_dim=data_dim,
        allow_empty=True,
    )
    sample_count = int(features.shape[0])

    if sample_count == 0:
        return {
            "sample_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "unique_rows": 0,
            "duplicate_rows": 0,
            "unique_ratio": None,
        }

    unique_rows = int(np.unique(features, axis=0).shape[0])
    return {
        "sample_count": sample_count,
        "minimum": float(features.min()),
        "maximum": float(features.max()),
        "mean": float(features.mean()),
        "standard_deviation": float(features.std()),
        "unique_rows": unique_rows,
        "duplicate_rows": int(sample_count - unique_rows),
        "unique_ratio": float(unique_rows / sample_count),
    }


def _build_result(
    *,
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    requested_samples: int,
    seed: int,
    include_diagnostics: bool,
) -> dict[str, Any]:
    """Construye una salida común para GAN y SMOTE."""
    result: dict[str, Any] = {
        "method": method,
        "X_synthetic": features,
        "y_synthetic": labels,
        "requested_samples": int(requested_samples),
        "generated_samples": int(len(features)),
        "feature_dimension": int(features.shape[1]),
        "seed": int(seed),
    }

    if include_diagnostics:
        result["diagnostics"] = summarize_synthetic_samples(
            features,
            data_dim=features.shape[1],
        )

    return result


def generate_gan_samples(
    generator: nn.Module,
    *,
    n_samples: int,
    noise_config: Mapping[str, Any],
    seed: int,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
    noise_dim: int | None = None,
    data_dim: int | None = None,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Genera exactamente ``n_samples`` fraudes con un generador entrenado.

    El generador no se inicializa ni se entrena. La función usa el mismo formato
    de ``noise_config`` que ``gan_pipeline.sample_noise``.
    """
    if not isinstance(generator, nn.Module):
        raise TypeError("generator debe ser una instancia de torch.nn.Module.")
    if not isinstance(noise_config, Mapping):
        raise TypeError("noise_config debe ser un mapeo de configuración.")

    n_samples = _validate_non_negative_integer(n_samples, "n_samples")
    seed = _validate_seed(seed)
    batch_size = _validate_positive_integer(batch_size, "batch_size")

    inferred_noise_dim = getattr(generator, "noise_dim", None)
    inferred_data_dim = getattr(generator, "data_dim", None)

    if noise_dim is None:
        if inferred_noise_dim is None:
            raise ValueError(
                "noise_dim debe proporcionarse cuando el generador no expone "
                "el atributo 'noise_dim'."
            )
        noise_dim = inferred_noise_dim

    if data_dim is None:
        if inferred_data_dim is None:
            raise ValueError(
                "data_dim debe proporcionarse cuando el generador no expone "
                "el atributo 'data_dim'."
            )
        data_dim = inferred_data_dim

    noise_dim = _validate_positive_integer(noise_dim, "noise_dim")
    data_dim = _validate_positive_integer(data_dim, "data_dim")

    if inferred_noise_dim is not None and int(inferred_noise_dim) != noise_dim:
        raise ValueError(
            "noise_dim no coincide con la dimensión esperada por el generador."
        )
    if inferred_data_dim is not None and int(inferred_data_dim) != data_dim:
        raise ValueError(
            "data_dim no coincide con la dimensión de salida del generador."
        )

    if n_samples == 0:
        features, labels = validate_synthetic_samples(
            np.empty((0, data_dim), dtype=np.float32),
            expected_count=0,
            data_dim=data_dim,
        )
        return _build_result(
            method="gan",
            features=features,
            labels=labels,
            requested_samples=0,
            seed=seed,
            include_diagnostics=include_diagnostics,
        )

    resolved_device = resolve_device(device)
    generator.to(resolved_device)

    random_device = "cuda" if resolved_device.type == "cuda" else "cpu"
    random_generator = torch.Generator(device=random_device).manual_seed(seed)

    previous_training_state = generator.training
    generated_batches: list[np.ndarray] = []

    generator.eval()
    try:
        with torch.inference_mode():
            remaining = n_samples
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                noise = sample_noise(
                    current_batch_size,
                    noise_dim,
                    noise_config,
                    device=resolved_device,
                    generator=random_generator,
                )
                generated = generator(noise)

                if not isinstance(generated, torch.Tensor):
                    raise RuntimeError(
                        "El generador no devolvió un tensor de PyTorch."
                    )
                if generated.ndim != 2:
                    raise RuntimeError(
                        "La salida del generador debe ser bidimensional."
                    )
                if generated.shape != (current_batch_size, data_dim):
                    raise RuntimeError(
                        "La salida del generador tiene forma incompatible: "
                        f"{tuple(generated.shape)}."
                    )
                if not torch.isfinite(generated).all():
                    raise RuntimeError(
                        "El generador produjo NaN o valores infinitos."
                    )

                generated_batches.append(
                    generated.detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
                remaining -= current_batch_size
    finally:
        generator.train(previous_training_state)

    raw_samples = np.concatenate(generated_batches, axis=0)
    features, labels = validate_synthetic_samples(
        raw_samples,
        expected_count=n_samples,
        data_dim=data_dim,
    )

    return _build_result(
        method="gan",
        features=features,
        labels=labels,
        requested_samples=n_samples,
        seed=seed,
        include_diagnostics=include_diagnostics,
    )


def generate_smote_samples(
    X_train: Any,
    y_train: Any,
    *,
    n_samples: int,
    seed: int,
    k_neighbors: int = 5,
    data_dim: int = 30,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Genera exactamente ``n_samples`` fraudes mediante plain SMOTE.

    La función devuelve únicamente las nuevas observaciones, no el conjunto
    original ni los fraudes originales.
    """
    n_samples = _validate_non_negative_integer(n_samples, "n_samples")
    seed = _validate_seed(seed)
    k_neighbors = _validate_positive_integer(k_neighbors, "k_neighbors")
    data_dim = _validate_positive_integer(data_dim, "data_dim")

    features, labels = _validate_training_data(
        X_train,
        y_train,
        data_dim=data_dim,
    )
    fraud_count = int(np.count_nonzero(labels == 1))

    if k_neighbors >= fraud_count:
        raise ValueError(
            "k_neighbors debe ser menor que la cantidad de fraudes originales "
            f"({fraud_count})."
        )

    if n_samples == 0:
        synthetic_features, synthetic_labels = validate_synthetic_samples(
            np.empty((0, data_dim), dtype=np.float32),
            expected_count=0,
            data_dim=data_dim,
        )
        return _build_result(
            method="smote",
            features=synthetic_features,
            labels=synthetic_labels,
            requested_samples=0,
            seed=seed,
            include_diagnostics=include_diagnostics,
        )

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "La generación SMOTE requiere 'imbalanced-learn'. "
            "Instálelo con: pip install imbalanced-learn"
        ) from exc

    target_fraud_count = fraud_count + n_samples
    smote = SMOTE(
        sampling_strategy={1: target_fraud_count},
        random_state=seed,
        k_neighbors=k_neighbors,
    )

    original_features = features.copy()
    original_labels = labels.copy()
    resampled_features, resampled_labels = smote.fit_resample(
        original_features,
        original_labels,
    )

    resampled_features = np.asarray(resampled_features)
    resampled_labels = np.asarray(resampled_labels, dtype=np.int64)

    expected_total = len(original_features) + n_samples
    if len(resampled_features) != expected_total:
        raise RuntimeError(
            "SMOTE no produjo la cantidad total esperada de observaciones."
        )
    if len(resampled_labels) != expected_total:
        raise RuntimeError(
            "SMOTE produjo cantidades incompatibles de características y etiquetas."
        )

    # imbalanced-learn conserva primero el conjunto original y añade después
    # las observaciones sintéticas. Se valida este contrato antes de extraerlas.
    if not np.array_equal(
        resampled_features[: len(original_features)],
        original_features,
    ):
        raise RuntimeError(
            "La versión de SMOTE utilizada no conservó el conjunto original "
            "como prefijo; no es posible aislar con seguridad las muestras nuevas."
        )
    if not np.array_equal(
        resampled_labels[: len(original_labels)],
        original_labels,
    ):
        raise RuntimeError(
            "SMOTE modificó el orden o las etiquetas del conjunto original."
        )

    new_features = resampled_features[len(original_features) :]
    new_labels = resampled_labels[len(original_labels) :]

    if len(new_features) != n_samples:
        raise RuntimeError(
            "La cantidad de muestras nuevas de SMOTE no coincide con la solicitada."
        )
    if not np.all(new_labels == 1):
        raise RuntimeError(
            "SMOTE produjo muestras sintéticas fuera de la clase fraudulenta."
        )

    synthetic_features, synthetic_labels = validate_synthetic_samples(
        new_features,
        expected_count=n_samples,
        data_dim=data_dim,
    )

    return _build_result(
        method="smote",
        features=synthetic_features,
        labels=synthetic_labels,
        requested_samples=n_samples,
        seed=seed,
        include_diagnostics=include_diagnostics,
    )


def generate_samples(
    method: str,
    *,
    n_samples: int,
    seed: int,
    generator: nn.Module | None = None,
    X_train: Any | None = None,
    y_train: Any | None = None,
    noise_config: Mapping[str, Any] | None = None,
    generation_batch_size: int = 1024,
    noise_dim: int | None = None,
    data_dim: int = 30,
    device: str | torch.device = "auto",
    smote_k_neighbors: int = 5,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Despacha la generación GAN o SMOTE con una interfaz común."""
    normalized_method = str(method).strip().lower()

    if normalized_method == "gan":
        if generator is None:
            raise ValueError("generator es obligatorio para el método GAN.")
        if noise_config is None:
            raise ValueError("noise_config es obligatorio para el método GAN.")

        return generate_gan_samples(
            generator,
            n_samples=n_samples,
            noise_config=noise_config,
            seed=seed,
            batch_size=generation_batch_size,
            device=device,
            noise_dim=noise_dim,
            data_dim=data_dim,
            include_diagnostics=include_diagnostics,
        )

    if normalized_method == "smote":
        if X_train is None or y_train is None:
            raise ValueError(
                "X_train e y_train son obligatorios para el método SMOTE."
            )

        return generate_smote_samples(
            X_train,
            y_train,
            n_samples=n_samples,
            seed=seed,
            k_neighbors=smote_k_neighbors,
            data_dim=data_dim,
            include_diagnostics=include_diagnostics,
        )

    raise ValueError(
        f"Método desconocido '{method}'. Métodos admitidos: "
        f"{', '.join(SUPPORTED_METHODS)}."
    )


def take_sample_prefix(
    generation_result: Mapping[str, Any],
    *,
    n_samples: int,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Extrae un prefijo reproducible de una reserva sintética ya generada.

    Esta función permite generar una sola reserva máxima y reutilizar sus
    primeros ``N_g`` elementos en los experimentos de menor tamaño.
    """
    if not isinstance(generation_result, Mapping):
        raise TypeError("generation_result debe ser un mapeo.")

    required_keys = {
        "method",
        "X_synthetic",
        "y_synthetic",
        "seed",
    }
    missing = required_keys.difference(generation_result)
    if missing:
        raise KeyError(
            "Faltan claves obligatorias en generation_result: "
            + ", ".join(sorted(missing))
        )

    n_samples = _validate_non_negative_integer(n_samples, "n_samples")
    method = str(generation_result["method"]).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError("generation_result contiene un método desconocido.")

    features = np.asarray(generation_result["X_synthetic"])
    labels = np.asarray(generation_result["y_synthetic"])

    if features.ndim != 2:
        raise ValueError("X_synthetic debe ser una matriz bidimensional.")
    if labels.ndim != 1:
        raise ValueError("y_synthetic debe ser un vector unidimensional.")
    if len(features) != len(labels):
        raise ValueError(
            "X_synthetic e y_synthetic deben tener la misma longitud."
        )
    if n_samples > len(features):
        raise ValueError(
            f"El prefijo solicitado ({n_samples}) supera la reserva disponible "
            f"({len(features)})."
        )

    prefix_features, prefix_labels = validate_synthetic_samples(
        features[:n_samples].copy(),
        expected_count=n_samples,
        data_dim=features.shape[1],
    )
    if n_samples and not np.all(labels[:n_samples] == 1):
        raise ValueError(
            "La reserva contiene etiquetas distintas de fraude en el prefijo."
        )

    return _build_result(
        method=method,
        features=prefix_features,
        labels=prefix_labels,
        requested_samples=n_samples,
        seed=_validate_seed(generation_result["seed"]),
        include_diagnostics=include_diagnostics,
    )


__all__ = [
    "SUPPORTED_METHODS",
    "generate_gan_samples",
    "generate_samples",
    "generate_smote_samples",
    "summarize_synthetic_samples",
    "take_sample_prefix",
    "validate_synthetic_samples",
]
