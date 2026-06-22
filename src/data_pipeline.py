"""Pipeline de preparación del dataset de fraude con tarjetas de crédito.

Este módulo implementa exclusivamente la lógica de la Fase 1 del experimento:
carga, validación, eliminación de duplicados, escalamiento Min-Max, partición
reproducible, validación de integridad, persistencia y recarga de artefactos.

El módulo es agnóstico del entorno: no monta Google Drive, no instala paquetes,
no define rutas absolutas y no ejecuta procesamiento al ser importado.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


EXPECTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
)

SPLIT_FILENAME = "paper_split_scaled.npz"
TRAIN_INDICES_FILENAME = "train_indices.npy"
TEST_INDICES_FILENAME = "test_indices.npy"
SCALER_FILENAME = "minmax_scaler.joblib"
METADATA_FILENAME = "data_preparation_config.json"


class DataPipelineError(RuntimeError):
    """Error base para fallos controlados del pipeline de datos."""


class DatasetValidationError(DataPipelineError):
    """El dataset no cumple el esquema o las restricciones metodológicas."""


class PartitionValidationError(DataPipelineError):
    """La partición de entrenamiento/prueba es inconsistente."""


class ArtifactError(DataPipelineError):
    """Un artefacto persistido falta, está incompleto o es incompatible."""


@dataclass(frozen=True)
class DataPipelineConfig:
    """Parámetros configurables requeridos por el pipeline."""

    seed: int
    target_column: str
    remove_duplicates: bool
    scaler_method: str
    feature_range: tuple[float, float]
    scaler_fit_scope: str
    clip_values: bool
    train_legitimate: int
    train_fraud: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DataPipelineConfig":
        """Construye y valida la configuración desde el YAML ya cargado."""
        if not isinstance(config, Mapping):
            raise TypeError("La configuración debe ser un diccionario o Mapping.")

        try:
            project = config["project"]
            preprocessing = config["preprocessing"]
            scaler = preprocessing["scaler"]
            split = preprocessing["split"]
        except KeyError as exc:
            raise DataPipelineError(
                f"Falta la sección o parámetro obligatorio en la configuración: {exc.args[0]}"
            ) from exc
        except TypeError as exc:
            raise DataPipelineError(
                "Las secciones project/preprocessing/scaler/split deben ser mappings."
            ) from exc

        feature_range = scaler.get("feature_range")
        if (
            not isinstance(feature_range, Sequence)
            or isinstance(feature_range, (str, bytes))
            or len(feature_range) != 2
        ):
            raise DataPipelineError(
                "preprocessing.scaler.feature_range debe contener exactamente dos valores."
            )

        lower = float(feature_range[0])
        upper = float(feature_range[1])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise DataPipelineError(
                "El rango de escalamiento debe contener valores finitos con mínimo < máximo."
            )

        parsed = cls(
            seed=int(project["seed"]),
            target_column=str(preprocessing["target_column"]),
            remove_duplicates=bool(preprocessing["remove_duplicates"]),
            scaler_method=str(scaler["method"]).strip().lower(),
            feature_range=(lower, upper),
            scaler_fit_scope=str(scaler["fit_scope"]).strip().lower(),
            clip_values=bool(scaler["clip_values"]),
            train_legitimate=int(split["train_legitimate"]),
            train_fraud=int(split["train_fraud"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.seed < 0:
            raise DataPipelineError("project.seed debe ser mayor o igual que cero.")
        if not self.target_column:
            raise DataPipelineError("preprocessing.target_column no puede estar vacío.")
        if self.scaler_method not in {"minmax", "minmaxscaler"}:
            raise DataPipelineError(
                "Solo se admite MinMaxScaler para reproducir la metodología del paper."
            )
        if self.scaler_fit_scope != "complete_clean_dataset":
            raise DataPipelineError(
                "El protocolo reproducido exige scaler.fit_scope='complete_clean_dataset'."
            )
        if self.train_legitimate <= 0 or self.train_fraud <= 0:
            raise DataPipelineError(
                "Las cantidades de entrenamiento por clase deben ser positivas."
            )


@dataclass
class PreparedData:
    """Resultado completo y persistible de la preparación de datos."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    scaler: MinMaxScaler
    feature_columns: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PersistencePaths:
    """Rutas concretas de los artefactos de la Fase 1."""

    split: Path
    train_indices: Path
    test_indices: Path
    scaler: Path
    metadata: Path


def build_persistence_paths(
    processed_dir: str | os.PathLike[str],
    phase1_artifacts_dir: str | os.PathLike[str],
) -> PersistencePaths:
    """Construye las rutas estándar sin depender de Colab o Google Drive."""
    processed = Path(processed_dir)
    artifacts = Path(phase1_artifacts_dir)
    return PersistencePaths(
        split=processed / SPLIT_FILENAME,
        train_indices=processed / TRAIN_INDICES_FILENAME,
        test_indices=processed / TEST_INDICES_FILENAME,
        scaler=processed / SCALER_FILENAME,
        metadata=artifacts / METADATA_FILENAME,
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _class_counts(values: np.ndarray | pd.Series) -> dict[str, int]:
    array = np.asarray(values)
    return {
        "legitimate": int(np.count_nonzero(array == 0)),
        "fraud": int(np.count_nonzero(array == 1)),
    }


def load_dataset(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Carga el CSV y produce errores explícitos ante entradas inválidas."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    if not path.is_file():
        raise DatasetValidationError(f"La ruta del dataset no es un archivo: {path}")
    if path.stat().st_size == 0:
        raise DatasetValidationError(f"El archivo CSV está vacío: {path}")

    try:
        dataframe = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(
            f"No fue posible interpretar el dataset CSV '{path}': {exc}"
        ) from exc
    except OSError as exc:
        raise DataPipelineError(f"No fue posible leer el dataset '{path}': {exc}") from exc

    if dataframe.empty:
        raise DatasetValidationError("El dataset no contiene registros.")
    return dataframe


def validate_and_order_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Valida el esquema del dataset y devuelve sus columnas en orden canónico."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe debe ser una instancia de pandas.DataFrame.")
    if dataframe.empty:
        raise DatasetValidationError("El dataset no contiene registros.")

    duplicated_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated_columns:
        raise DatasetValidationError(
            f"El dataset contiene nombres de columnas duplicados: {duplicated_columns}"
        )

    expected_columns = [*EXPECTED_FEATURE_COLUMNS, target_column]
    missing = [column for column in expected_columns if column not in dataframe.columns]
    extra = [column for column in dataframe.columns if column not in expected_columns]

    if missing:
        raise DatasetValidationError(
            f"Faltan columnas obligatorias: {missing}. "
            f"Se esperaban {len(expected_columns)} columnas."
        )
    if extra:
        raise DatasetValidationError(
            f"Se encontraron columnas no previstas por la metodología: {extra}."
        )

    ordered = dataframe.loc[:, expected_columns].copy()

    try:
        ordered.loc[:, EXPECTED_FEATURE_COLUMNS] = ordered.loc[
            :, EXPECTED_FEATURE_COLUMNS
        ].apply(pd.to_numeric, errors="raise")
        target_numeric = pd.to_numeric(ordered[target_column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError(
            "Todas las características y la variable objetivo deben ser numéricas."
        ) from exc

    if ordered.loc[:, EXPECTED_FEATURE_COLUMNS].isna().any().any():
        raise DatasetValidationError("Las características contienen valores nulos.")
    if target_numeric.isna().any():
        raise DatasetValidationError("La variable objetivo contiene valores nulos.")

    feature_values = ordered.loc[:, EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise DatasetValidationError("Las características contienen valores infinitos.")

    target_values = target_numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise DatasetValidationError("La variable objetivo contiene valores infinitos.")

    unique_labels = set(np.unique(target_values).tolist())
    if unique_labels != {0.0, 1.0}:
        raise DatasetValidationError(
            "La variable objetivo debe contener exactamente las etiquetas 0 y 1; "
            f"se encontraron {sorted(unique_labels)}."
        )

    ordered[target_column] = target_numeric.astype(np.int64)
    return ordered, list(EXPECTED_FEATURE_COLUMNS)


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
    enabled: bool,
) -> tuple[pd.DataFrame, int]:
    """Elimina duplicados exactos considerando todas las columnas."""
    if not enabled:
        return dataframe.reset_index(drop=True).copy(), 0

    clean = dataframe.drop_duplicates(keep="first").reset_index(drop=True)
    removed = int(len(dataframe) - len(clean))
    return clean, removed


def scale_features(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    feature_range: tuple[float, float],
    clip_values: bool,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Ajusta MinMaxScaler sobre todo el dataset limpio y transforma a float32."""
    X_original = dataframe.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y = dataframe[target_column].to_numpy(dtype=np.int64)

    if not np.isfinite(X_original).all():
        raise DatasetValidationError(
            "Las características contienen NaN o infinitos antes del escalamiento."
        )

    scaler = MinMaxScaler(feature_range=feature_range)
    X = scaler.fit_transform(X_original).astype(np.float32)

    if clip_values:
        X = np.clip(X, feature_range[0], feature_range[1]).astype(np.float32)

    if not np.isfinite(X).all():
        raise DatasetValidationError(
            "El escalamiento produjo valores NaN o infinitos."
        )

    tolerance = np.finfo(np.float32).eps * 8
    observed_min = float(np.min(X))
    observed_max = float(np.max(X))
    if observed_min < feature_range[0] - tolerance:
        raise DatasetValidationError(
            f"El mínimo escalado ({observed_min}) está fuera del rango {feature_range}."
        )
    if observed_max > feature_range[1] + tolerance:
        raise DatasetValidationError(
            f"El máximo escalado ({observed_max}) está fuera del rango {feature_range}."
        )

    return X, y, scaler


def create_exact_classwise_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_legitimate: int,
    train_fraud: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye la partición exacta y determinista utilizada en el notebook."""
    if X.ndim != 2:
        raise DatasetValidationError(f"X debe ser bidimensional; forma recibida: {X.shape}")
    if y.ndim != 1:
        raise DatasetValidationError(f"y debe ser unidimensional; forma recibida: {y.shape}")
    if len(X) != len(y):
        raise DatasetValidationError(
            f"X e y tienen tamaños incompatibles: {len(X)} y {len(y)}."
        )

    fraud_indices = np.flatnonzero(y == 1)
    legitimate_indices = np.flatnonzero(y == 0)

    if len(fraud_indices) < train_fraud:
        raise DatasetValidationError(
            f"Fraudes insuficientes: se requieren {train_fraud} y existen {len(fraud_indices)}."
        )
    if len(legitimate_indices) < train_legitimate:
        raise DatasetValidationError(
            "Transacciones legítimas insuficientes: "
            f"se requieren {train_legitimate} y existen {len(legitimate_indices)}."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(fraud_indices)
    rng.shuffle(legitimate_indices)

    train_fraud_indices = fraud_indices[:train_fraud]
    test_fraud_indices = fraud_indices[train_fraud:]
    train_legitimate_indices = legitimate_indices[:train_legitimate]
    test_legitimate_indices = legitimate_indices[train_legitimate:]

    train_indices = np.concatenate(
        [train_legitimate_indices, train_fraud_indices]
    ).astype(np.int64, copy=False)
    test_indices = np.concatenate(
        [test_legitimate_indices, test_fraud_indices]
    ).astype(np.int64, copy=False)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices].astype(np.float32, copy=False),
        y[train_indices].astype(np.int64, copy=False),
        X[test_indices].astype(np.float32, copy=False),
        y[test_indices].astype(np.int64, copy=False),
        train_indices,
        test_indices,
    )


def validate_prepared_data(
    prepared: PreparedData,
    *,
    expected_train_legitimate: int | None = None,
    expected_train_fraud: int | None = None,
    expected_total_rows: int | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Valida dimensiones, tipos, cobertura, clases y rango de las particiones."""
    arrays = {
        "X_train": prepared.X_train,
        "y_train": prepared.y_train,
        "X_test": prepared.X_test,
        "y_test": prepared.y_test,
        "train_indices": prepared.train_indices,
        "test_indices": prepared.test_indices,
    }
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise PartitionValidationError(f"{name} debe ser un numpy.ndarray.")

    if prepared.X_train.ndim != 2 or prepared.X_test.ndim != 2:
        raise PartitionValidationError("X_train y X_test deben ser matrices bidimensionales.")
    if prepared.y_train.ndim != 1 or prepared.y_test.ndim != 1:
        raise PartitionValidationError("y_train y y_test deben ser vectores unidimensionales.")
    if prepared.train_indices.ndim != 1 or prepared.test_indices.ndim != 1:
        raise PartitionValidationError("Los índices deben ser vectores unidimensionales.")

    feature_count = len(prepared.feature_columns)
    if feature_count != len(EXPECTED_FEATURE_COLUMNS):
        raise PartitionValidationError(
            f"Se esperaban {len(EXPECTED_FEATURE_COLUMNS)} características y se recibieron {feature_count}."
        )
    if prepared.feature_columns != list(EXPECTED_FEATURE_COLUMNS):
        raise PartitionValidationError("El orden de las características no es el canónico.")
    if prepared.X_train.shape[1] != feature_count or prepared.X_test.shape[1] != feature_count:
        raise PartitionValidationError("Las matrices no contienen exactamente 30 características.")

    if len(prepared.X_train) != len(prepared.y_train):
        raise PartitionValidationError("X_train e y_train tienen tamaños incompatibles.")
    if len(prepared.X_test) != len(prepared.y_test):
        raise PartitionValidationError("X_test e y_test tienen tamaños incompatibles.")
    if len(prepared.train_indices) != len(prepared.y_train):
        raise PartitionValidationError("train_indices no coincide con el tamaño de entrenamiento.")
    if len(prepared.test_indices) != len(prepared.y_test):
        raise PartitionValidationError("test_indices no coincide con el tamaño de prueba.")

    if prepared.X_train.dtype != np.float32 or prepared.X_test.dtype != np.float32:
        raise PartitionValidationError("X_train y X_test deben tener dtype float32.")
    if prepared.y_train.dtype != np.int64 or prepared.y_test.dtype != np.int64:
        raise PartitionValidationError("y_train y y_test deben tener dtype int64.")

    for name, matrix in (("X_train", prepared.X_train), ("X_test", prepared.X_test)):
        if not np.isfinite(matrix).all():
            raise PartitionValidationError(f"{name} contiene NaN o infinitos.")
        observed_min = float(np.min(matrix))
        observed_max = float(np.max(matrix))
        tolerance = np.finfo(np.float32).eps * 8
        if observed_min < feature_range[0] - tolerance or observed_max > feature_range[1] + tolerance:
            raise PartitionValidationError(
                f"{name} está fuera del rango {feature_range}: mínimo={observed_min}, máximo={observed_max}."
            )

    for name, labels in (("y_train", prepared.y_train), ("y_test", prepared.y_test)):
        unique = set(np.unique(labels).tolist())
        if not unique.issubset({0, 1}):
            raise PartitionValidationError(f"{name} contiene etiquetas inválidas: {sorted(unique)}")

    train_unique = np.unique(prepared.train_indices)
    test_unique = np.unique(prepared.test_indices)
    if len(train_unique) != len(prepared.train_indices):
        raise PartitionValidationError("train_indices contiene índices duplicados.")
    if len(test_unique) != len(prepared.test_indices):
        raise PartitionValidationError("test_indices contiene índices duplicados.")
    if np.intersect1d(train_unique, test_unique).size != 0:
        raise PartitionValidationError("Existe superposición entre entrenamiento y prueba.")

    total_rows = len(prepared.train_indices) + len(prepared.test_indices)
    if expected_total_rows is not None and total_rows != expected_total_rows:
        raise PartitionValidationError(
            f"Cobertura incompleta: se esperaban {expected_total_rows} filas y se cubrieron {total_rows}."
        )

    combined_indices = np.concatenate([train_unique, test_unique])
    if total_rows > 0:
        if int(combined_indices.min()) < 0:
            raise PartitionValidationError("Se encontraron índices negativos.")
        if expected_total_rows is not None:
            expected_indices = np.arange(expected_total_rows, dtype=np.int64)
            if not np.array_equal(np.sort(combined_indices), expected_indices):
                raise PartitionValidationError(
                    "La unión de entrenamiento y prueba no cubre exactamente el dataset limpio."
                )

    train_counts = _class_counts(prepared.y_train)
    if (
        expected_train_legitimate is not None
        and train_counts["legitimate"] != expected_train_legitimate
    ):
        raise PartitionValidationError(
            "Cantidad legítima de entrenamiento incorrecta: "
            f"esperada={expected_train_legitimate}, obtenida={train_counts['legitimate']}."
        )
    if expected_train_fraud is not None and train_counts["fraud"] != expected_train_fraud:
        raise PartitionValidationError(
            "Cantidad de fraudes de entrenamiento incorrecta: "
            f"esperada={expected_train_fraud}, obtenida={train_counts['fraud']}."
        )

    if not isinstance(prepared.scaler, MinMaxScaler):
        raise PartitionValidationError("El escalador debe ser una instancia ajustada de MinMaxScaler.")
    if not hasattr(prepared.scaler, "n_features_in_"):
        raise PartitionValidationError("El escalador no está ajustado.")
    if int(prepared.scaler.n_features_in_) != feature_count:
        raise PartitionValidationError(
            "El escalador fue ajustado con una cantidad incompatible de características."
        )


def prepare_data(
    csv_path: str | os.PathLike[str],
    config: Mapping[str, Any] | DataPipelineConfig,
) -> PreparedData:
    """Ejecuta el pipeline completo sin guardar archivos ni depender del entorno."""
    pipeline_config = (
        config if isinstance(config, DataPipelineConfig) else DataPipelineConfig.from_mapping(config)
    )
    csv = Path(csv_path)

    original = load_dataset(csv)
    ordered, feature_columns = validate_and_order_dataset(
        original,
        target_column=pipeline_config.target_column,
    )

    original_counts = _class_counts(ordered[pipeline_config.target_column])
    clean, duplicates_removed = remove_duplicate_rows(
        ordered,
        enabled=pipeline_config.remove_duplicates,
    )
    clean_counts = _class_counts(clean[pipeline_config.target_column])

    X, y, scaler = scale_features(
        clean,
        feature_columns=feature_columns,
        target_column=pipeline_config.target_column,
        feature_range=pipeline_config.feature_range,
        clip_values=pipeline_config.clip_values,
    )

    X_train, y_train, X_test, y_test, train_indices, test_indices = (
        create_exact_classwise_split(
            X,
            y,
            seed=pipeline_config.seed,
            train_legitimate=pipeline_config.train_legitimate,
            train_fraud=pipeline_config.train_fraud,
        )
    )

    metadata: dict[str, Any] = {
        "seed": pipeline_config.seed,
        "source_file": csv.name,
        "source_sha256": _sha256_file(csv),
        "target_column": pipeline_config.target_column,
        "feature_columns": feature_columns,
        "number_of_features": len(feature_columns),
        "dataset": {
            "original_rows": int(len(ordered)),
            "clean_rows": int(len(clean)),
            "duplicates_removed": duplicates_removed,
            "original_class_counts": original_counts,
            "clean_class_counts": clean_counts,
        },
        "scaling": {
            "method": "MinMaxScaler",
            "feature_range": list(pipeline_config.feature_range),
            "fit_scope": pipeline_config.scaler_fit_scope,
            "clip_values": pipeline_config.clip_values,
            "observed_min": float(np.min(X)),
            "observed_max": float(np.max(X)),
        },
        "split": {
            "strategy": "classwise_random_exact_counts",
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_class_counts": _class_counts(y_train),
            "test_class_counts": _class_counts(y_test),
        },
        "dtypes": {
            "features": str(X_train.dtype),
            "labels": str(y_train.dtype),
            "indices": str(train_indices.dtype),
        },
    }

    prepared = PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        scaler=scaler,
        feature_columns=feature_columns,
        metadata=metadata,
    )

    validate_prepared_data(
        prepared,
        expected_train_legitimate=pipeline_config.train_legitimate,
        expected_train_fraud=pipeline_config.train_fraud,
        expected_total_rows=len(clean),
        feature_range=pipeline_config.feature_range,
    )
    return prepared


def _ensure_writable_targets(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ArtifactError(
            "Los siguientes artefactos ya existen y overwrite=False: " + ", ".join(existing)
        )


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            writer(temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def persist_prepared_data(
    prepared: PreparedData,
    processed_dir: str | os.PathLike[str],
    phase1_artifacts_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> PersistencePaths:
    """Guarda todos los artefactos de forma atómica en rutas explícitas."""
    validate_prepared_data(prepared)
    paths = build_persistence_paths(processed_dir, phase1_artifacts_dir)
    all_paths = [
        paths.split,
        paths.train_indices,
        paths.test_indices,
        paths.scaler,
        paths.metadata,
    ]
    _ensure_writable_targets(all_paths, overwrite=overwrite)

    _atomic_write(
        paths.split,
        lambda file: np.savez_compressed(
            file,
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
        ),
    )
    _atomic_write(
        paths.train_indices,
        lambda file: np.save(file, prepared.train_indices, allow_pickle=False),
    )
    _atomic_write(
        paths.test_indices,
        lambda file: np.save(file, prepared.test_indices, allow_pickle=False),
    )
    _atomic_write(paths.scaler, lambda file: joblib.dump(prepared.scaler, file))

    metadata_bytes = json.dumps(
        prepared.metadata,
        indent=4,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write(paths.metadata, lambda file: file.write(metadata_bytes))

    return paths


def load_prepared_data(
    processed_dir: str | os.PathLike[str],
    phase1_artifacts_dir: str | os.PathLike[str],
    *,
    validate: bool = True,
) -> PreparedData:
    """Carga los artefactos de Fase 1 y valida su compatibilidad."""
    paths = build_persistence_paths(processed_dir, phase1_artifacts_dir)
    missing = [str(path) for path in paths.__dict__.values() if not path.is_file()]
    if missing:
        raise ArtifactError("Faltan artefactos de la Fase 1: " + ", ".join(missing))

    try:
        with np.load(paths.split, allow_pickle=False) as archive:
            required_keys = {"X_train", "y_train", "X_test", "y_test"}
            missing_keys = required_keys.difference(archive.files)
            if missing_keys:
                raise ArtifactError(
                    f"El archivo {paths.split} no contiene: {sorted(missing_keys)}"
                )
            X_train = archive["X_train"]
            y_train = archive["y_train"]
            X_test = archive["X_test"]
            y_test = archive["y_test"]

        train_indices = np.load(paths.train_indices, allow_pickle=False)
        test_indices = np.load(paths.test_indices, allow_pickle=False)
        scaler = joblib.load(paths.scaler)
        with paths.metadata.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except ArtifactError:
        raise
    except (OSError, ValueError, EOFError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"No fue posible cargar los artefactos de Fase 1: {exc}") from exc

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ArtifactError("Los metadatos no contienen una lista válida de características.")

    prepared = PreparedData(
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        X_test=np.asarray(X_test),
        y_test=np.asarray(y_test),
        train_indices=np.asarray(train_indices),
        test_indices=np.asarray(test_indices),
        scaler=scaler,
        feature_columns=feature_columns,
        metadata=metadata,
    )

    if validate:
        expected_total = metadata.get("dataset", {}).get("clean_rows")
        expected_train_counts = metadata.get("split", {}).get("train_class_counts", {})
        feature_range_raw = metadata.get("scaling", {}).get("feature_range", [0.0, 1.0])
        if not isinstance(feature_range_raw, list) or len(feature_range_raw) != 2:
            raise ArtifactError("Los metadatos contienen un feature_range inválido.")

        validate_prepared_data(
            prepared,
            expected_train_legitimate=expected_train_counts.get("legitimate"),
            expected_train_fraud=expected_train_counts.get("fraud"),
            expected_total_rows=int(expected_total) if expected_total is not None else None,
            feature_range=(float(feature_range_raw[0]), float(feature_range_raw[1])),
        )

    return prepared
