"""Entrenamiento GAN para generar transacciones fraudulentas sintéticas.

Implementa la arquitectura descrita por Fiore et al. y exige por configuración
los hiperparámetros que el paper no publica de forma inequívoca. El módulo no
contiene rutas, persistencia ni dependencias de Google Colab.
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    """Tres capas lineales con activaciones ReLU, ReLU y Sigmoid."""

    def __init__(
        self,
        noise_dim: int,
        hidden_dims: tuple[int, int],
        data_dim: int = 30,
    ) -> None:
        super().__init__()
        if noise_dim <= 0 or data_dim <= 0:
            raise ValueError("noise_dim y data_dim deben ser positivos.")
        if len(hidden_dims) != 2 or min(hidden_dims) <= 0:
            raise ValueError("El generador requiere dos capas ocultas positivas.")

        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], data_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """Dos capas ocultas Sigmoid y una salida lineal de un logit."""

    def __init__(
        self,
        data_dim: int = 30,
        hidden_dims: tuple[int, int] = (36, 36),
    ) -> None:
        super().__init__()
        if data_dim <= 0:
            raise ValueError("data_dim debe ser positivo.")
        if len(hidden_dims) != 2 or min(hidden_dims) <= 0:
            raise ValueError("El discriminador requiere dos capas ocultas positivas.")

        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dims[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def set_seed(seed: int) -> None:
    """Restablece Python, NumPy, PyTorch y CUDA."""
    if seed < 0:
        raise ValueError("La semilla debe ser no negativa.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    else:
        name = str(device).lower().strip()
        if name == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif name in {"cpu", "cuda"}:
            resolved = torch.device(name)
        else:
            raise ValueError("device debe ser 'auto', 'cpu', 'cuda' o torch.device.")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible.")
    return resolved


def _required(config: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in config:
        raise KeyError(f"Falta el parámetro obligatorio '{section}.{key}'.")
    return config[key]


def extract_fraud_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    data_dim: int = 30,
    expected_count: int | None = 315,
) -> np.ndarray:
    """Valida entrenamiento y devuelve únicamente las filas fraudulentas."""
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        raise TypeError("X_train e y_train deben ser arreglos NumPy.")
    if X_train.ndim != 2 or y_train.ndim != 1:
        raise ValueError("X_train debe ser 2D e y_train debe ser 1D.")
    if len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener igual longitud no vacía.")
    if X_train.shape[1] != data_dim:
        raise ValueError(
            f"Se esperaban {data_dim} características y se recibieron "
            f"{X_train.shape[1]}."
        )
    if not np.issubdtype(X_train.dtype, np.number) or not np.isfinite(X_train).all():
        raise ValueError("X_train debe contener valores numéricos finitos.")
    if X_train.min() < -1e-7 or X_train.max() > 1.0 + 1e-7:
        raise ValueError("X_train debe estar escalado al intervalo [0, 1].")
    if not np.issubdtype(y_train.dtype, np.number) or not np.isfinite(y_train).all():
        raise ValueError("y_train debe contener etiquetas binarias finitas.")
    if not np.equal(y_train, np.floor(y_train)).all():
        raise ValueError("y_train debe contener etiquetas enteras.")

    labels = y_train.astype(np.int64, copy=False)
    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError("y_train debe contener únicamente las etiquetas 0 y 1.")

    features = np.ascontiguousarray(np.clip(X_train, 0.0, 1.0), dtype=np.float32)
    frauds = np.ascontiguousarray(features[labels == 1], dtype=np.float32)
    if len(frauds) == 0:
        raise ValueError("No se encontraron fraudes en entrenamiento.")
    if expected_count is not None and len(frauds) != expected_count:
        raise ValueError(
            f"Se esperaban {expected_count} fraudes y se encontraron {len(frauds)}."
        )
    return frauds


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_fraud_dataloader(
    fraud_samples: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: str | torch.device = "auto",
) -> DataLoader:
    """Crea un DataLoader reproducible, mezclado y sin descartar el último lote."""
    if fraud_samples.ndim != 2 or len(fraud_samples) == 0:
        raise ValueError("fraud_samples debe ser una matriz 2D no vacía.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size debe ser positivo y num_workers no negativo.")

    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(fraud_samples, dtype=np.float32))
    )
    use_pin_memory = bool(pin_memory and resolve_device(device).type == "cuda")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def initialize_network(model: nn.Module, config: Mapping[str, Any]) -> None:
    """Inicializa pesos y sesgos con la distribución configurada."""
    method = str(_required(config, "method", "gan_training.initialization")).lower()

    for layer in model.modules():
        if not isinstance(layer, nn.Linear):
            continue

        if method == "uniform":
            minimum = float(_required(config, "min_value", "initialization"))
            maximum = float(_required(config, "max_value", "initialization"))
            if minimum >= maximum:
                raise ValueError("min_value debe ser menor que max_value.")
            nn.init.uniform_(layer.weight, minimum, maximum)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, minimum, maximum)

        elif method == "normal":
            mean = float(_required(config, "mean", "initialization"))
            std = float(_required(config, "std", "initialization"))
            if std <= 0:
                raise ValueError("std debe ser positivo.")
            nn.init.normal_(layer.weight, mean, std)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean, std)
        else:
            raise ValueError("initialization.method debe ser 'uniform' o 'normal'.")


def build_gan_models(
    model_config: Mapping[str, Any],
    initialization_config: Mapping[str, Any],
    *,
    seed: int,
    device: str | torch.device = "auto",
) -> tuple[Generator, Discriminator]:
    """Construye las arquitecturas del paper e inicializa ambos modelos."""
    data_dim = int(model_config.get("data_dim", 30))
    noise_dim = int(model_config.get("noise_dim", 100))
    generator_config = _required(model_config, "generator", "gan_model")
    discriminator_config = _required(model_config, "discriminator", "gan_model")

    generator_hidden = tuple(
        int(value)
        for value in _required(generator_config, "hidden_layers", "gan_model.generator")
    )
    discriminator_hidden = tuple(
        int(value) for value in discriminator_config.get("hidden_layers", [36, 36])
    )

    if len(generator_hidden) != 2:
        raise ValueError("gan_model.generator.hidden_layers debe tener dos valores.")
    if discriminator_hidden != (36, 36):
        raise ValueError("El paper requiere capas ocultas [36, 36] en D.")
    if [str(x).lower() for x in generator_config.get(
        "hidden_activations", ["relu", "relu"]
    )] != ["relu", "relu"]:
        raise ValueError("El generador requiere activaciones ReLU y ReLU.")
    if str(generator_config.get("output_activation", "sigmoid")).lower() != "sigmoid":
        raise ValueError("La salida del generador debe usar Sigmoid.")
    if str(discriminator_config.get("hidden_activation", "sigmoid")).lower() != "sigmoid":
        raise ValueError("Las capas ocultas del discriminador deben usar Sigmoid.")

    set_seed(seed)
    generator = Generator(noise_dim, generator_hidden, data_dim)
    discriminator = Discriminator(data_dim, discriminator_hidden)
    initialize_network(generator, initialization_config)
    initialize_network(discriminator, initialization_config)

    resolved_device = resolve_device(device)
    return generator.to(resolved_device), discriminator.to(resolved_device)


def calculate_momentum(
    epoch_index: int,
    initial: float,
    final: float,
    warmup_epochs: int,
) -> float:
    """Aumenta linealmente el momentum y lo satura en el valor final."""
    if epoch_index < 0 or warmup_epochs <= 0:
        raise ValueError("epoch_index y warmup_epochs deben ser válidos.")
    if not 0.0 <= initial <= final < 1.0:
        raise ValueError("Se requiere 0 <= initial <= final < 1.")
    if epoch_index >= warmup_epochs:
        return final

    progress = epoch_index / max(warmup_epochs - 1, 1)
    return initial + progress * (final - initial)


def sample_noise(
    batch_size: int,
    noise_dim: int,
    noise_config: Mapping[str, Any],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Muestrea ruido normal o uniforme según la decisión de reproducción."""
    distribution = str(_required(noise_config, "distribution", "gan_training.noise")).lower()

    if distribution == "normal":
        mean = float(noise_config.get("mean", 0.0))
        std = float(noise_config.get("std", 1.0))
        if std <= 0:
            raise ValueError("noise.std debe ser positivo.")
        return torch.randn(
            batch_size,
            noise_dim,
            device=device,
            generator=generator,
        ).mul(std).add(mean)

    if distribution == "uniform":
        minimum = float(_required(noise_config, "min_value", "gan_training.noise"))
        maximum = float(_required(noise_config, "max_value", "gan_training.noise"))
        if minimum >= maximum:
            raise ValueError("noise.min_value debe ser menor que max_value.")
        return torch.rand(
            batch_size,
            noise_dim,
            device=device,
            generator=generator,
        ).mul(maximum - minimum).add(minimum)

    raise ValueError("noise.distribution debe ser 'normal' o 'uniform'.")


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    fraud_loader: DataLoader,
    training_config: Mapping[str, Any],
    *,
    seed: int,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Entrena D y G con actualizaciones alternadas y pérdidas binarias."""
    resolved_device = resolve_device(device)
    generator.to(resolved_device)
    discriminator.to(resolved_device)

    epochs = int(_required(training_config, "epochs", "gan_training"))
    generator_lr = float(
        _required(training_config, "generator_learning_rate", "gan_training")
    )
    discriminator_lr = float(
        _required(training_config, "discriminator_learning_rate", "gan_training")
    )
    discriminator_steps = int(training_config.get("discriminator_steps", 1))
    generator_steps = int(training_config.get("generator_steps", 1))
    log_frequency = int(training_config.get("log_frequency", 1))
    objective = str(
        _required(training_config, "generator_objective", "gan_training")
    ).lower()

    if epochs <= 0 or generator_lr <= 0 or discriminator_lr <= 0:
        raise ValueError("epochs y learning rates deben ser positivos.")
    if discriminator_steps <= 0 or generator_steps <= 0 or log_frequency <= 0:
        raise ValueError("Los pasos y log_frequency deben ser positivos.")
    if objective not in {"non_saturating", "minimax"}:
        raise ValueError("generator_objective debe ser 'non_saturating' o 'minimax'.")
    if len(fraud_loader) == 0:
        raise ValueError("El DataLoader de fraudes está vacío.")

    optimizer_config = _required(training_config, "optimizer", "gan_training")
    if str(optimizer_config.get("name", "sgd")).lower() != "sgd":
        raise ValueError("La reproducción metodológica utiliza SGD.")

    momentum_config = _required(training_config, "momentum", "gan_training")
    initial_momentum = float(_required(momentum_config, "initial", "momentum"))
    final_momentum = float(_required(momentum_config, "final", "momentum"))
    warmup_epochs = int(_required(momentum_config, "warmup_epochs", "momentum"))
    nesterov = bool(optimizer_config.get("nesterov", True))
    calculate_momentum(0, initial_momentum, final_momentum, warmup_epochs)
    if nesterov and initial_momentum <= 0.0:
        raise ValueError("Nesterov requiere momentum inicial mayor que cero.")

    generator_optimizer = torch.optim.SGD(
        generator.parameters(),
        lr=generator_lr,
        momentum=initial_momentum,
        nesterov=nesterov,
    )
    discriminator_optimizer = torch.optim.SGD(
        discriminator.parameters(),
        lr=discriminator_lr,
        momentum=initial_momentum,
        nesterov=nesterov,
    )
    criterion = nn.BCEWithLogitsLoss()
    noise_config = _required(training_config, "noise", "gan_training")
    generator_device = "cuda" if resolved_device.type == "cuda" else "cpu"
    noise_generator = torch.Generator(device=generator_device).manual_seed(seed + 1)
    history: list[dict[str, int | float]] = []

    for epoch_index in range(epochs):
        started_at = time.perf_counter()
        generator.train()
        discriminator.train()

        momentum = calculate_momentum(
            epoch_index,
            initial_momentum,
            final_momentum,
            warmup_epochs,
        )
        for optimizer in (generator_optimizer, discriminator_optimizer):
            for group in optimizer.param_groups:
                group["momentum"] = momentum

        totals = np.zeros(6, dtype=np.float64)
        processed = 0

        for (real_samples,) in fraud_loader:
            real_samples = real_samples.to(resolved_device, non_blocking=True)
            batch_size = real_samples.size(0)

            d_values = np.zeros(5, dtype=np.float64)
            for _ in range(discriminator_steps):
                discriminator_optimizer.zero_grad(set_to_none=True)
                noise = sample_noise(
                    batch_size,
                    generator.noise_dim,
                    noise_config,
                    device=resolved_device,
                    generator=noise_generator,
                )
                with torch.no_grad():
                    fake_samples = generator(noise)

                real_logits = discriminator(real_samples)
                fake_logits = discriminator(fake_samples)
                real_loss = criterion(real_logits, torch.ones_like(real_logits))
                fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
                discriminator_loss = real_loss + fake_loss

                if not torch.isfinite(discriminator_loss):
                    raise RuntimeError("La pérdida del discriminador no es finita.")

                discriminator_loss.backward()
                discriminator_optimizer.step()
                d_values += np.array([
                    discriminator_loss.item(),
                    real_loss.item(),
                    fake_loss.item(),
                    torch.sigmoid(real_logits).mean().item(),
                    torch.sigmoid(fake_logits).mean().item(),
                ])
            d_values /= discriminator_steps

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            generator_loss_total = 0.0
            try:
                for _ in range(generator_steps):
                    generator_optimizer.zero_grad(set_to_none=True)
                    noise = sample_noise(
                        batch_size,
                        generator.noise_dim,
                        noise_config,
                        device=resolved_device,
                        generator=noise_generator,
                    )
                    fake_logits = discriminator(generator(noise))
                    if objective == "non_saturating":
                        generator_loss = criterion(
                            fake_logits,
                            torch.ones_like(fake_logits),
                        )
                    else:
                        generator_loss = -criterion(
                            fake_logits,
                            torch.zeros_like(fake_logits),
                        )

                    if not torch.isfinite(generator_loss):
                        raise RuntimeError("La pérdida del generador no es finita.")

                    generator_loss.backward()
                    generator_optimizer.step()
                    generator_loss_total += generator_loss.item()
            finally:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

            generator_loss_mean = generator_loss_total / generator_steps
            totals += np.array([*d_values, generator_loss_mean]) * batch_size
            processed += batch_size

        if processed == 0:
            raise RuntimeError("El DataLoader no produjo muestras.")

        averages = totals / processed
        row: dict[str, int | float] = {
            "epoch": epoch_index + 1,
            "discriminator_loss": float(averages[0]),
            "discriminator_real_loss": float(averages[1]),
            "discriminator_fake_loss": float(averages[2]),
            "discriminator_real_probability": float(averages[3]),
            "discriminator_fake_probability": float(averages[4]),
            "generator_loss": float(averages[5]),
            "generator_learning_rate": generator_lr,
            "discriminator_learning_rate": discriminator_lr,
            "momentum": momentum,
            "duration_seconds": time.perf_counter() - started_at,
        }
        history.append(row)

        if verbose and (
            (epoch_index + 1) % log_frequency == 0 or epoch_index + 1 == epochs
        ):
            print(
                f"Epoch {epoch_index + 1:04d}/{epochs:04d} | "
                f"D={row['discriminator_loss']:.6f} | "
                f"G={row['generator_loss']:.6f} | "
                f"D(real)={row['discriminator_real_probability']:.4f} | "
                f"D(fake)={row['discriminator_fake_probability']:.4f} | "
                f"{row['duration_seconds']:.2f}s"
            )

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history,
        "termination_reason": "maximum_epochs_reached",
    }


def generate_diagnostic_samples(
    generator: Generator,
    *,
    sample_count: int,
    noise_config: Mapping[str, Any],
    seed: int,
    device: str | torch.device = "auto",
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Genera una vista previa y valida forma, rango y diversidad básica."""
    if sample_count <= 0:
        raise ValueError("sample_count debe ser positivo.")

    resolved_device = resolve_device(device)
    generator_device = "cuda" if resolved_device.type == "cuda" else "cpu"
    random_generator = torch.Generator(device=generator_device).manual_seed(seed)
    previous_state = generator.training
    generator.eval()

    try:
        with torch.inference_mode():
            noise = sample_noise(
                sample_count,
                generator.noise_dim,
                noise_config,
                device=resolved_device,
                generator=random_generator,
            )
            samples = generator(noise).cpu().numpy().astype(np.float32)
    finally:
        generator.train(previous_state)

    if samples.shape != (sample_count, generator.data_dim):
        raise RuntimeError(f"Forma de muestras inválida: {samples.shape}.")
    if not np.isfinite(samples).all():
        raise RuntimeError("Las muestras generadas contienen valores no finitos.")
    if samples.min() < -1e-7 or samples.max() > 1.0 + 1e-7:
        raise RuntimeError("Las muestras generadas salieron de [0, 1].")

    unique_rows = int(np.unique(samples, axis=0).shape[0])
    mean_feature_std = float(np.std(samples, axis=0).mean())
    if unique_rows <= 1 or mean_feature_std == 0.0:
        warnings.warn(
            "La vista previa no presenta variación; posible colapso del generador.",
            RuntimeWarning,
            stacklevel=2,
        )

    diagnostics: dict[str, int | float] = {
        "sample_count": sample_count,
        "minimum": float(samples.min()),
        "maximum": float(samples.max()),
        "mean": float(samples.mean()),
        "standard_deviation": float(samples.std()),
        "mean_feature_standard_deviation": mean_feature_std,
        "unique_rows": unique_rows,
        "unique_ratio": float(unique_rows / sample_count),
    }
    return samples, diagnostics


def run_gan_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo usando las secciones globales del YAML."""
    project_config = _required(config, "project", "config")
    model_config = _required(config, "gan_model", "config")
    training_config = _required(config, "gan_training", "config")

    seed = int(_required(project_config, "seed", "project"))
    data_dim = int(model_config.get("data_dim", 30))
    expected = training_config.get("expected_fraud_samples", 315)
    expected_count = None if expected is None else int(expected)
    initialization = _required(training_config, "initialization", "gan_training")

    set_seed(seed)
    fraud_samples = extract_fraud_samples(
        X_train,
        y_train,
        data_dim=data_dim,
        expected_count=expected_count,
    )
    fraud_loader = create_fraud_dataloader(
        fraud_samples,
        batch_size=int(_required(training_config, "batch_size", "gan_training")),
        seed=seed,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", True)),
        device=device,
    )
    generator, discriminator = build_gan_models(
        model_config,
        initialization,
        seed=seed,
        device=device,
    )
    result = train_gan(
        generator,
        discriminator,
        fraud_loader,
        training_config,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    preview, diagnostics = generate_diagnostic_samples(
        generator,
        sample_count=int(training_config.get("diagnostic_samples", 64)),
        noise_config=_required(training_config, "noise", "gan_training"),
        seed=seed + 2,
        device=device,
    )

    return {
        **result,
        "metadata": {
            "seed": seed,
            "fraud_samples_used": int(len(fraud_samples)),
            "data_dim": data_dim,
            "noise_dim": int(model_config.get("noise_dim", 100)),
            "generator_hidden_layers": list(model_config["generator"]["hidden_layers"]),
            "discriminator_hidden_layers": list(
                model_config["discriminator"].get("hidden_layers", [36, 36])
            ),
            "generator_objective": str(training_config["generator_objective"]),
            "epochs_executed": len(result["history"]),
        },
        "preview_samples": preview,
        "preview_diagnostics": diagnostics,
    }


__all__ = [
    "Discriminator",
    "Generator",
    "build_gan_models",
    "calculate_momentum",
    "create_fraud_dataloader",
    "extract_fraud_samples",
    "generate_diagnostic_samples",
    "initialize_network",
    "resolve_device",
    "run_gan_pipeline",
    "sample_noise",
    "set_seed",
    "train_gan",
]
