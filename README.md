# Reproducción experimental de GAN para detección de fraude

Este repositorio contiene una reproducción académica de un enfoque basado en redes generativas adversarias (GAN) para apoyar la detección de fraude con tarjetas de crédito.

El experimento compara una línea base de clasificación con estrategias de aumento de datos para la clase minoritaria, incluyendo muestras sintéticas generadas por una GAN y un escenario de comparación con SMOTE.

## Objetivo

El trabajo busca analizar si el aumento de datos sintéticos puede mejorar la detección de transacciones fraudulentas en un problema fuertemente desbalanceado.

La implementación separa el flujo en etapas de preparación de datos, entrenamiento del clasificador, entrenamiento de la GAN, generación de muestras y evaluación.

## Flujo experimental

```text
Dataset de transacciones
        |
        v
Validación y preprocesamiento
        |
        v
Partición reproducible
        |
        +--------------------+
        |                    |
        v                    v
Clasificador base      Entrenamiento GAN
                             |
                             v
                   Muestras sintéticas de fraude
                             |
                             v
                  Experimentos de aumento
                             |
                             v
                      Evaluación final
```

## Componentes principales

- `src/data_pipeline.py`: carga, validación, limpieza, escalamiento y partición.
- `src/classifier_pipeline.py`: entrenamiento del clasificador base.
- `src/gan_pipeline.py`: arquitectura y entrenamiento de la GAN.
- `src/sample_generation.py`: generación de muestras sintéticas.
- `src/augmentation_experiment.py`: experimentos de aumento de datos.
- `src/evaluation.py`: cálculo de métricas y comparación de resultados.
- `configs/experiment_config.yaml`: parámetros del experimento.
- `notebooks/Laboratorio_4_IA_2.ipynb`: ejecución y análisis experimental.

## Reproducibilidad

El proyecto utiliza una semilla fija y una configuración centralizada en YAML.

La preparación de datos incluye validaciones explícitas sobre estructura, valores y particiones. Los artefactos generados se almacenan por fases para poder reutilizar resultados sin repetir todo el pipeline.

## Configuración

Los principales parámetros del experimento se encuentran en:

```text
configs/experiment_config.yaml
```

Allí se definen, entre otros:

- semilla;
- rutas de entrada y salida;
- tamaño de los conjuntos de entrenamiento;
- arquitectura del clasificador;
- arquitectura de la GAN;
- tasas de aprendizaje;
- número de épocas;
- cantidades de muestras sintéticas;
- configuración de SMOTE.

## Dependencias

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

El proyecto utiliza Python, NumPy, pandas, scikit-learn, imbalanced-learn, PyTorch, PyYAML y joblib.

## Datos

El dataset no se incluye en el repositorio.

La configuración espera por defecto un archivo:

```text
data/raw/creditcard.csv
```

La clase objetivo utilizada por el pipeline es `Class`, con 0 para transacciones legítimas y 1 para fraude.

## Evaluación

El clasificador se evalúa con métricas apropiadas para clasificación binaria y escenarios desbalanceados, entre ellas:

- precision;
- recall;
- F1-Score;
- ROC-AUC;
- PR-AUC;
- matriz de confusión.

El proyecto también compara distintos tamaños de aumento de datos para analizar cómo cambia el comportamiento del clasificador al incorporar muestras sintéticas.

## Limitaciones

Este repositorio corresponde a una reproducción experimental y no a un sistema antifraude listo para producción.

El uso de datos sintéticos puede alterar la distribución original y no garantiza mejoras en todos los escenarios. Los resultados dependen de la partición, la configuración del entrenamiento y la representatividad del dataset utilizado.

## Uso académico

El proyecto fue desarrollado con fines de aprendizaje y validación metodológica dentro del curso de Inteligencia Artificial II.
