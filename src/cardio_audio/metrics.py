"""Métricas com definições explícitas e suporte de todas as classes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score

from .audio import CLASS_NAMES


def classification_metrics(truth, predictions) -> dict:
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if truth.ndim != 1 or predictions.ndim != 1 or len(truth) == 0 or truth.shape != predictions.shape:
        raise ValueError("Verdade e previsão devem ter o mesmo número positivo de elementos.")
    for values in (truth, predictions):
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1, 2, 3]).all():
            raise ValueError("Categorias devem ser inteiros de 0 a 3.")
    truth, predictions = truth.astype(int), predictions.astype(int)
    matrix = confusion_matrix(truth, predictions, labels=list(range(4)))
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        per_class.append({"label": name, "support": tp + fn,
                          "sensitivity": tp / (tp + fn) if tp + fn else None,
                          "specificity": tn / (tn + fp) if tn + fp else None,
                          "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
    abnormal_count = int(matrix[1:].sum())
    normal_count = int(matrix[0].sum())
    # Benchmark: classificação correta da classe anormal exata (não apenas anormal vs normal).
    sensitivity = float(np.trace(matrix[1:, 1:]) / abnormal_count) if abnormal_count else None
    specificity = float(matrix[0, 0] / normal_count) if normal_count else None
    return {"n_cycles": len(truth), "accuracy": float(accuracy_score(truth, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, labels=list(range(4)), average="macro", zero_division=0)),
            "confusion_matrix": matrix.tolist(), "classes": CLASS_NAMES, "per_class": per_class,
            "icbhi_sensitivity": sensitivity, "icbhi_specificity": specificity,
            "icbhi_score": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "evaluation_unit": "ciclo respiratório", "specificity_definition": "um-contra-os-demais por classe",
            "icbhi_definition": "média entre acerto dos ciclos sem eventos e acerto da categoria exata nos ciclos com eventos"}


def patient_bootstrap(truth, predictions, patient_ids, repetitions: int = 500, seed: int = 42) -> dict:
    truth, predictions, groups = np.asarray(truth), np.asarray(predictions), np.asarray(patient_ids).astype(str)
    patients = np.unique(groups)
    matrices = np.array([confusion_matrix(truth[groups == group], predictions[groups == group], labels=list(range(4))) for group in patients])
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        summed = matrices[generator.integers(0, len(patients), size=len(patients))].sum(axis=0)
        tp = np.diag(summed).astype(float)
        denominators = summed.sum(axis=0) + summed.sum(axis=1)
        f1 = np.divide(2 * tp, denominators, out=np.zeros(4), where=denominators > 0)
        values.append(float(f1.mean()))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"metric": "macro_f1", "lower": float(lower), "upper": float(upper), "confidence": 0.95,
            "method": "bootstrap por paciente", "patients": len(patients), "repetitions": repetitions, "seed": seed}

