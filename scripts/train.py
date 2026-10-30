"""Treino reprodutível em ciclos reais. Seleção somente pela validação."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cardio_audio.audio import CLASS_NAMES, FEATURE_CONFIG, AudioData, baseline_features, extract_features, select_segment
from cardio_audio.data import build_manifest
from cardio_audio.metrics import classification_metrics, patient_bootstrap
from cardio_audio.model import SmallCNN


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_features(frame: pd.DataFrame, cache: Path, fingerprint: str):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as loaded:
            if str(loaded["fingerprint"].item()) == fingerprint:
                return loaded["features"].astype(np.float32), loaded["baseline"].astype(np.float32)
    features = np.empty((len(frame), 64, 256), dtype=np.float32)
    baseline = np.empty((len(frame), 40), dtype=np.float32)
    started = time.monotonic()
    for counter, (_, recording) in enumerate(frame.groupby("audio_path", sort=False), 1):
        audio_path = Path(recording.iloc[0].audio_path)
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = AudioData(samples.mean(axis=1), int(rate), samples.shape[1], len(samples) / rate)
        for index, row in recording.iterrows():
            cycle = select_segment(audio, float(row.start_s), float(row.end_s))
            matrix = extract_features(cycle, audio.sample_rate)
            features[index] = matrix
            baseline[index] = baseline_features(matrix)
        if counter % 100 == 0:
            print(f"features: {counter} gravações, {time.monotonic() - started:.1f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, baseline=baseline, fingerprint=np.asarray(fingerprint))
    # Sem quantização: a mesma representação float32 é usada na inferência.
    return features.astype(np.float32), baseline


def make_predictions(model, features, batch_size=64):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[first:first + batch_size]).unsqueeze(1)
            outputs.append(model(batch).argmax(dim=1).numpy())
    return np.concatenate(outputs)


def export_examples(frame, output_dir: Path, maximum=4):
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    test = frame[frame.split == "test"]
    for label in range(4):
        choices = test[(test.label == label) & ((test.end_s - test.start_s) >= 1) & ((test.end_s - test.start_s) <= 10)]
        if choices.empty:
            continue
        # Primeiro exemplo elegível, não escolhido por acerto do modelo.
        row = choices.sort_values(["patient_id", "recording_id", "start_s"]).iloc[0]
        audio_path = Path(row.audio_path)
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        selected = samples[int(round(row.start_s * rate)):int(round(row.end_s * rate))]
        name = f"ciclo_teste_{label}.wav"
        sf.write(output_dir / name, selected, rate, subtype="PCM_16")
        examples.append({"id": str(label), "name": f"Exemplo de teste · {CLASS_NAMES[label]}",
                         "label": CLASS_NAMES[label], "start": 0.0, "end": len(selected) / rate,
                         "file": name, "patient_id": str(row.patient_id), "recording_id": str(row.recording_id),
                         "source_start_s": float(row.start_s), "source_end_s": float(row.end_s),
                         "selection": "primeiro ciclo elegível por classe; não selecionado pelo acerto"})
    save_json(output_dir / "index.json", examples[:maximum])
    (output_dir / "ORIGEM.md").write_text(
        "# Exemplos de pesquisa\n\nRecortes reais da ICBHI 2017, exclusivamente da partição de teste. "
        "São exemplos acadêmicos, não gravações produzidas por este projeto. "
        "A identidade técnica da origem, limites e classe constam em index.json. "
        "Não foram escolhidos para favorecer o modelo. Consulte docs/DADOS.md para fonte, termos e citação.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "icbhi")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Épocas e tamanho do lote devem ser positivos.")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    frame = build_manifest(args.data_root, args.manifest, seed=args.seed, official_overlap_policy="exclude_train").reset_index(drop=True)
    fingerprint = hashlib.sha256(args.manifest.read_bytes() + json.dumps(FEATURE_CONFIG, sort_keys=True).encode()).hexdigest()
    features, traditional = prepare_features(frame, ROOT / "data" / "processed" / f"features_{fingerprint[:16]}.npz", fingerprint)
    labels = frame.label.to_numpy(dtype=np.int64)
    selections = {split: np.flatnonzero(frame.split.to_numpy() == split) for split in ("train", "validation", "test")}
    for split, indices in selections.items():
        if len(indices) == 0 or len(np.unique(labels[indices])) < 4:
            raise RuntimeError(f"A partição {split} está vazia ou não tem as quatro categorias.")
    train_idx, val_idx, test_idx = (selections[key] for key in ("train", "validation", "test"))
    dummy = DummyClassifier(strategy="most_frequent").fit(traditional[train_idx], labels[train_idx])
    baseline = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=args.seed, n_jobs=min(6, os.cpu_count() or 1))
    baseline.fit(traditional[train_idx], labels[train_idx])
    baseline_validation = classification_metrics(labels[val_idx], baseline.predict(traditional[val_idx]))
    joblib.dump(baseline, args.output / "baseline.joblib", compress=3)
    model = SmallCNN()
    counts = np.bincount(labels[train_idx], minlength=4)
    weights = torch.tensor(len(train_idx) / (4 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(features[train_idx]).unsqueeze(1), torch.from_numpy(labels[train_idx])), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    history = []
    best_score, best_epoch, patience = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for values, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Perda não finita no treinamento.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = make_predictions(model, features[val_idx])
        score = float(f1_score(labels[val_idx], predictions, labels=list(range(4)), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        print(f"epoch={epoch} loss={np.mean(losses):.5f} validation_macro_f1={score:.5f}", flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save(model.state_dict(), args.output / "cnn.pt")
        else:
            patience += 1
        save_json(reports / "training_history.json", history)
        if patience >= 6:
            break
    # Teste consultado somente depois de escolher e recarregar os pesos pela validação.
    model.load_state_dict(torch.load(args.output / "cnn.pt", weights_only=True, map_location="cpu"))
    test_prediction = make_predictions(model, features[test_idx])
    baseline_prediction = baseline.predict(traditional[test_idx])
    cnn_test = classification_metrics(labels[test_idx], test_prediction)
    cnn_test["macro_f1_ci95"] = patient_bootstrap(labels[test_idx], test_prediction, frame.iloc[test_idx].patient_id)
    validation = classification_metrics(labels[val_idx], make_predictions(model, features[val_idx]))
    metrics = {"cnn": {"validation": validation, "test": cnn_test},
               "random_forest": {"validation": baseline_validation, "test": classification_metrics(labels[test_idx], baseline_prediction)},
               "majority_baseline": {"test": classification_metrics(labels[test_idx], dummy.predict(traditional[test_idx]))},
               "selection": {"criterion": "validation_macro_f1", "best_epoch": best_epoch, "seed": args.seed,
                             "test_used_for_selection": False, "feature_cache": "float32, same representation as inference"}}
    save_json(args.output / "metrics.json", metrics)
    frame_test = frame.iloc[test_idx].copy()
    frame_test["cnn_prediction"] = test_prediction
    frame_test["random_forest_prediction"] = baseline_prediction
    # Manifesto de avaliação sem caminhos absolutos da máquina.
    export_columns = ["patient_id", "recording_id", "start_s", "end_s", "label", "split", "cnn_prediction", "random_forest_prediction"]
    frame_test[export_columns].to_csv(reports / "test_predictions.csv", index=False)
    dataset = {"name": "ICBHI 2017", "patients": int(frame.patient_id.nunique()),
               "recordings": int(frame.recording_id.nunique()), "cycles": len(frame),
               "splits": {split: {"patients": int(frame.iloc[indices].patient_id.nunique()), "cycles": len(indices),
                                  "class_counts": np.bincount(labels[indices], minlength=4).tolist()} for split, indices in selections.items()},
               "split_protocol": "teste oficial preservado; gravações de treino de pacientes presentes no teste excluídas; validação interna por paciente",
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    save_json(args.output / "dataset.json", dataset)
    weights_hash = hashlib.sha256((args.output / "cnn.pt").read_bytes()).hexdigest()
    metadata = {"version": "1.0.0", "name": "CNN respiratória compacta", "task": "pulmonar_4_classes", "classes": CLASS_NAMES,
                "feature_config": FEATURE_CONFIG, "architecture": "Conv(16)-Conv(32)-Conv(64)-AdaptivePool(4,8)-FC64-FC4",
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trained_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "best_epoch": best_epoch, "epochs_executed": len(history), "maximum_epochs": args.epochs,
                "batch_size": args.batch_size, "optimizer": "AdamW", "learning_rate": 0.0007, "class_weights": weights.tolist(),
                "weights_sha256": weights_hash, "manifest_sha256": dataset["manifest_sha256"],
                "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
                "training_seconds": time.monotonic() - started,
                "clinical_validation": False, "arrhythmia_available": False,
                "cache_quantization": "none; float32 in training and inference"}
    save_json(args.output / "model.json", metadata)
    export_examples(frame, ROOT / "examples")
    print(json.dumps({"status": "complete", "best_epoch": best_epoch, "cnn_test_macro_f1": cnn_test["macro_f1"],
                      "rf_test_macro_f1": metrics["random_forest"]["test"]["macro_f1"], "seconds": metadata["training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
