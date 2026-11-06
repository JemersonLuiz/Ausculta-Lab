"""Treina candidatos em diretório separado e avalia a seleção congelada."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cardio_audio.audio import CLASS_NAMES, FEATURE_CONFIG
from cardio_audio.metrics import classification_metrics, patient_bootstrap
from cardio_audio.model import BandCNN
from cardio_audio.predictor import Predictor
from train import make_predictions, prepare_features, save_json


def quality(metrics):
    return metrics["macro_f1"] + 0.2 * min(row["sensitivity"] for row in metrics["per_class"])


def augment(batch, generator):
    # Varia o balanço espectral e oculta uma pequena região de cada eixo.
    batch_size, _, bands, frames = batch.shape
    tilt = (torch.rand((batch_size, 1, 1, 1), generator=generator) - 0.5) * 0.24
    batch = (batch + tilt * torch.linspace(-1, 1, bands).view(1, 1, bands, 1)).clamp(0, 1)
    for axis, length, max_width in ((2, bands, 6), (3, frames, 20)):
        widths = torch.randint(0, max_width + 1, (batch_size, 1), generator=generator)
        starts = (torch.rand((batch_size, 1), generator=generator) * (length - widths + 1)).long()
        position = torch.arange(length).unsqueeze(0)
        mask = (position >= starts) & (position < starts + widths)
        shape = [batch_size, 1, 1, 1]
        shape[axis] = length
        batch = batch.masked_fill(mask.view(shape), 0)
    return batch


def train_candidate(train_features, labels, validation, validation_labels, seed, power, epochs):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = BandCNN()
    counts = torch.bincount(labels, minlength=4)
    weights = (labels.numel() / (4 * counts)).pow(power)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    loader = DataLoader(TensorDataset(train_features, labels), batch_size=64, shuffle=True, generator=generator)
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(augment(batch, generator)), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Perda não finita durante o treinamento.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * len(target)
        scheduler.step()
        scores = classification_metrics(validation_labels, make_predictions(model, validation))
        rank = quality(scores)
        history.append({"epoch": epoch, "train_loss": total_loss / len(labels), "validation_macro_f1": scores["macro_f1"],
                        "validation_recall": [row["sensitivity"] for row in scores["per_class"]], "selection_score": rank})
        print(f"seed={seed} epoch={epoch:02d} macro_f1={scores['macro_f1']:.4f} selection={rank:.4f}", flush=True)
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "epoch": epoch, "metrics": scores,
                    "weights": {name: value.clone() for name, value in model.state_dict().items()}}
        if epoch - best["epoch"] >= 7:
            break
    return best, history, weights.tolist()


def balance_decision(source, output, validation, labels, manifest_hash):
    """Escolhe deslocamentos de logits com foco no menor ganho por classe."""
    predictor = Predictor(source)
    if predictor.metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("O manifesto não corresponde ao candidato treinado.")
    reference = json.loads((ROOT / "models/metrics.json").read_text(encoding="utf-8"))
    original = reference.get("cnn_v1", reference["cnn"])
    baseline = np.array([row["sensitivity"] for row in original["validation"]["per_class"]])
    with torch.inference_mode():
        logits = np.concatenate([
            predictor.model(torch.from_numpy(validation[first:first + 64]).unsqueeze(1)).numpy()
            for first in range(0, len(validation), 64)
        ])
    counts = np.bincount(labels, minlength=4)
    winner = None
    for shifts in product(np.arange(-1.5, 1.51, 0.25), repeat=3):
        offsets = np.array((0, *shifts), dtype=np.float32)
        predicted = (logits + offsets).argmax(axis=1)
        matrix = np.bincount(4 * labels + predicted, minlength=16).reshape(4, 4)
        recall = matrix.diagonal() / counts
        macro_f1 = np.mean(2 * matrix.diagonal() / (counts + matrix.sum(axis=0)))
        rank = (min(recall - baseline), macro_f1, -float(offsets @ offsets))
        if winner is None or rank > winner[0]:
            winner = (rank, offsets)
    _, offsets = winner
    with torch.no_grad():
        predictor.model.classifier[-1].bias.add_(torch.from_numpy(offsets))
    validation_metrics = classification_metrics(labels, make_predictions(predictor.model, validation))
    selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    selection.update({"validation": validation_metrics, "decision_offsets": offsets.tolist(),
                      "criterion": "maximin_validation_recall_gain_vs_v1_then_macro_f1"})
    output.mkdir(parents=True, exist_ok=False)
    torch.save(predictor.model.state_dict(), output / "cnn.pt")
    metadata = {**predictor.metadata, "decision_offsets": offsets.tolist(),
                "weights_sha256": hashlib.sha256((output / "cnn.pt").read_bytes()).hexdigest()}
    save_json(output / "model.json", metadata)
    save_json(output / "selection.json", selection)
    print(json.dumps({"offsets": offsets.tolist(), "validation": validation_metrics}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--evaluate", action="store_true", help="Avalia o candidato congelado no teste.")
    parser.add_argument("--balance-from", type=Path, help="Ajusta a decisão de um candidato usando só a validação.")
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    manifest = ROOT / "data/manifest.csv"
    manifest_bytes = manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    frame = pd.read_csv(manifest, dtype={"patient_id": str})
    if frame.groupby("patient_id").split.nunique().max() != 1:
        raise ValueError("O manifesto mistura pacientes entre conjuntos.")
    fingerprint = hashlib.sha256(manifest_bytes + json.dumps(FEATURE_CONFIG, sort_keys=True).encode()).hexdigest()
    features, _ = prepare_features(frame, ROOT / "data/processed" / f"features_{fingerprint[:16]}.npz", fingerprint)
    output = args.output

    if args.balance_from:
        validation = frame.split == "validation"
        balance_decision(args.balance_from, output, features[validation],
                         frame.loc[validation, "label"].to_numpy(dtype=np.int64), manifest_hash)
        return

    if args.evaluate:
        predictor = Predictor(output)
        if predictor.metadata["manifest_sha256"] != manifest_hash:
            raise ValueError("O manifesto não corresponde ao candidato treinado.")
        mask = frame.split == "test"
        test = frame.loc[mask]
        prediction = make_predictions(predictor.model, features[mask])
        comparison = json.loads((ROOT / "models/metrics.json").read_text(encoding="utf-8"))
        original = comparison.get("cnn_v1", comparison["cnn"])
        evaluation = classification_metrics(test.label, prediction)
        evaluation["macro_f1_ci95"] = patient_bootstrap(test.label, prediction, test.patient_id)
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        comparison["cnn_v1"] = original
        comparison["cnn"] = {"validation": selection["validation"], "test": evaluation}
        comparison["selection"] = {"criterion": selection.get("criterion", "val_macro_f1_plus_0.2_min_recall"),
                                   "best_epoch": selection["epoch"],
                                   "seed": selection["seed"], "test_used_for_selection": False,
                                   "benchmark_previously_used": True, "feature_cache": "float32"}
        columns = ["patient_id", "recording_id", "start_s", "end_s", "label", "split"]
        reference = pd.read_csv(ROOT / "reports/test_predictions.csv", dtype={"patient_id": str})
        pd.testing.assert_frame_equal(test[columns].reset_index(drop=True), reference[columns])
        reference["cnn_v1_prediction"] = reference.get("cnn_v1_prediction", reference["cnn_prediction"])
        reference["cnn_prediction"] = prediction
        reference.to_csv(output / "test_predictions.csv", index=False)
        save_json(output / "metrics.json", comparison)
        print(json.dumps({"validation_macro_f1": selection["validation"]["macro_f1"], "test": evaluation}, ensure_ascii=False))
        return

    output.mkdir(parents=True, exist_ok=False)
    train = frame.split == "train"
    validation = frame.split == "validation"
    train_features = torch.from_numpy(features[train]).unsqueeze(1)
    labels = torch.from_numpy(frame.loc[train, "label"].to_numpy(dtype=np.int64, copy=True))
    val_features = features[validation]
    val_labels = frame.loc[validation, "label"].to_numpy(dtype=np.int64)
    candidates = []
    winner = None
    for seed, power in ((42, 0.45), (137, 0.8)):
        best, history, class_weights = train_candidate(train_features, labels, val_features, val_labels, seed, power, args.epochs)
        record = {"seed": seed, "class_weight_exponent": power, "epoch": best["epoch"], "rank": best["rank"],
                  "validation": best["metrics"], "history": history, "class_weights": class_weights}
        candidates.append(record)
        if winner is None or record["rank"] > winner["rank"]:
            winner = record
            torch.save(best["weights"], output / "cnn.pt")
        save_json(output / "candidates.json", candidates)
    save_json(output / "selection.json", winner)
    weights_hash = hashlib.sha256((output / "cnn.pt").read_bytes()).hexdigest()
    save_json(output / "model.json", {
        "version": "1.1.0", "name": "CNN respiratória por faixas", "task": "pulmonar_4_classes",
        "classes": CLASS_NAMES, "feature_config": FEATURE_CONFIG, "architecture_id": "band_cnn_v2",
        "architecture": "BandCNN: Conv 24/32/48/64, pooling médio/máximo por faixas, FC96/4",
        "parameters": sum(parameter.numel() for parameter in BandCNN().parameters()),
        "trained_at": datetime.now(timezone.utc).isoformat(), "seed": winner["seed"],
        "best_epoch": winner["epoch"], "epochs_executed": len(winner["history"]), "maximum_epochs": args.epochs,
        "batch_size": 64, "optimizer": "AdamW", "learning_rate": 0.001, "class_weights": winner["class_weights"],
        "manifest_sha256": manifest_hash, "weights_sha256": weights_hash,
        "clinical_validation": False, "arrhythmia_available": False,
    })
    print(f"Seleção congelada em {output} (época {winner['epoch']}, semente {winner['seed']}).", flush=True)


if __name__ == "__main__":
    main()
