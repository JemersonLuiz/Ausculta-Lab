"""Strict ICBHI ingestion and auditable patient-disjoint partitions.

No patient diagnosis is used as an acoustic label. Label = crackles + 2*wheezes.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


class DataValidationError(ValueError):
    """Input data would invalidate the supervised experiment."""


def discover_records(root: str | Path) -> list[dict]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DataValidationError(f"Diretório de dados inexistente: {root}")
    records, seen = [], set()
    for audio in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"):
        parts = audio.stem.split("_")
        if len(parts) < 5 or not parts[0].isdigit():
            raise DataValidationError(f"Nome de gravação ICBHI inválido: {audio.name}")
        if audio.stem in seen:
            raise DataValidationError(f"Identificador de gravação repetido: {audio.stem}")
        annotation = audio.with_suffix(".txt")
        if not annotation.is_file():
            raise DataValidationError(f"Anotação ausente: {annotation.name}")
        seen.add(audio.stem)
        records.append({"audio_path": str(audio), "annotation_path": str(annotation),
                        "patient_id": parts[0], "recording_id": audio.stem})
    if not records:
        raise DataValidationError("Nenhum par WAV/TXT encontrado.")
    return records


def _parse_row(line: str, duration_s: float | None = None) -> dict:
    fields = line.split()
    if len(fields) != 4:
        raise DataValidationError("Anotação deve ter exatamente quatro colunas.")
    try:
        start, end, crackles, wheezes = map(float, fields)
    except ValueError as exc:
        raise DataValidationError("Anotação contém valor não numérico.") from exc
    if not all(math.isfinite(x) for x in (start, end, crackles, wheezes)):
        raise DataValidationError("Anotação contém valor não finito.")
    if start < 0 or end <= start:
        raise DataValidationError("Limites temporais inválidos.")
    if crackles not in (0, 1) or wheezes not in (0, 1):
        raise DataValidationError("Indicadores de estertores/sibilos devem ser 0 ou 1.")
    if duration_s is not None and end > duration_s + 1e-6:
        raise DataValidationError("Fim do ciclo ultrapassa a duração da gravação.")
    return {"start_s": start, "end_s": end, "label": int(crackles + 2 * wheezes)}


def parse_annotations(path: str | Path, duration_s: float | None = None) -> list[dict]:
    """Read rows strictly; malformed/out-of-range rows raise DataValidationError."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(_parse_row(line, duration_s))
        except DataValidationError as exc:
            raise DataValidationError(f"{Path(path).name}:{number}: {exc}") from exc
    if not rows:
        raise DataValidationError(f"Anotação vazia: {path}")
    return rows


def _canonical_record_id(value: str) -> str:
    # The official release documents corrected equipment names. Other fields stay unchanged.
    return "_".join(Path(value).stem.split("_")[:4])


def _official_patient_split(path: Path, records: list[dict], overlap_policy: str) -> tuple[dict, list]:
    by_record, by_patient, record_assignments = {}, {}, {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2 or fields[1].lower() not in {"train", "test"}:
            raise DataValidationError(f"Split oficial inválido na linha {number}.")
        key, split = _canonical_record_id(fields[0]), fields[1].lower()
        if key in by_record and by_record[key] != split:
            raise DataValidationError(f"Gravação com partições conflitantes: {key}")
        by_record[key] = split
    for record in records:
        key = _canonical_record_id(record["recording_id"])
        # Also permit explicitly patient-level two-column files.
        split = by_record.get(key, by_record.get(record["patient_id"]))
        if split is None:
            raise DataValidationError(f"Gravação sem partição oficial: {record['recording_id']}")
        patient = record["patient_id"]
        by_patient.setdefault(patient, set()).add(split)
        record_assignments[record["recording_id"]] = split
    overlapping = sorted(p for p, splits in by_patient.items() if len(splits) > 1)
    if overlapping and overlap_policy == "error":
        raise DataValidationError("Paciente presente em treino e teste oficiais: " + ", ".join(overlapping))
    excluded = [{"recording_id": record["recording_id"], "patient_id": record["patient_id"],
                 "original_split": "train", "reason": "official_patient_overlap_train_excluded"}
                for record in records if record["patient_id"] in overlapping
                and record_assignments[record["recording_id"]] == "train"]
    by_patient = {patient: "test" if "test" in splits else "train"
                  for patient, splits in by_patient.items()}
    if set(by_patient.values()) != {"train", "test"}:
        raise DataValidationError("Split oficial precisa conter pacientes de treino e teste.")
    return by_patient, excluded


def _split_patients(records: list[dict], official: Path | None, seed: int,
                    overlap_policy: str) -> tuple[dict, str, list]:
    patients = sorted({r["patient_id"] for r in records})
    if len(patients) < 3:
        raise DataValidationError("São necessários ao menos três pacientes para treino/validação/teste.")
    rng = random.Random(seed)
    excluded = []
    if official is not None:
        assignment, excluded = _official_patient_split(official, records, overlap_policy)
        train = sorted(p for p, split in assignment.items() if split == "train")
        if len(train) < 2:
            raise DataValidationError("Treino oficial precisa permitir validação por paciente.")
        protocol = ("official_test_overlap_train_excluded_validation_20pct_of_training"
                    if excluded else "official_test_patient_disjoint_validation_20pct_of_training")
    else:
        rng.shuffle(patients)
        test_count = max(1, round(len(patients) * 0.2))
        assignment = {p: "test" for p in patients[:test_count]}
        train = sorted(patients[test_count:])
        assignment.update({p: "train" for p in train})
        protocol = "seeded_patient_split_64pct_train_16pct_validation_20pct_test"
    rng.shuffle(train)
    validation_count = max(1, min(len(train) - 1, round(len(train) * 0.2)))
    for patient in train[:validation_count]:
        assignment[patient] = "validation"
    return assignment, protocol, excluded


def _pcm_sha256(audio: Path) -> tuple[str, dict]:
    """Hash canonical decoded PCM plus rate/channels, not just container bytes."""
    import numpy as np
    import soundfile as sf

    digest = hashlib.sha256()
    try:
        with sf.SoundFile(audio) as handle:
            if handle.frames < 1 or handle.samplerate < 1:
                raise DataValidationError(f"Áudio vazio: {audio.name}")
            info = {"sample_rate_hz": int(handle.samplerate), "channels": int(handle.channels),
                    "duration_s": float(handle.frames / handle.samplerate)}
            digest.update(f"{handle.samplerate}:{handle.channels}:".encode("ascii"))
            for block in handle.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise DataValidationError(f"Áudio não finito: {audio.name}")
                digest.update(np.asarray(block, dtype="<f4").tobytes())
    except (RuntimeError, OSError) as exc:
        raise DataValidationError(f"Áudio ilegível: {audio.name}: {exc}") from exc
    return digest.hexdigest(), info


def build_manifest(root: str | Path, output_csv: str | Path,
                   official_split_path: str | Path | None = None, seed: int = 42,
                   official_overlap_policy: str = "error"):
    """Build CSV + sibling audit JSON; bad annotation rows are excluded explicitly.

    Invalid official splits and duplicate decoded audio across patient IDs are fatal.
    Corrupt recordings and invalid annotation rows are retained in the audit exclusions.
    """
    import pandas as pd
    import numpy as np
    import soundfile as sf

    if official_overlap_policy not in {"error", "exclude_train"}:
        raise DataValidationError("official_overlap_policy deve ser 'error' ou 'exclude_train'.")
    root, output = Path(root).resolve(), Path(output_csv).resolve()
    records = discover_records(root)
    official = Path(official_split_path).resolve() if official_split_path else None
    if official is None:
        candidates = sorted(root.rglob("ICBHI_challenge_train_test.txt"))
        if candidates:
            official = candidates[0]
    if official is not None and not official.is_file():
        raise DataValidationError(f"Split oficial inexistente: {official}")
    assignment, protocol, overlap_excluded = _split_patients(records, official, seed, official_overlap_policy)
    raw_cycle_counts = {record["recording_id"]: sum(bool(line.strip()) for line in
                        Path(record["annotation_path"]).read_text(encoding="utf-8-sig").splitlines())
                        for record in records}
    for exclusion in overlap_excluded:
        exclusion["excluded_cycle_count"] = raw_cycle_counts[exclusion["recording_id"]]
    excluded_ids = {row["recording_id"] for row in overlap_excluded}
    rows, excluded, duplicate_recordings = [], list(overlap_excluded), []
    digests: dict[str, dict] = {}
    for record in records:
        if record["recording_id"] in excluded_ids:
            continue
        try:
            audio_digest, info = _pcm_sha256(Path(record["audio_path"]))
        except DataValidationError as exc:
            excluded.append({"recording_id": record["recording_id"], "reason": str(exc)})
            continue
        if audio_digest in digests:
            original = digests[audio_digest]
            if original["patient_id"] != record["patient_id"]:
                raise DataValidationError("Conteúdo de áudio duplicado entre pacientes: "
                                          f"{original['recording_id']} / {record['recording_id']}")
            duplicate_recordings.append({"recording_id": record["recording_id"],
                                         "same_as": original["recording_id"]})
            continue
        digests[audio_digest] = record
        text = Path(record["annotation_path"]).read_text(encoding="utf-8-sig")
        nonempty = [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
        if not nonempty:
            excluded.append({"recording_id": record["recording_id"], "reason": "Anotação vazia"})
        for number, line in nonempty:
            try:
                row = _parse_row(line, info["duration_s"])
                cycle_duration = row["end_s"] - row["start_s"]
                if cycle_duration < 0.1 - 1e-9 or cycle_duration > 20.0 + 1e-9:
                    raise DataValidationError("Duração do ciclo fora do intervalo suportado de 0,1 a 20 s.")
                first = round(row["start_s"] * info["sample_rate_hz"])
                last = round(row["end_s"] * info["sample_rate_hz"])
                samples, _ = sf.read(record["audio_path"], start=first, stop=last,
                                     dtype="float32", always_2d=True)
                mono = samples.mean(axis=1)
                if len(mono) < 2 or np.sqrt(np.mean(mono.astype(np.float64) ** 2)) < 1e-7:
                    raise DataValidationError("Ciclo silencioso ou amplitude insuficiente para análise.")
            except DataValidationError as exc:
                excluded.append({"recording_id": record["recording_id"], "line": number,
                                 "annotation": line, "reason": str(exc)})
                continue
            rows.append({"dataset_id": "ICBHI2017", **record, **row,
                         "cycle_id": f"{record['recording_id']}:{number}",
                         "split": assignment[record["patient_id"]],
                         "audio_sha256": audio_digest, **info})
    if not rows:
        raise DataValidationError("Nenhum ciclo válido após auditoria.")
    frame = pd.DataFrame(rows)
    if (frame.groupby("patient_id")["split"].nunique() > 1).any():
        raise DataValidationError("Sobreposição de pacientes entre partições.")
    if (frame.groupby("audio_sha256")["split"].nunique() > 1).any():
        raise DataValidationError("Conteúdo de áudio sobreposto entre partições.")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise DataValidationError("Alguma partição ficou sem ciclos válidos.")
    counts = {}
    for split, subset in frame.groupby("split", sort=True):
        counts[split] = {"patients": int(subset.patient_id.nunique()),
                         "recordings": int(subset.recording_id.nunique()),
                         "cycles": int(len(subset)),
                         "classes": {str(k): int(v) for k, v in sorted(Counter(subset.label).items())}}
    audit = {"dataset_id": "ICBHI2017", "seed": int(seed), "split_protocol": protocol,
             "official_split_path": str(official) if official else None,
             "official_overlap_policy": official_overlap_policy,
             "official_overlap_patients": sorted({x["patient_id"] for x in overlap_excluded}),
             "official_overlap_train_exclusions": overlap_excluded,
             "raw_recordings": len(records), "raw_patients": len(assignment),
             "raw_cycles": sum(raw_cycle_counts.values()),
             "excluded_cycle_count": sum(raw_cycle_counts.values()) - len(frame),
             "official_overlap_excluded_cycle_count": sum(x["excluded_cycle_count"] for x in overlap_excluded),
             "valid_cycles": len(frame), "splits": counts, "excluded": excluded,
             "duplicate_recordings_removed": duplicate_recordings,
             "pcm_duplicate_check": "sha256_float32_little_endian_with_samplerate_and_channels",
             "patient_overlap": False, "audio_content_overlap": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.attrs["audit"] = audit
    return frame
