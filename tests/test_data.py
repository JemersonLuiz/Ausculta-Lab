"""Synthetic fixtures exercise parsing/splitting only, never research accuracy."""

import pandas as pd
import json
import pytest

from cardio_audio.data import build_manifest, discover_records, parse_annotations


def test_discovery_pairs_audio_with_its_annotation(icbhi_fixture):
    records = discover_records(icbhi_fixture)
    assert len(records) == 32
    assert len({str(record["patient_id"]) for record in records}) == 16
    assert all(record["recording_id"] for record in records)
    assert all(str(record["audio_path"]).endswith(".wav") for record in records)
    assert all(str(record["annotation_path"]).endswith(".txt") for record in records)


def test_manifest_maps_labels_and_keeps_all_cycles(icbhi_fixture, tmp_path):
    destination = tmp_path / "manifest.csv"
    frame = build_manifest(icbhi_fixture, destination, seed=42)
    assert destination.is_file()
    assert len(frame) == 128
    assert set(frame["label"]) == {0, 1, 2, 3}
    assert frame.groupby("label").size().to_dict() == {0: 32, 1: 32, 2: 32, 3: 32}
    required = {"dataset_id", "patient_id", "recording_id", "audio_path", "start_s", "end_s", "label", "split"}
    assert required.issubset(frame.columns)
    assert (frame["end_s"] > frame["start_s"]).all()
    assert set(frame["split"]) == {"train", "validation", "test"}


def test_patient_never_crosses_partitions(icbhi_fixture, tmp_path):
    frame = build_manifest(icbhi_fixture, tmp_path / "manifest.csv", seed=42)
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    groups = {split: set(part.patient_id.astype(str)) for split, part in frame.groupby("split")}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]


def test_same_seed_produces_same_partition(icbhi_fixture, tmp_path):
    first = build_manifest(icbhi_fixture, tmp_path / "first.csv", seed=13)
    second = build_manifest(icbhi_fixture, tmp_path / "second.csv", seed=13)
    cols = ["patient_id", "recording_id", "start_s", "end_s", "label", "split"]
    pd.testing.assert_frame_equal(
        first[cols].sort_values(cols).reset_index(drop=True),
        second[cols].sort_values(cols).reset_index(drop=True),
    )


def test_out_of_recording_annotation_does_not_enter_training(icbhi_fixture, tmp_path):
    annotation = next(icbhi_fixture.glob("*.txt"))
    annotation.write_text("0\t200\t1\t0\n", encoding="utf-8")
    try:
        frame = build_manifest(icbhi_fixture, tmp_path / "manifest.csv", seed=42)
    except ValueError:
        return
    assert not (frame["recording_id"].astype(str) == annotation.stem).any(), "Invalid annotated cycle entered training."


def test_nonbinary_label_does_not_enter_training(icbhi_fixture, tmp_path):
    annotation = next(icbhi_fixture.glob("*.txt"))
    annotation.write_text("0\t1\t5\t0\n", encoding="utf-8")
    try:
        frame = build_manifest(icbhi_fixture, tmp_path / "manifest.csv", seed=42)
    except ValueError:
        return
    assert not (frame["recording_id"].astype(str) == annotation.stem).any(), "Non-binary annotation was silently relabelled."


@pytest.mark.parametrize(
    "annotation",
    ["-1\t1\t0\t0\n", "1\t1\t0\t0\n", "2\t1\t0\t0\n",
     "0\t5\t0\t0\n", "nan\t1\t0\t0\n", "0\tinf\t0\t0\n",
     "0\t1\t2\t0\n", "0\t1\t0\t-1\n", "0\t1\t0\n"],
)
def test_invalid_annotation_rows_are_rejected(tmp_path, annotation):
    path = tmp_path / "invalid.txt"
    path.write_text(annotation, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_annotations(path, duration_s=4.0)


def test_parser_preserves_time_and_label_semantics(tmp_path):
    path = tmp_path / "valid.txt"
    path.write_text("0\t1\t0\t0\n1\t2\t1\t0\n2\t3\t0\t1\n3\t4\t1\t1\n", encoding="utf-8")
    rows = parse_annotations(path, duration_s=4.0)
    assert [(row["start_s"], row["end_s"], row["label"]) for row in rows] == [
        (0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 2), (3.0, 4.0, 3)]


def test_official_test_patients_remain_reserved(icbhi_fixture, tmp_path):
    official = tmp_path / "official.txt"
    entries = []
    for path in sorted(icbhi_fixture.glob("*.wav")):
        patient = int(path.stem.split("_")[0])
        entries.append(f"{path.name}\t{'train' if patient <= 108 else 'test'}")
    official.write_text("\n".join(entries) + "\n", encoding="utf-8")
    frame = build_manifest(icbhi_fixture, tmp_path / "manifest.csv", official_split_path=official, seed=42)
    test_patients = set(frame.loc[frame["split"] == "test", "patient_id"].astype(str))
    assert test_patients == {str(patient) for patient in range(109, 117)}
    assert not set(frame.loc[frame["split"] == "train", "patient_id"].astype(str)) & test_patients


def test_conflicting_official_split_is_rejected(icbhi_fixture, tmp_path):
    official = tmp_path / "official.txt"
    entries = []
    for path in sorted(icbhi_fixture.glob("*.wav")):
        patient = int(path.stem.split("_")[0])
        split = "train" if patient <= 108 else "test"
        if path.stem.startswith("101_2"):
            split = "test"
        entries.append(f"{path.name}\t{split}")
    official.write_text("\n".join(entries) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_manifest(icbhi_fixture, tmp_path / "manifest.csv", official_split_path=official, seed=42)


def test_identical_audio_under_different_patient_ids_is_rejected(icbhi_fixture, tmp_path):
    paths = sorted(icbhi_fixture.glob("*.wav"))
    original = paths[0]
    other_patient = next(path for path in paths if path.stem.split("_")[0] != original.stem.split("_")[0])
    other_patient.write_bytes(original.read_bytes())
    with pytest.raises(ValueError):
        build_manifest(icbhi_fixture, tmp_path / "manifest.csv", seed=42)


def test_explicit_overlap_policy_preserves_official_test_and_audits_train_exclusion(icbhi_fixture, tmp_path):
    official = tmp_path / "official.txt"
    entries = []
    for path in sorted(icbhi_fixture.glob("*.wav")):
        patient = int(path.stem.split("_")[0])
        split = "train" if patient <= 108 else "test"
        if path.stem.startswith("101_2"):
            split = "test"
        entries.append(f"{path.name}\t{split}")
    official.write_text("\n".join(entries) + "\n", encoding="utf-8")
    output = tmp_path / "manifest.csv"
    frame = build_manifest(icbhi_fixture, output, official_split_path=official,
                           official_overlap_policy="exclude_train", seed=42)
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    assert "101_1b1_Al_sc_Meditron" not in set(frame.recording_id)
    held_out = frame[frame.recording_id == "101_2b1_Al_sc_Meditron"]
    assert len(held_out) == 4
    assert set(held_out.split) == {"test"}
    test_patients = set(frame.loc[frame["split"] == "test", "patient_id"].astype(str))
    assert test_patients == {"101"} | {str(patient) for patient in range(109, 117)}
    audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
    assert audit["official_overlap_policy"] == "exclude_train"
    assert list(map(str, audit["official_overlap_patients"])) == ["101"]
    exclusions = audit["official_overlap_train_exclusions"]
    assert len(exclusions) == 1
    assert exclusions[0]["recording_id"] == "101_1b1_Al_sc_Meditron"
    assert exclusions[0]["original_split"] == "train"
