"""No-network archive safety/integrity tests for the real-data downloader."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture
def downloader():
    spec = importlib.util.spec_from_file_location("qa_download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def local_archive(root, members, downloader, monkeypatch):
    root.mkdir()
    archive = root / "ICBHI_final_database.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, payload in members.items():
            zipped.writestr(name, payload)
    split = b"101_1b1_Al_sc_Meditron train\n102_1b1_Al_sc_Meditron test\n"
    (root / "ICBHI_challenge_train_test.txt").write_bytes(split)
    monkeypatch.setattr(downloader, "EXPECTED_SIZE", archive.stat().st_size)
    monkeypatch.setattr(downloader, "EXPECTED_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(downloader, "EXPECTED_SPLIT_SHA256", hashlib.sha256(split).hexdigest())

    def prohibit_network(*args, **kwargs):
        raise AssertionError("Unit tests must not download any data.")
    monkeypatch.setattr(downloader, "_curl", prohibit_network)
    return archive


@pytest.mark.parametrize("member", ["../escape.txt", "..\\escape.txt", "/escape.txt", "C:/escape.txt"])
def test_archive_cannot_escape_destination(tmp_path, downloader, monkeypatch, member):
    root = tmp_path / "data"
    local_archive(root, {member: b"must not escape"}, downloader, monkeypatch)
    with pytest.raises(RuntimeError):
        downloader.download(root, extract=True)
    assert not (tmp_path / "escape.txt").exists()


def test_archive_hash_mismatch_stops_before_extraction(tmp_path, downloader, monkeypatch):
    root = tmp_path / "data"
    local_archive(root, {"inner/good.txt": b"valid"}, downloader, monkeypatch)
    monkeypatch.setattr(downloader, "EXPECTED_MD5", "0" * 32)
    with pytest.raises(RuntimeError):
        downloader.download(root, extract=True)
    assert not (root / "inner/good.txt").exists()


def test_extraction_repairs_same_size_file_with_wrong_crc(tmp_path, downloader, monkeypatch):
    root = tmp_path / "data"
    local_archive(root, {"inner/good.txt": b"valid"}, downloader, monkeypatch)
    target = root / "inner/good.txt"
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    provenance = downloader.download(root, extract=True)
    assert target.read_bytes() == b"valid"
    assert provenance["transport"].startswith("HTTPS with certificate verification")
    assert (root / "provenance.json").is_file()
