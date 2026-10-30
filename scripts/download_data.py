"""Download the public ICBHI archive over verified HTTPS and check published hash.

Uses a versioned Harvard Dataverse mirror because the official site's certificate
was expired at acquisition. Range requests make large downloads resumable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

SOURCE_URL = "https://dataverse.harvard.edu/api/access/datafile/7127117"
SOURCE_DOI = "https://doi.org/10.7910/DVN/HT6PKI"
EXPECTED_SIZE = 1978998275
EXPECTED_MD5 = "f6a5d0ea40bbcd25e17754f9d5ddae48"
SPLIT_URL = "https://raw.githubusercontent.com/jaygshah/ICBHI/c88d99c53f1da0094bf9fec65fdd79a1c0f5f7ad/ICBHI_challenge_train_test.txt"
EXPECTED_SPLIT_SHA256 = "e774311185c42bb0242e3bc0fc4a06ee192b6a180ba668ddefbfdbd1e267bfbe"


def checksum(path: Path, algorithm="sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def _curl(url: str, destination: Path, byte_range=None):
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl é necessário para o download HTTPS.")
    command = [executable, "--fail", "--location", "--silent", "--show-error",
               "--retry", "3", "--retry-delay", "1", "--connect-timeout", "20",
               "--max-time", "180", "--output", str(destination)]
    if byte_range:
        command += ["--range", f"{byte_range[0]}-{byte_range[1]}"]
    command += [url]
    subprocess.run(command, check=True, capture_output=True)


def download(root: Path, workers=6, chunk_mb=8, extract=True) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ICBHI_final_database.zip"
    progress = root / "download_progress.json"
    if not archive.exists():
        chunks = root / ".download_chunks"
        chunks.mkdir(exist_ok=True)
        size = chunk_mb * 1024 * 1024
        tasks = [(i, start, min(start + size, EXPECTED_SIZE) - 1)
                 for i, start in enumerate(range(0, EXPECTED_SIZE, size))]

        def worker(task):
            index, start, end = task
            part = chunks / f"{index:05d}.part"
            if not part.exists() or part.stat().st_size != end - start + 1:
                _curl(SOURCE_URL, part, (start, end))
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"Servidor não respeitou faixa {index}.")
            return part.stat().st_size

        total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, future in enumerate(concurrent.futures.as_completed(
                    [pool.submit(worker, task) for task in tasks]), 1):
                total += future.result()
                state = {"completed_chunks": completed, "total_chunks": len(tasks),
                         "completed_bytes": total, "expected_bytes": EXPECTED_SIZE}
                progress.write_text(json.dumps(state), encoding="utf-8")
                print(json.dumps(state), flush=True)
        joined = archive.with_suffix(".zip.joining")
        with joined.open("wb") as output:
            for index, _, _ in tasks:
                with (chunks / f"{index:05d}.part").open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        joined.replace(archive)
    if archive.stat().st_size != EXPECTED_SIZE or checksum(archive, "md5") != EXPECTED_MD5:
        raise RuntimeError("Tamanho ou MD5 do ZIP difere do checksum publicado pelo Dataverse.")
    archive_sha = checksum(archive)
    if extract:
        with zipfile.ZipFile(archive) as zipped:
            bad = zipped.testzip()
            if bad:
                raise RuntimeError(f"CRC inválido no ZIP: {bad}")
            for member in zipped.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
                    raise RuntimeError("Caminho inseguro no ZIP.")
                if member.is_dir():
                    continue
                target = root.joinpath(*path.parts).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError("Destino de extração fora da pasta de dados.")
                if (target.exists() and target.stat().st_size == member.file_size
                        and crc32(target) == member.CRC):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    split_file = root / "ICBHI_challenge_train_test.txt"
    if not split_file.exists():
        _curl(SPLIT_URL, split_file)
    if checksum(split_file) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("SHA-256 da lista oficial difere do espelho fixado.")
    metadata = {
        "dataset": "ICBHI 2017 Respiratory Sound Database",
        "original_source": "https://bhichallenge.med.auth.gr/",
        "download_source": SOURCE_URL, "mirror_doi": SOURCE_DOI,
        "mirror_file_id": 7127117, "mirror_version": "1.0 (2023-05-30)",
        "archive_bytes": EXPECTED_SIZE, "archive_md5": EXPECTED_MD5,
        "archive_sha256": archive_sha, "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transport": "HTTPS with certificate verification; no insecure TLS option",
        "official_source_status": "Certificate expired at acquisition; archive obtained from Dataverse",
        "split_source": SPLIT_URL, "split_sha256": checksum(split_file),
        "split_note": "Mirror of the official per-recording list; patient disjointness validated during ingestion",
        "terms": "Original portal: freely available for research; cite Rocha et al. 2019. Mirror lists CC0; do not treat mirror metadata as overriding original terms.",
        "redistribution": "Do not bundle the complete dataset in the application release; downloader and provenance are supplied.",
        "citation_doi": "10.1088/1361-6579/ab03ea",
    }
    (root / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/icbhi"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12 or not 1 <= args.chunk_mb <= 32:
        parser.error("Use 1–12 workers e 1–32 MB por parte.")
    metadata = download(args.output, args.workers, args.chunk_mb, not args.skip_extract)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
