"""Pacote autocontido de fontes/artefatos, sem ambiente ou base completa."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "AuscultaLab"


def release_files():
    files = []
    for name in ("README.md", "PROJETO.md", "pyproject.toml", "requirements.lock.txt", "run.py",
                 "Instalar.ps1", "Iniciar.ps1", "INSTALAR.bat", "INICIAR.bat", ".gitignore"):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(name)
        files.append(path)
    for directory in ("src", "web", "scripts", "tests", "docs", "models", "examples", "reports"):
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in relative.parts) and path.suffix not in {".pyc", ".tmp"}:
                files.append(path)
    for name in ("manifest.csv", "manifest.audit.json"):
        files.append(ROOT / "data" / name)
    for name in ("provenance.json", "dataverse_metadata.json", "ICBHI_challenge_train_test.txt", "official_split_pinned.txt"):
        path = ROOT / "data" / "raw" / "icbhi" / name
        if path.is_file():
            files.append(path)
    # Somente evidências finais selecionadas do navegador.
    preview = ROOT / "output" / "playwright"
    if preview.is_dir():
        files += [path for path in sorted(preview.glob("real-*")) if path.is_file() and path.suffix in {".png", ".json"}]
        if (preview / "ui-real-review.json").is_file():
            files.append(preview / "ui-real-review.json")
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "release" / "AuscultaLab-v1.0.0.zip")
    args = parser.parse_args()
    files = release_files()
    model = json.loads((ROOT / "models" / "model.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "models" / "cnn.pt").read_bytes()).hexdigest() == model["weights_sha256"]
    checksums = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
            archive.write(path, f"{PREFIX}/{relative}")
        archive.writestr(f"{PREFIX}/CHECKSUMS.sha256", "\n".join(checksums) + "\n")
        for name in ("data/processed", "notebooks"):
            archive.writestr(f"{PREFIX}/{name}/", "")
    temporary.replace(args.output)
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert not any("/.venv/" in name or name.endswith("ICBHI_final_database.zip") or "/.download_chunks/" in name for name in archive.namelist())
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(args.output), "files": len(files), "bytes": args.output.stat().st_size, "sha256": checksum}))


if __name__ == "__main__":
    main()
