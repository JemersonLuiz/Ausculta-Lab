"""Verify a release in a new directory and genuinely fresh Python environment.

Only standard-library modules are imported by this orchestrator. Dependencies,
pytest, the light integration retest and the actual server run from the copy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(root: Path) -> dict:
    files: dict[str, str] = {}
    roots = ("src", "web", "scripts", "tests", "models", "examples")
    for directory in roots:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part == "__pycache__" or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".md"}:
                continue
            files[relative.as_posix()] = sha256(path)
    for name in ("run.py", "pyproject.toml", "requirements.lock.txt", "Instalar.ps1", "Iniciar.ps1",
                 "INSTALAR.bat", "INICIAR.bat", "data/manifest.csv", "data/manifest.audit.json",
                 "reports/test_predictions.csv"):
        path = root / name
        if path.is_file():
            files[name] = sha256(path)
    ordered = dict(sorted(files.items()))
    encoded = "\n".join(f"{digest}  {name}" for name, digest in ordered.items()).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": ordered,
            "scope": "code, UI, models, dependencies, launchers, examples, manifests and test predictions; excludes docs/reports except test_predictions.csv"}


def extract_checked(archive: Path, destination: Path) -> tuple[Path, dict]:
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 5000 or sum(member.file_size for member in members) > 512 * 1024 * 1024:
            raise ValueError("Archive exceeds the release verification limits.")
        names: set[str] = set()
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            if not name.parts or name.parts[0] != "AuscultaLab":
                raise ValueError("Unexpected archive prefix.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlinks are not accepted.")
            key = name.as_posix().casefold()
            if key in names:
                raise ValueError("Duplicate/case-colliding ZIP member.")
            names.add(key)
        bad = zipped.testzip()
        if bad:
            raise ValueError(f"Invalid ZIP CRC: {bad}")
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Extraction target escaped destination.")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
    project = destination / "AuscultaLab"
    listed: dict[str, str] = {}
    for line in (project / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or len(digest) != 64 or relative in listed:
            raise ValueError("Invalid checksum manifest entry.")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        listed[relative] = digest
    actual = {path.relative_to(project).as_posix() for path in project.rglob("*") if path.is_file()}
    if actual - {"CHECKSUMS.sha256"} != set(listed):
        raise ValueError("Checksum manifest does not cover all archived files.")
    return project, {"zip_crc_passed": True, "safe_paths": True, "checked_files": len(listed)}


def run_logged(command: list[str], cwd: Path, env: dict, log: Path, timeout: int = 1200) -> dict:
    print(f"verify: {log.stem}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        output.write(json.dumps(command, ensure_ascii=False) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    evidence = {"command": command, "log": str(log), "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 3)}
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")
    return evidence


def http_request(url: str, payload: bytes | None = None, headers: dict | None = None):
    request = urllib.request.Request(url, data=payload, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read(), dict(response.headers)


def multipart(payload: bytes, start, end) -> tuple[bytes, dict]:
    boundary = "AuscultaQA" + uuid.uuid4().hex
    parts = []
    for name, value in (("start", start), ("end", end)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="example.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    parts.extend((payload, f"\r\n--{boundary}--\r\n".encode()))
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def test_live_server(python: Path, project: Path, env: dict, log: Path, preferred_port: int) -> dict:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred_port))
        except OSError:
            probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    server = None
    result: dict = {"url": base, "log": str(log)}
    with log.open("wb") as output:
        try:
            server = subprocess.Popen([str(python), "run.py", "--port", str(port)], cwd=project, env=env,
                                      stdout=output, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            result["pid"] = server.pid
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("The copied application server exited before startup.")
                try:
                    status, body, _ = http_request(base + "/api/status")
                    if status == 200 and json.loads(body)["model_ready"]:
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Copied server did not become ready within 60 seconds.")
                time.sleep(0.5)
            status, html, headers = http_request(base + "/")
            assert status == 200 and b"<html" in html.lower() and "text/html" in headers.get("content-type", headers.get("Content-Type", ""))
            for asset in ("/style.css", "/app.js"):
                assert http_request(base + asset)[0] == 200
            examples = json.loads(http_request(base + "/api/examples")[1])
            assert examples
            example = examples[0]
            payload = http_request(base + f"/api/examples/{example['id']}/audio")[1]
            request_body, request_headers = multipart(payload, example["start"], example["end"])
            first = json.loads(http_request(base + "/api/analyze", request_body, request_headers)[1])
            second = json.loads(http_request(base + "/api/analyze", request_body, request_headers)[1])
            assert first["prediction"] == second["prediction"]
            assert len(first["prediction"]["scores"]) == 4
            assert abs(sum(row["score"] for row in first["prediction"]["scores"]) - 1) < 1e-6
            bad_body, bad_headers = multipart(b"not-a-wav", 0, 1)
            try:
                http_request(base + "/api/analyze", bad_body, bad_headers)
                raise AssertionError("Invalid file unexpectedly accepted over real HTTP.")
            except urllib.error.HTTPError as error:
                assert error.code == 422
                assert "prediction" not in json.loads(error.read())
            result.update({"root_status": 200, "status_model_ready": True, "static_assets_ok": True,
                           "example_id": example["id"], "prediction_repeated_identically": True,
                           "invalid_upload_status": 422, "passed": True})
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            result["own_server_stopped"] = server is not None and server.poll() is not None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "release" / "AuscultaLab-v1.1.0.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "qa" / "pacote_extraido.json")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_fingerprint = fingerprint(ROOT)
    evidence = {"started_utc": datetime.now(timezone.utc).isoformat(), "archive": str(args.archive.resolve()),
                "archive_sha256": sha256(args.archive), "source_fingerprint": original_fingerprint,
                "steps": [], "passed": False}
    try:
        # Short, direct path: long nested project paths can prevent torch from
        # installing on Windows without LongPaths enabled. No junction/registry
        # changes are used; files and the venv belong to this new TEMP directory.
        destination = Path(tempfile.mkdtemp(prefix="al-")).resolve()
        logs = ROOT / "reports" / "qa" / ("package-check-" + destination.name)
        logs.mkdir(parents=True)
        evidence["logs_directory"] = str(logs)
        evidence["workspace_policy"] = "Fresh direct short TEMP extraction and in-copy venv; logs retained in source reports/qa."
        project, evidence["archive_integrity"] = extract_checked(args.archive, destination)
        evidence["extracted_project"] = str(project)
        evidence["execution_project"] = str(project)
        evidence["copied_fingerprint"] = fingerprint(project)
        assert evidence["copied_fingerprint"]["sha256"] == original_fingerprint["sha256"], "Archive runtime differs from current source."
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "AUSCULTA_ROOT", "VIRTUAL_ENV"):
            env.pop(name, None)
        env.update({"PYTHONNOUSERSITE": "1", "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        venv = project / ".venv"
        assert not venv.exists()
        base_python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        evidence["base_python"] = str(base_python)
        evidence["steps"].append(run_logged([str(base_python), "-m", "venv", str(venv)], project, env, logs / "01-create-venv.log"))
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        evidence["steps"].append(run_logged([str(python), "-m", "pip", "install", "-r", "requirements.lock.txt"], project, env, logs / "02-install-lock.log"))
        evidence["steps"].append(run_logged([str(python), "-m", "pip", "install", "--no-deps", "-e", "."], project, env, logs / "03-install-project.log"))
        evidence["steps"].append(run_logged([str(python), "-m", "pip", "check"], project, env, logs / "04-pip-check.log"))
        proof = "import cardio_audio,json,sys; print(json.dumps({'module':cardio_audio.__file__,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':sys.version}))"
        evidence["steps"].append(run_logged([str(python), "-c", proof], project, env, logs / "05-import-proof.log"))
        imported = json.loads((logs / "05-import-proof.log").read_text(encoding="utf-8").splitlines()[-1])
        assert Path(imported["module"]).resolve().is_relative_to((project / "src").resolve())
        assert Path(imported["prefix"]).resolve() == venv.resolve()
        assert Path(imported["base_prefix"]).resolve() != venv.resolve()
        assert imported["version"].startswith("3.13.")
        evidence["import_proof"] = imported
        junit = project / "reports" / "qa" / "pacote_novo_ambiente.xml"
        evidence["steps"].append(run_logged([str(python), "-m", "pytest", "tests", "-q", "--require-real-artifacts", f"--junitxml={junit}"], project, env, logs / "06-pytest.log"))
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {field: sum(int(suite.attrib.get(field, 0)) for suite in suites) for field in ("tests", "failures", "errors", "skipped")}
        assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0
        evidence["pytest"] = counts
        light_output = project / "reports" / "qa" / "pacote_reteste_leve.json"
        evidence["steps"].append(run_logged([str(python), "scripts/retest.py", "--output", str(light_output)], project, env, logs / "07-light-retest.log"))
        evidence["light_retest"] = json.loads(light_output.read_text(encoding="utf-8"))
        assert evidence["light_retest"]["passed"] is True
        print("verify: live-server", flush=True)
        evidence["live_server"] = test_live_server(python, project, env, logs / "08-live-server.log", args.port)
        assert evidence["live_server"]["own_server_stopped"]
        assert fingerprint(ROOT)["sha256"] == original_fingerprint["sha256"], "Runtime source changed during verification."
        assert fingerprint(project)["sha256"] == original_fingerprint["sha256"], "Copied runtime changed during verification."
        evidence["passed"] = True
    except Exception as error:
        evidence["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"passed": evidence["passed"], "report": str(args.output),
                          "error": evidence.get("error")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
