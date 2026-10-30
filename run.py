"""Inicia apenas no computador local por padrão."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ausculta Lab — aplicação acadêmica local")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    print(f"Ausculta Lab: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run("cardio_audio.api:app", host="127.0.0.1", port=args.port, log_level="info")
