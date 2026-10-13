"""Generate, fit, draw and publish a self-contained synthetic run."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile

import matplotlib
import numpy as np

from . import __version__
from .data import generate, load_daily, write_daily
from .indices import BASE_F, seasonal_indices, validate_years
from .model import analyze
from .plots import draw

YEARS = tuple(range(2005, 2025))
TRAIN_END = 2018
ARTIFACTS = ("synthetic_daily.csv", "seasonal_indices.csv", "predictions.csv", "results.json",
             "index_relationship.png", "test_tracking_errors.png")
SUPPORTED_VERSIONS = {"0.1.0"}
CODE_FILES = {"__init__.py", "__main__.py", "data.py", "indices.py", "model.py", "plots.py", "run.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, contents: dict):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(contents, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def save_csv(path: Path, rows: list[dict]):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(manifest: dict) -> None:
    required = {"status", "synthetic", "version", "parameters", "runtime", "code_files", "files"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("manifest is missing required metadata")
    if (manifest["status"] != "complete" or manifest["synthetic"] is not True
            or not isinstance(manifest["version"], str) or manifest["version"] not in SUPPORTED_VERSIONS):
        raise ValueError("no complete supported synthetic manifest")

    parameters = manifest["parameters"]
    if not isinstance(parameters, dict) or not {"seed", "base_f", "season_years", "train_end"}.issubset(parameters):
        raise ValueError("manifest is missing run parameters")
    if type(parameters["seed"]) is not int or parameters["seed"] < 0:
        raise ValueError("invalid manifest seed")
    base = parameters["base_f"]
    if type(base) not in (int, float) or not math.isfinite(base):
        raise ValueError("invalid manifest base temperature")
    if not isinstance(parameters["season_years"], list):
        raise ValueError("invalid manifest season years")
    years = validate_years(parameters["season_years"])
    boundary = parameters["train_end"]
    if type(boundary) is not int or not years[0] <= boundary < years[-1]:
        raise ValueError("invalid manifest training boundary")

    runtime = manifest["runtime"]
    if not isinstance(runtime, dict) or any(not isinstance(runtime.get(key), str) or not runtime[key].strip()
                                             for key in ("python", "numpy", "matplotlib")):
        raise ValueError("invalid manifest runtime versions")
    for key, expected in (("code_files", CODE_FILES), ("files", set(ARTIFACTS))):
        values = manifest[key]
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"invalid manifest {key} inventory")
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in values.values()):
            raise ValueError(f"invalid manifest {key} SHA-256")


def verify(directory: Path) -> dict:
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
        for name, digest in manifest["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"output hash mismatch: {name}")
        return manifest
    except (OSError, KeyError, TypeError, AttributeError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError("unreadable or invalid run") from error


def run(output: Path, seed: int = 7) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("output already exists; use a new run name")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        code = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        write_daily(staging / "synthetic_daily.csv", generate(YEARS, seed))
        daily_hash = sha256(staging / "synthetic_daily.csv")
        daily = load_daily(staging / "synthetic_daily.csv")
        indices = seasonal_indices(daily, YEARS)
        results, predictions = analyze(indices, TRAIN_END)
        payload = {"synthetic": True, "seed": seed, "base_f": BASE_F,
                   "daily_rows": len(daily), "paired_season_rows": len(indices), "results": results,
                   "interpretation": "index tracking on invented temperatures; no priced payoff, firm losses or trading costs"}
        save_csv(staging / "seasonal_indices.csv", indices)
        save_csv(staging / "predictions.csv", predictions)
        save_json(staging / "results.json", payload)
        draw(staging, results, predictions)
        if sha256(staging / "synthetic_daily.csv") != daily_hash:
            raise ValueError("synthetic input changed during the run")
        code_after = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        if code_after != code:
            raise ValueError("code changed during the run")
        manifest = {"status": "complete", "synthetic": True, "version": __version__,
                    "parameters": {"seed": seed, "base_f": BASE_F, "season_years": list(YEARS), "train_end": TRAIN_END},
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    "code_files": code, "files": {name: sha256(staging / name) for name in ARTIFACTS}}
        save_json(staging / "run_manifest.json", manifest)
        verify(staging)
        if output.exists():
            raise FileExistsError("output appeared during the run")
        os.rename(staging, output)
        return payload
    except Exception as error:
        marker = staging / "run_manifest.json"
        if marker.exists():
            marker.unlink()
        save_json(staging / "diagnostics.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
