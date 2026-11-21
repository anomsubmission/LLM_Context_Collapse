"""Immutable experiment directories and credential-free scientific provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_SECRET_KEYS = {"api_key", "hf_token", "access_token", "token", "password",
                "authorization", "secret", "credentials", "credential"}
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def scrub_secrets(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower() in _SECRET_KEYS else scrub_secrets(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scrub_secrets(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record():
    versions = {}
    for package in ("torch", "transformers", "peft", "datasets", "numpy", "matplotlib",
                    "accelerate", "bitsandbytes", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
        # Include new implementation files as well as tracked changes.
        sources = {str(p.relative_to(repo)): file_hash(p)
                   for sub in ("cliff", "tests", "configs")
                   for p in (repo / sub).rglob("*")
                   if p.is_file() and p.suffix in {".py", ".json", ".toml", ".yaml"}}
        source_digest = content_hash(sources)
    except (OSError, subprocess.SubprocessError):
        commit, patch, source_digest = None, b"", None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "code_commit": commit, "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
            "source_tree_sha256": source_digest}


def _write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_sources(destination):
    """Retain the executable dirty-tree sources, not only irretrievable hashes."""
    repo = Path(__file__).resolve().parent.parent
    destination = Path(destination)
    files = [p for directory in ("cliff", "tests", "configs")
             for p in (repo / directory).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json", ".toml", ".yaml"}]
    files += [repo / name for name in ("pyproject.toml",) if (repo / name).exists()]
    hashes = {}
    for source in sorted(files):
        relative = source.relative_to(repo)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        target.write_bytes(payload)
        hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    _write_json(destination / "source_manifest.json", hashes)
    return {"directory": str(destination), "files": len(hashes),
            "manifest_sha256": file_hash(destination / "source_manifest.json")}


class RunWriter:
    """Create a new run; reuse/overwrite is intentionally unsupported.

    A complete marker means technical validation passed. Whether the run is
    scientific evidence is a separate, explicit manifest field.
    """

    def __init__(self, root, experiment, config, run_id=None, *, scientific_evidence=False):
        if not _COMPONENT.fullmatch(experiment):
            raise ValueError("experiment must be one path component")
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:10]
        if not _COMPONENT.fullmatch(run_id):
            raise ValueError("run_id must be one path component")
        self.path = Path(root) / experiment / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self._predictions = self.path / "predictions.jsonl"
        self._predictions.touch(exist_ok=False)
        self._closed = False
        self._count = 0
        sanitized = scrub_secrets(config)
        self.manifest = {"schema_version": 1, "run_id": run_id, "experiment": experiment,
                         "status": "running", "scientific_evidence": bool(scientific_evidence),
                         "started_at": datetime.now(timezone.utc).isoformat(),
                         "configuration": sanitized, "config_sha256": content_hash(sanitized),
                         "environment": environment_record()}
        self.manifest["source_snapshot"] = snapshot_sources(self.path / "source_snapshot")
        _write_json(self.path / "manifest.json", self.manifest)

    def write_prediction(self, record):
        if self._closed:
            raise RuntimeError("run is closed")
        value = scrub_secrets(record)
        if value.get("run_id") not in (None, self.manifest["run_id"]):
            raise ValueError("prediction belongs to another run")
        value["run_id"] = self.manifest["run_id"]
        with self._predictions.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
        self._count += 1

    def complete(self, metrics, validation):
        if self._closed:
            raise RuntimeError("run is closed")
        if validation.get("passed") is not True:
            raise ValueError("technical validation must explicitly pass")
        _write_json(self.path / "metrics.json", scrub_secrets(metrics))
        _write_json(self.path / "validation.json", scrub_secrets(validation))
        self.manifest.update({"status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
                              "prediction_count": self._count,
                              "predictions_sha256": file_hash(self._predictions),
                              "metrics_sha256": file_hash(self.path / "metrics.json"),
                              "validation_sha256": file_hash(self.path / "validation.json")})
        _write_json(self.path / "manifest.json", self.manifest)
        self._closed = True

    def fail(self, reason):
        if self._closed:
            raise RuntimeError("run is closed")
        self.manifest.update({"status": "failed", "scientific_evidence": False, "reason": str(reason),
                              "finished_at": datetime.now(timezone.utc).isoformat(),
                              "prediction_count": self._count,
                              "predictions_sha256": file_hash(self._predictions)})
        _write_json(self.path / "manifest.json", self.manifest)
        self._closed = True
