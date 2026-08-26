"""URI-addressed storage, so no code in this repository knows what a cloud is.

Every path the experiments read or write goes through here. A path may be a plain
local directory or any URI that ``fsspec`` can open:

    ./results                     local, relative
    /mnt/shared/results           local, absolute
    file:///mnt/shared/results    the same thing, spelled as a URI
    s3://bucket/prefix            AWS S3 -- or MinIO / Ceph / R2 with AWS_ENDPOINT_URL
    gs://bucket/prefix            Google Cloud Storage
    az://container/prefix         Azure Blob Storage

There is no provider-specific branch anywhere below. S3 works because ``fsspec`` has
an S3 driver, not because this module knows anything about S3, which is why a
provider this file has never heard of works too as long as it ships an fsspec driver.

Credentials are never read, stored, or forwarded by this module. They are resolved by
each provider's own standard mechanism -- environment variables, ``~/.aws/credentials``,
a workload identity, an attached instance role. Keep it that way: a repository that
reads a credential is a repository that can leak one.

Local paths take a fast path that avoids importing ``fsspec`` at all, so the default
reference run has no cloud dependencies to install.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

# A URI we should hand to fsspec rather than treat as a filesystem path. Note that
# "C:\..." is not one, hence requiring more than one character before the colon.
_REMOTE_PREFIXES = ("s3://", "gs://", "gcs://", "az://", "abfs://", "adl://",
                    "http://", "https://", "hdfs://", "sftp://", "ssh://", "memory://")


def is_remote(uri: str) -> bool:
    """True if *uri* needs fsspec. ``file://`` is stripped and handled locally."""
    return str(uri).startswith(_REMOTE_PREFIXES)


def _strip_file_scheme(uri: str) -> str:
    return uri[len("file://"):] if str(uri).startswith("file://") else str(uri)


def _fs(uri: str):
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - only hit on a remote URI
        raise SystemExit(
            f"'{uri}' is a remote URI, which needs fsspec plus that provider's driver:\n"
            "    pip install fsspec s3fs      # for s3://\n"
            "    pip install fsspec gcsfs     # for gs://\n"
            "    pip install fsspec adlfs     # for az://\n"
            "Or point RESULTS_DIR at a local directory to avoid the dependency."
        ) from exc
    return fsspec.filesystem(uri.split("://", 1)[0])


def join(base: str, *parts: str) -> str:
    """Join URI or path components. Unlike ``os.path.join`` this never converts the
    separator on Windows, because ``s3://b\\k`` is not a key."""
    out = str(base).rstrip("/")
    for p in parts:
        out = f"{out}/{str(p).strip('/')}"
    return out


def exists(uri: str) -> bool:
    if is_remote(uri):
        return _fs(uri).exists(uri)
    return Path(_strip_file_scheme(uri)).exists()


def makedirs(uri: str) -> None:
    """Ensure *uri* can be written to. A no-op on object stores, which have no
    directories -- creating one would write a stray zero-byte key."""
    if is_remote(uri):
        return
    Path(_strip_file_scheme(uri)).mkdir(parents=True, exist_ok=True)


def open_file(uri: str, mode: str = "rb"):
    """Open *uri*. Parent directories are created for local writes."""
    if is_remote(uri):
        return _fs(uri).open(uri, mode)
    p = Path(_strip_file_scheme(uri))
    if any(c in mode for c in "wax"):
        p.parent.mkdir(parents=True, exist_ok=True)
    return p.open(mode)


def read_text(uri: str) -> str:
    with open_file(uri, "rb") as fh:
        return fh.read().decode("utf-8")


def write_text(uri: str, text: str) -> None:
    with open_file(uri, "wb") as fh:
        fh.write(text.encode("utf-8"))


def append_jsonl(uri: str, record: dict) -> None:
    """Append one JSON record.

    Object stores cannot append, so a remote target is read-modify-written. That is
    fine for a results log written once per cell (minutes apart), and wrong for a
    hot loop. Two shards must therefore never share a log URI -- they don't:
    ``run_shard.sh`` gives each shard its own suffixed file and ``collect.sh`` merges
    them, which is also what makes the sharding coordination-free.
    """
    line = json.dumps(record, sort_keys=True)
    if is_remote(uri):
        prior = read_text(uri) if exists(uri) else ""
        if prior and not prior.endswith("\n"):
            prior += "\n"
        write_text(uri, prior + line + "\n")
    else:
        with open_file(uri, "a") as fh:
            fh.write(line + "\n")


def read_jsonl(uri: str) -> list[dict]:
    if not exists(uri):
        return []
    return [json.loads(l) for l in read_text(uri).splitlines() if l.strip()]


def listdir(uri: str, suffix: str = "") -> list[str]:
    """Immediate children of *uri*, as full URIs, optionally filtered by suffix."""
    if is_remote(uri):
        names = _fs(uri).ls(uri, detail=False)
        scheme = uri.split("://", 1)[0]
        out = [n if "://" in n else f"{scheme}://{n}" for n in names]
    else:
        base = Path(_strip_file_scheme(uri))
        out = [str(p) for p in sorted(base.iterdir())] if base.is_dir() else []
    return [p for p in out if p.endswith(suffix)]


def get(src_uri: str, dst_path: str | Path) -> Path:
    """Fetch *src_uri* to a local path. A no-op copy if it is already local."""
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_remote(src_uri):
        _fs(src_uri).get(src_uri, str(dst), recursive=True)
    else:
        s = Path(_strip_file_scheme(src_uri))
        if s.resolve() != dst.resolve():
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, dst)
    return dst


def put(src_path: str | Path, dst_uri: str) -> str:
    """Upload a local file or directory to *dst_uri*."""
    if is_remote(dst_uri):
        _fs(dst_uri).put(str(src_path), dst_uri, recursive=True)
    else:
        d = Path(_strip_file_scheme(dst_uri))
        s = Path(src_path)
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.resolve() != d.resolve():
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, d, dirs_exist_ok=True) \
                if s.is_dir() else shutil.copy2(s, d)
    return dst_uri


def results_dir() -> str:
    """The run's output root. Set ``RESULTS_DIR`` to redirect it anywhere."""
    return os.environ.get("RESULTS_DIR", "./results")


def check(uri: str) -> int:
    """Round-trip a small object to prove the URI is reachable and writable.

    Worth doing before a run that will cost GPU-hours: a typo'd bucket name or a
    missing credential otherwise surfaces only when the first cell tries to write.
    """
    probe = join(uri, ".storage_check")
    try:
        makedirs(uri)
        write_text(probe, "ok\n")
        got = read_text(probe).strip()
        if got != "ok":
            print(f"FAIL {uri}: wrote 'ok', read back {got!r}", file=sys.stderr)
            return 1
        if is_remote(uri):
            _fs(uri).rm(probe)
        else:
            Path(_strip_file_scheme(probe)).unlink()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL {uri}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    kind = "remote" if is_remote(uri) else "local"
    print(f"OK   {uri}  ({kind}, readable and writable)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", metavar="URI", default=None,
                    help="round-trip a probe object; defaults to $RESULTS_DIR")
    args = ap.parse_args()
    raise SystemExit(check(args.check or results_dir()))
