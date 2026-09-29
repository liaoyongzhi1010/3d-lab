"""Deterministic provenance manifests that never overwrite prior evidence."""

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_code_hash(repo) -> str:
    repo = Path(repo)
    digest = hashlib.sha256()
    digest.update(_git(repo, "rev-parse", "HEAD").strip())
    digest.update(b"\0tracked-diff\0")
    digest.update(_git(repo, "diff", "--binary", "HEAD", "--"))
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    for encoded in sorted(path for path in untracked if path):
        path = repo / encoded.decode()
        if path.is_file():
            digest.update(b"\0untracked\0" + encoded + b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_manifest(*, code_hash, split_path, checkpoint_path, config, seed):
    if not isinstance(code_hash, str) or not code_hash:
        raise ValueError("code_hash is required")
    if config is None or seed is None:
        raise ValueError("config and seed are required")
    split_path = Path(split_path)
    checkpoint_path = Path(checkpoint_path)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "code_hash": code_hash,
        "config": config,
        "seed": seed,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
    }


def write_manifest_once(path, manifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with path.open("xb") as handle:
        handle.write(payload)
