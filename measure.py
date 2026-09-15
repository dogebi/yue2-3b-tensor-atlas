#!/usr/bin/env python3
"""Measure exact per-tensor bytes from HuggingFace safetensors headers over HTTP Range.

Usage:  python3 measure.py <org>/<model> [<org>/<model>-FP8 ...]

No weights are downloaded: each shard's header is range-read (8-byte little-endian length
prefix + JSON of dtype/shape/data_offsets). Some quantized repos ship ONE huge shard, whose
header can exceed a couple of MB, so the range is grown until the header fits.

huggingface.co needs the local proxy (xray: 1099 on the Windows side, 1098 inside WSL2).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

PROXY = os.environ.get("HF_PROXY", "http://127.0.0.1:1099")
OUT = Path(os.environ.get("HF_MEASURE_DIR", "/tmp/hf-measure"))
OUT.mkdir(parents=True, exist_ok=True)
RANGES = ["0-3000000", "0-16000000", "0-64000000"]


def curl(url: str, out: Path, rng: str | None = None) -> bool:
    cmd = ["curl", "-sS", "--max-time", "300", "-x", PROXY, "-L"]
    if rng:
        cmd += ["-r", rng]
    cmd += ["-o", str(out), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def tree(repo: str) -> list[dict]:
    f = OUT / (repo.replace("/", "__") + ".tree.json")
    if not curl(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", f):
        raise SystemExit(f"tree fetch failed: {repo} (is the proxy {PROXY} up?)")
    return json.loads(f.read_text())


def header(repo: str, path: str) -> dict:
    stem = OUT / (repo.replace("/", "__") + "___" + path.replace("/", "_"))
    raw = b""
    for rng in RANGES:
        f = Path(str(stem) + "." + rng.split("-")[1] + ".hdr")
        if not f.exists() or f.stat().st_size == 0:
            if not curl(f"https://huggingface.co/{repo}/resolve/main/{path}", f, rng):
                raise SystemExit(f"header fetch failed: {repo}/{path}")
        raw = f.read_bytes()
        if len(raw) >= 8:
            n = struct.unpack("<Q", raw[:8])[0]
            if n <= len(raw) - 8:
                return json.loads(raw[8 : 8 + n])
    n = struct.unpack("<Q", raw[:8])[0] if len(raw) >= 8 else -1
    raise SystemExit(f"header still truncated for {repo}/{path}: need {n}, have {len(raw) - 8}")


def measure(repo: str) -> dict:
    files = sorted((e for e in tree(repo) if e["type"] == "file" and e["path"].endswith(".safetensors")),
                   key=lambda e: e["path"])
    print(f"== {repo}: {len(files)} shards", flush=True)
    tensors: dict[str, dict] = {}
    for e in files:
        h = header(repo, e["path"])
        cnt = 0
        for name, t in h.items():
            if name == "__metadata__":
                continue
            off = t["data_offsets"]
            if name in tensors:
                raise SystemExit(f"duplicate tensor {name} in {repo}")
            tensors[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "bytes": off[1] - off[0], "shard": e["path"]}
            cnt += 1
        print(f"   {e['path']:34s} {cnt:6d} tensors", flush=True)
    payload = sum(t["bytes"] for t in tensors.values())
    disk = sum((e.get("lfs") or {}).get("size") or e.get("size") or 0 for e in files)
    out = {"repo": repo, "shards": len(files), "tensor_count": len(tensors),
           "payload_bytes": payload, "disk_bytes": disk, "tensors": tensors}
    (OUT / (repo.replace("/", "__") + ".measured.json")).write_text(json.dumps(out, indent=1))
    print(f"   payload {payload / 1e9:.4f} GB · disk {disk / 1e9:.4f} GB · tensors {len(tensors)}", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        measure(repo)
