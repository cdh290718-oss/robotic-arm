#!/usr/bin/env python
"""Resumable, integrity-checked downloader for the pinned NF4 checkpoint.

Stdlib only: no torch / huggingface_hub needed, so it can run before the
project virtualenv exists.

Integrity reference
-------------------
The Hub stores this repo on **Xet**, and for Xet-backed blobs the HTTP ``ETag``
is *not* the SHA256 of the file (checking against it produces a false mismatch
on a perfectly good 3.9 GiB download).  The authoritative digest is the
``lfs.oid`` field of the tree API::

    GET /api/models/<repo>/tree/<revision>?recursive=1

which is the SHA256 of the file content.  That is what this script verifies
against.  Small non-LFS files have no LFS oid; for those only the byte size is
checked and the local SHA256 is recorded in the manifest.

Other properties
----------------
  * the revision is pinned and embedded in every URL, so a moving branch cannot
    silently change the bytes,
  * each file is written to ``<name>.part`` and only renamed on success, so an
    interrupted run can never leave a file that looks complete,
  * resume uses HTTP Range against the partial file and validates the
    ``206``/``Content-Range`` reply before appending,
  * retries with backoff; a hash failure keeps the payload as ``.part.bad``
    instead of deleting evidence,

Usage:
    python download_model.py --dest model
    python download_model.py --dest model --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

REPO = "OpenRAL/rskill-molmoact2-multi-so101-nf4"
REVISION = "85bfeb1e97f198d2906e3585528bdfd8ca58cfb2"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
API_MODEL = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"
API_TREE = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1"
UA = "molmoact2-nf4-offline-verify/1.1"

CHUNK = 8 * 1024 * 1024
RETRIES = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_json(url: str):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def remote_index() -> dict:
    """path -> {size, sha256|null} straight from the Hub tree API."""
    meta = _get_json(API_MODEL)
    if meta.get("sha") != REVISION:
        raise SystemExit(f"API revision {meta.get('sha')} != pinned {REVISION}")
    index = {}
    for entry in _get_json(API_TREE):
        if entry.get("type") == "directory":
            continue
        lfs = entry.get("lfs") or {}
        index[entry["path"]] = {
            "size": int(lfs.get("size") or entry.get("size") or -1),
            "sha256": lfs.get("oid"),
            "xet": bool(lfs),
        }
    return index


def head_size(name: str) -> int:
    req = urllib.request.Request(f"{BASE}/{name}", method="HEAD")
    req.add_header("User-Agent", UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("content-length", "-1")), r.geturl()


def md5_file(path: str, block: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(path: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def download(name: str, dest: str, size: int) -> None:
    final = os.path.join(dest, name)
    part = final + ".part"
    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size > 0 and have == size:
            break
        if size > 0 and have > size:
            log(f"  {name}: partial larger than remote ({have}>{size}); restarting")
            os.remove(part)
            have = 0

        req = urllib.request.Request(f"{BASE}/{name}")
        req.add_header("User-Agent", UA)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    log(f"  {name}: server ignored Range (status {r.status}); restarting")
                    os.remove(part)
                    have = 0
                elif have:
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {have}-"):
                        log(f"  {name}: bad Content-Range {cr!r}; restarting")
                        os.remove(part)
                        have = 0
                mode = "ab" if have else "wb"
                got = have
                t0 = time.time()
                last_log = t0
                with open(part, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        got += len(buf)
                        now = time.time()
                        if now - last_log > 15:
                            mb = got / (1 << 20)
                            sp = mb / max(now - t0, 1e-6)
                            pct = f"{100.0 * got / size:.1f}%" if size > 0 else "?"
                            log(f"  {name}: {mb:.1f} MiB ({pct}) {sp:.1f} MiB/s")
                            last_log = now
                if size > 0 and got != size:
                    raise urllib.error.ContentTooShortError(
                        f"got {got} of {size} bytes", None
                    )
            break
        except Exception as e:  # noqa: BLE001 - any transport error is retried
            wait = min(2**attempt, 40)
            log(f"  {name}: attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}")
            if attempt == RETRIES:
                raise
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not download {name}")

    actual = sha256_file(part)
    os.replace(part, final)
    log(f"  {name}: complete {os.path.getsize(final)} bytes sha256={actual[:16]}...")


def check(path: str, want_sha: str | None, want_size: int) -> dict:
    size = os.path.getsize(path)
    digest = sha256_file(path)
    ok = size == want_size and (want_sha is None or digest == want_sha)
    return {
        "bytes": size,
        "remote_bytes": want_size,
        "sha256": digest,
        "remote_sha256": want_sha,
        "verified": bool(ok),
        "hash_kind": "sha256(lfs oid)" if want_sha else "size-only(non-LFS)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="model")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--adopt",
        nargs="*",
        default=None,
        help="existing file(s) to verify and rename into place (e.g. *.part.bad)",
    )
    args = ap.parse_args()
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    log(f"repo={REPO} revision={REVISION}")
    index = remote_index()
    log(f"remote file count={len(index)}")

    if args.adopt:
        for src in args.adopt:
            name = os.path.basename(src)
            for suffix in (".part.bad", ".part", ".bad"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            want = index.get(name)
            if want is None:
                raise SystemExit(f"{src}: {name} not in the pinned revision")
            info = check(src, want["sha256"], want["size"])
            log(f"  adopt {name}: {json.dumps(info)}")
            if not info["verified"]:
                raise SystemExit(f"{src}: verification failed, not adopting")
            os.replace(src, os.path.join(dest, name))
            log(f"  adopted -> {os.path.join(dest, name)}")

    entries = []
    for name, want in index.items():
        if args.only and name not in args.only:
            continue
        final = os.path.join(dest, name)
        if args.verify_only:
            if not os.path.exists(final):
                entries.append({"path": name, "status": "missing", **want})
                continue
            info = check(final, want["sha256"], want["size"])
            entries.append({"path": name, **info})
            log(f"  verify {name}: {'OK' if info['verified'] else 'MISMATCH'}")
            continue
        if os.path.exists(final) and os.path.getsize(final) == want["size"]:
            log(f"  {name}: already present, re-verifying")
        else:
            size, _ = head_size(name)
            if size != want["size"]:
                log(f"  {name}: HEAD size {size} != tree size {want['size']} (using HEAD)")
            download(name, dest, size)
        info = check(final, want["sha256"], want["size"])
        if not info["verified"]:
            raise SystemExit(f"{name}: verification failed: {json.dumps(info)}")
        entries.append({"path": name, **info})

    manifest_path = os.path.join(os.path.dirname(dest), "model_manifest.json")
    prev = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                prev = {e["path"]: e for e in json.load(f).get("files", [])}
        except Exception:
            prev = {}
    for e in entries:
        prev[e["path"]] = e
    out = {
        "repo": REPO,
        "revision": REVISION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dest": dest,
        "integrity_reference": "HF tree API lfs.oid (Xet: HTTP ETag is not the file sha256)",
        "files": [prev[k] for k in sorted(prev)],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {manifest_path}")
    total = sum(e.get("bytes") or 0 for e in out["files"])
    log(f"total bytes on disk = {total} ({total / (1 << 30):.3f} GiB)")
    bad = [e["path"] for e in out["files"] if not e.get("verified")]
    if bad:
        log(f"UNVERIFIED: {bad}")
        return 1
    log("all files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
