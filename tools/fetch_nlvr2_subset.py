#!/usr/bin/env python
"""Download only the NLVR2 images a CIRR split needs, from the official zip.

The NLVR2 host throttles each connection (~16 KB/s from some networks) but the
aggregate scales with the number of connections, and CIRR validation needs only
2,297 of the 8,102 images in dev_img.zip. This reads the zip's central
directory over HTTP range requests, then fetches and inflates just the
required members with many parallel connections, checking size and CRC32.

The source is the same official download URL the dataset access form grants
(http://clic.nlp.cornell.edu/resources/NLVR2/); nothing is taken from mirrors.

Example:
    python tools/fetch_nlvr2_subset.py --split-json /workspace/data/CIRR/cirr/image_splits/split.rc2.val.json \
        --out-dir /workspace/data/CIRR/cirr/img_raw --zip-name dev_img.zip
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

BASE = "https://clic.nlp.cornell.edu/resources/NLVR2/"


def http_range(url: str, start: int, end: int, timeout: int = 60, retries: int = 6) -> bytes:
    """Bytes [start, end] inclusive, with retries."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            if len(data) == end - start + 1:
                return data
            last = f"short read {len(data)} of {end - start + 1}"
        except Exception as error:  # network hiccup: back off and retry
            last = repr(error)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def total_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as response:
        return int(response.headers["Content-Length"])


def read_central_directory(url: str, size: int, threads: int) -> dict[str, tuple]:
    """member name -> (method, compressed size, uncompressed size, crc32, local header offset)."""
    tail_start = max(0, size - 65_600)
    tail = http_range(url, tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("end-of-central-directory record not found")
    entries, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
    if cd_offset == 0xFFFFFFFF:
        raise RuntimeError("zip64 archives are not supported")
    chunk = 65_536
    spans = [(o, min(o + chunk, cd_offset + cd_size) - 1) for o in range(cd_offset, cd_offset + cd_size, chunk)]
    with cf.ThreadPoolExecutor(threads) as pool:
        directory = b"".join(pool.map(lambda s: http_range(url, *s), spans))

    members: dict[str, tuple] = {}
    pos = 0
    while pos < len(directory) and directory[pos : pos + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", directory[pos + 10 : pos + 12])
        crc, csize, usize, name_len, extra_len, comment_len = struct.unpack("<IIIHHH", directory[pos + 16 : pos + 34])
        local_offset, = struct.unpack("<I", directory[pos + 42 : pos + 46])
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8")
        members[name] = (method, csize, usize, crc, local_offset)
        pos += 46 + name_len + extra_len + comment_len
    if len(members) != entries:
        raise RuntimeError(f"parsed {len(members)} entries, expected {entries}")
    return members


def fetch_member(url: str, name: str, info: tuple, out_path: Path) -> None:
    method, csize, usize, crc, local_offset = info
    slack = 512  # local header + name + extra (the local extra field can differ from the central one)
    data = http_range(url, local_offset, local_offset + 30 + len(name.encode()) + slack + csize - 1)
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    body = data[30 + name_len + extra_len : 30 + name_len + extra_len + csize]
    if len(body) != csize:
        raise RuntimeError(f"{name}: truncated body")
    raw = body if method == 0 else zlib.decompress(body, -15)
    if len(raw) != usize or (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
        raise RuntimeError(f"{name}: size/CRC mismatch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True, help="CIRR image_splits/split.rc2.<split>.json")
    parser.add_argument("--out-dir", required=True, help="the img_raw directory (files land in <out-dir>/dev/...)")
    parser.add_argument("--zip-name", default="dev_img.zip", help="dev_img.zip | test1_img.zip | train_img.zip")
    parser.add_argument("--threads", type=int, default=160)
    args = parser.parse_args()

    url = BASE + args.zip_name
    wanted = {rel.lstrip("./") for rel in json.load(open(args.split_json, encoding="utf-8")).values()}
    out_dir = Path(args.out_dir)
    print(f"{len(wanted)} images wanted; reading zip directory...", flush=True)
    size = total_size(url)
    members = read_central_directory(url, size, min(args.threads, 32))
    missing = sorted(w for w in wanted if w not in members)
    if missing:
        raise SystemExit(f"{len(missing)} wanted files are not in {args.zip_name}, e.g. {missing[:3]}")

    todo = [w for w in sorted(wanted) if not (out_dir / w).exists()]
    total_bytes = sum(members[w][1] for w in todo)
    print(f"{len(todo)} still to fetch, {total_bytes / 1e6:.0f} MB compressed", flush=True)
    done = 0
    start = time.time()
    with cf.ThreadPoolExecutor(args.threads) as pool:
        futures = {pool.submit(fetch_member, url, w, members[w], out_dir / w): w for w in todo}
        for future in cf.as_completed(futures):
            future.result()  # raises on failure
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} files, {time.time() - start:.0f}s", flush=True)
    present = sum(1 for w in wanted if (out_dir / w).exists())
    print(f"NLVR2_SUBSET_DONE {present}/{len(wanted)}", flush=True)


if __name__ == "__main__":
    main()
