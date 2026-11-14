"""Async load test for ``POST /v1/rca`` and ``POST /v1/alarms/validate`` (asyncio + httpx).

Payloads come from the DB (label-free alarm columns, random 15-minute windows with alarms), so the
server sees realistic requests. Run against a started API::

    python -m prahari.monitoring.loadtest --base-url http://127.0.0.1:8000 --clients 1,8,32

Writes a markdown table to ``docs/loadtest.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from prahari.config import ROOT, get_settings
from prahari.db import get_engine
from prahari.pipeline.store import load_alarms

ENDPOINTS = ("/v1/rca", "/v1/alarms/validate")


def build_payloads(n: int = 200, seed: int = 42, batch: int = 20) -> dict[str, list[dict]]:
    """Request bodies: RCA windows around alarm bursts and validate batches of real alarms."""
    settings = get_settings()
    engine = get_engine(settings)
    end = pd.Timestamp(pd.read_sql("SELECT MAX(ts) AS m FROM alarms", engine)["m"][0])
    al = load_alarms(engine, end - pd.Timedelta(days=30), end)
    rng = np.random.default_rng(seed)
    al["ts"] = pd.to_datetime(al["ts"])
    centres = al["ts"].iloc[rng.integers(0, len(al), n)]
    rca = [
        {
            "start": (c - pd.Timedelta(minutes=5)).isoformat(),
            "end": (c + pd.Timedelta(minutes=10)).isoformat(),
        }
        for c in centres
    ]
    val = []
    for c in centres:
        sub = al[
            (al["ts"] >= c - pd.Timedelta(minutes=5)) & (al["ts"] < c + pd.Timedelta(minutes=10))
        ]
        sub = sub.head(batch).copy()
        sub["ts"] = sub["ts"].map(lambda t: t.isoformat())
        if len(sub):
            val.append({"alarms": sub.to_dict("records")})
    return {"/v1/rca": rca, "/v1/alarms/validate": val}


async def _client(http, path, bodies, stop_at, lat, errs, rng):
    while time.perf_counter() < stop_at:
        body = bodies[int(rng.integers(len(bodies)))]
        t = time.perf_counter()
        try:
            r = await http.post(path, json=body)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        dt = (time.perf_counter() - t) * 1000
        if ok:
            lat.append(dt)
        else:
            errs.append(dt)


async def run_level(base_url: str, path: str, bodies: list[dict], clients: int, duration: float):
    """Run ``clients`` concurrent closed-loop clients for ``duration`` seconds."""
    lat: list[float] = []
    errs: list[float] = []
    limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
    async with httpx.AsyncClient(base_url=base_url, timeout=60, limits=limits) as http:
        warm = await http.post(path, json=bodies[0])
        if warm.status_code != 200:
            raise RuntimeError(f"{path} returned {warm.status_code}: {warm.text[:200]}")
        t0 = time.perf_counter()
        stop = t0 + duration
        await asyncio.gather(
            *[
                _client(http, path, bodies, stop, lat, errs, np.random.default_rng(i))
                for i in range(clients)
            ]
        )
        wall = time.perf_counter() - t0
    p = np.percentile(lat, [50, 95, 99]) if lat else [float("nan")] * 3
    return {
        "endpoint": path,
        "clients": clients,
        "requests": len(lat),
        "errors": len(errs),
        "rps": len(lat) / wall,
        "p50": p[0],
        "p95": p[1],
        "p99": p[2],
    }


def render(rows: list[dict], base_url: str, duration: float) -> str:
    """Markdown report."""
    L = [
        "# Load test",
        "",
        f"Target {base_url}, closed-loop clients, {duration:.0f} s per level, "
        f"client and server on one {platform.machine()} machine. Latency in ms, successful "
        "requests only; errors counted separately.",
        "",
        "| endpoint | clients | requests | errors | req/s | p50 | p95 | p99 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['endpoint']} | {r['clients']} | {r['requests']} | {r['errors']} | "
            f"{r['rps']:.1f} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} |"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--clients", default="1,8,32", help="comma separated concurrency levels")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per level")
    ap.add_argument("--out", default=str(ROOT / "docs" / "loadtest.md"))
    a = ap.parse_args()
    payloads = build_payloads()
    rows = []
    for path in ENDPOINTS:
        for c in [int(x) for x in a.clients.split(",")]:
            rows.append(asyncio.run(run_level(a.base_url, path, payloads[path], c, a.duration)))
            print(rows[-1])
    text = render(rows, a.base_url, a.duration)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
