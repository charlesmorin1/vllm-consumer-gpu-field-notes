"""Concurrency sweep for any OpenAI-compatible LLM server (vLLM, Ollama, ...).

Reports TTFT and decode speed SEPARATELY at each concurrency level. That
separation is the point:

  - TTFT rising steeply with concurrency  -> prefill-bound. This is what agent
    loops trigger, with their large system prompts and growing tool histories.

  - Aggregate t/s plateauing while TTFT climbs -> the backend is saturated.
    Everything past that point is queueing, not throughput. That plateau is the
    correct setting for max_num_seqs.

Configure via environment:
  LLM_BASE_URL   default http://127.0.0.1:8000/v1  (use 127.0.0.1, not localhost -
                 see README: IPv6-first resolution added ~2s to every request)
  LLM_MODEL      default Qwen/Qwen3-4B-AWQ
  LLM_API_KEY    default not-needed
  LLM_LABEL      free-form tag recorded with results, e.g. 4060ti-cuda

Usage:
  python bench.py
  python bench.py --levels 1,2,4,8,16,32 --reps 3 --tokens 256
  python bench.py --levels 4,8,16,32 --tokens 2000   # KV-cache stress
"""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
LABEL = os.getenv("LLM_LABEL", "unlabelled")

# The comparable-across-runs part of this string is fine; using it VERBATIM on
# every call is what broke prefill measurement (see _prompt() below) - each
# call gets a unique nonce prepended so this text is never sent twice.
PROMPT = (
    "You are a systems engineer. Explain, in careful detail, how memory "
    "bandwidth limits token generation speed for a dense transformer during "
    "decode, and why a mixture-of-experts model with a small number of active "
    "parameters behaves differently on the same hardware. Cover the arithmetic "
    "intensity of the decode step, why batching changes the picture, and what "
    "that implies for choosing between a high-bandwidth discrete GPU and a "
    "large-capacity unified-memory system."
)


def _prompt() -> str:
    # A random token sequence goes FIRST, not last. vLLM (and Ollama) automatic
    # prefix caching matches the longest common prefix starting at token 0, so
    # a shared PROMPT reused verbatim across every call becomes a 100% cache
    # hit after the very first request this server ever sees - every later
    # call, at every concurrency level, skips real prefill entirely and reports
    # a near-zero TTFT that has nothing to do with the hardware. Putting the
    # nonce first guarantees zero shared prefix with any prior call, forcing
    # the real prefill this benchmark exists to measure. Confirmed: two
    # otherwise-identical sweeps (levels 1..32) reported ttft_p50 1.877s vs
    # 0.051s at concurrency=32 - a caching artifact, not hardware variance.
    return "[req %s] " % uuid.uuid4().hex + PROMPT


def _one(client: OpenAI, tokens: int) -> dict:
    t0 = time.perf_counter()
    t_first = None
    chars = 0
    completion_tokens = None
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _prompt()}],
            max_tokens=tokens, temperature=0.7, stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices or chunk.choices[0].delta is None:
                continue
            delta = chunk.choices[0].delta
            # Thinking models stream a reasoning block first. Ollama and vLLM
            # expose it under different attribute names, and it is absent
            # entirely on non-thinking models - hence getattr. Missing this
            # makes TTFT unmeasurable on any thinking model.
            text = (getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or delta.content)
            if text:
                if t_first is None:
                    t_first = time.perf_counter()
                chars += len(text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                "total_s": round(time.perf_counter() - t0, 3)}

    t_end = time.perf_counter()
    estimated = completion_tokens is None
    if estimated:
        # Rough fallback when the backend omits usage. ~4 chars/token.
        # Flagged so it is never mistaken for a measured value.
        completion_tokens = max(1, chars // 4)
    ttft = round(t_first - t0, 4) if t_first is not None else None
    decode_s = round(t_end - t_first, 4) if t_first is not None else None
    return {"ok": True, "ttft_s": ttft, "decode_s": decode_s,
            "total_s": round(t_end - t0, 4), "tokens": completion_tokens,
            "decode_tps": (round(completion_tokens / decode_s, 2)
                           if decode_s else None),
            "estimated": estimated}


def sweep(client: OpenAI, levels: list[int], reps: int,
          tokens: int) -> list[dict]:
    # Warm up first. The opening call pays to load weights and capture CUDA
    # graphs, and that lands entirely in the concurrency-1 TTFT, making the
    # first row look catastrophically prefill-bound and every comparison
    # against it meaningless. Costs one call; saves a wrong verdict.
    print("  warmup ", end="", flush=True)
    w = _one(client, 16)
    print("(load %.1fs)" % (w.get("total_s") or 0.0))

    rows: list[dict] = []
    for n in levels:
        print("  concurrency %-3d " % n, end="", flush=True)
        calls: list[dict] = []
        batch_walls: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                batch = list(pool.map(lambda _i: _one(client, tokens), range(n)))
            wall = time.perf_counter() - t0
            batch_walls.append(wall)
            for b in batch:
                b["batch_wall_s"] = round(wall, 3)
            calls.extend(batch)
            print(".", end="", flush=True)

        good = [c for c in calls if c["ok"] and c.get("ttft_s") is not None]
        if not good:
            print("  all failed")
            rows.append({"concurrency": n, "ok": 0, "failed": len(calls),
                         "error": calls[0].get("error", "") if calls else ""})
            continue

        ttfts = sorted(c["ttft_s"] for c in good)
        tps = [c["decode_tps"] for c in good if c.get("decode_tps")]
        toks = sum(c["tokens"] or 0 for c in good)

        # Aggregate = all tokens produced / wall time actually elapsed.
        # One wall per rep, NOT one per call: every stream in a batch shares the
        # same elapsed seconds, so summing per-call walls counts that period
        # `n` times and understates throughput by exactly the concurrency factor.
        total_wall = sum(batch_walls) or 1.0
        agg = round(toks / total_wall, 2)

        # NOTE: do NOT compute efficiency as sum(total_s)/wall. total_s includes
        # time a request spent queued, so that formula counts waiting as work
        # and reports healthy parallelism for a server that is purely serial.
        # Real scaling is aggregate(n)/aggregate(1), computed in report().

        # p95 must never fall below p50. For small n the correct answer is the
        # worst observed sample, not an index that rounds down to the minimum.
        idx95 = min(len(ttfts) - 1, max(0, math.ceil(0.95 * len(ttfts)) - 1))

        row = {
            "concurrency": n,
            "ok": len(good),
            "failed": len(calls) - len(good),
            "ttft_p50": round(statistics.median(ttfts), 3),
            "ttft_p95": round(ttfts[idx95], 3),
            "decode_tps_per_stream": round(statistics.median(tps), 2) if tps else None,
            "aggregate_tps": agg,
            # scaling vs concurrency 1 is filled in by report()
            "tokens_estimated": any(c.get("estimated") for c in good),
        }
        rows.append(row)
        print("  ttft p50 %-7s  per-stream %-7s  aggregate %s" % (
            row["ttft_p50"], row["decode_tps_per_stream"], row["aggregate_tps"]))
    return rows


def report(rows: list[dict]) -> None:
    # Fill scaling before the table renders, not after.
    _ok = [r for r in rows if r.get("ttft_p50")]
    if _ok:
        _base = _ok[0]["aggregate_tps"] or 1.0
        for r in _ok:
            r["scaling"] = round((r["aggregate_tps"] or 0) / _base, 2)

    print("\n" + "=" * 78)
    print("CONCURRENCY SWEEP - %s / %s" % (LABEL, MODEL))
    print("=" * 78)
    print("%-6s %-6s %-10s %-10s %-14s %-12s %-8s" % (
        "conc", "ok", "ttft_p50", "ttft_p95", "tps/stream", "aggregate", "scaling"))
    print("-" * 78)
    for r in rows:
        if not r.get("ttft_p50"):
            print("%-6s %-6s  FAILED  %s" % (r["concurrency"], r.get("ok", 0),
                                             r.get("error", "")[:40]))
            continue
        print("%-6s %-6s %-10s %-10s %-14s %-12s %-8s" % (
            r["concurrency"], r["ok"], r["ttft_p50"], r["ttft_p95"],
            r["decode_tps_per_stream"], r["aggregate_tps"],
            r.get("scaling", "-")))

    ok = [r for r in rows if r.get("ttft_p50")]
    if len(ok) >= 2:
        print("\nReading:")
        first, last = ok[0], ok[-1]
        ttft_growth = last["ttft_p50"] / first["ttft_p50"] if first["ttft_p50"] else 0
        agg_growth = (last["aggregate_tps"] / first["aggregate_tps"]
                      if first["aggregate_tps"] else 0)
        print("  TTFT grew %.1fx from concurrency %d to %d" % (
            ttft_growth, first["concurrency"], last["concurrency"]))
        print("  Throughput scaling at concurrency %d: %.2fx (ideal %.1fx)"
              % (last["concurrency"], agg_growth, float(last["concurrency"])))

        # Find where aggregate stops improving - that is max_num_seqs.
        best = max(ok, key=lambda r: r["aggregate_tps"])
        if best["concurrency"] != ok[-1]["concurrency"]:
            print("  Aggregate peaked at concurrency %d (%s t/s), then flattened."
                  % (best["concurrency"], best["aggregate_tps"]))
        print("  Suggested max_num_seqs: %d" % best["concurrency"])

        if agg_growth < 1.3:
            print("  -> NOT actually running in parallel. Requests are queueing.")
            print("     Either the server caps concurrency (raise it) or one")
            print("     stream already saturates the hardware.")
        elif ttft_growth > 3.0:
            print("  -> Real concurrency, but TTFT degrades badly.")
            print("     Prefill is the bottleneck.")
        else:
            print("  -> Scales cleanly. Throughput-bound workload.")

    if any(r.get("tokens_estimated") for r in rows):
        print("\nNOTE: backend did not report token usage; completion counts")
        print("      are estimated at ~4 chars/token. Treat t/s as approximate.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--out", default=str(Path(__file__).parent / "bench.jsonl"))
    a = p.parse_args()

    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    print("backend %s | model %s | label %s" % (BASE_URL, MODEL, LABEL))
    print("levels %s | reps %d | max_tokens %d\n" % (levels, a.reps, a.tokens))

    client = OpenAI(base_url=BASE_URL,
                    api_key=os.getenv("LLM_API_KEY", "not-needed"))
    rows = sweep(client, levels, a.reps, a.tokens)
    report(rows)

    rec = {"label": LABEL, "model": MODEL, "levels": levels, "reps": a.reps,
           "max_tokens": a.tokens, "rows": rows,
           "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("\nAppended to %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
