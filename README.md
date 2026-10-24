# vLLM on a consumer GPU: field notes

What actually happens when you serve an LLM for concurrent agent workloads on an
**8GB RTX 4060 Ti** under **WSL2**: the bring-up blockers, a benchmark bug that
inflated my own numbers, where the real KV-cache ceiling sits, and which tuning
levers helped versus hurt. Everything here was measured, and several headline
numbers were corrected after I found bugs in my own methodology. Those
corrections are part of the story.

**Stack:** vLLM 0.28.0 · CUDA 13.3 (pip `nvidia-cuda-nvcc-cu13`) · WSL2 Ubuntu 24.04 ·
`Qwen/Qwen3-4B-AWQ` · `FLASH_ATTN` backend · OpenAI-compatible API with
structured tool calling (`--enable-auto-tool-choice --tool-call-parser hermes`)

**Tool:** [`bench.py`](bench.py) is a concurrency sweep that reports TTFT and
per-stream decode speed separately at each concurrency level, plus aggregate
throughput and scaling against concurrency 1.

---

## TL;DR

| Finding | Result |
|---|---|
| Real concurrency on 8GB? | Yes: **18.2x aggregate scaling at 32 concurrent** (short completions) |
| My first number | 23.2x. **Wrong**, inflated by prefix caching in my own benchmark |
| Real KV-cache ceiling | ~**16 concurrent** at ~2,000-token completions; past it, preemption with full recompute |
| `--watermark 0.05` | ✅ Turned unrecoverable thrashing into clean queue-and-drain |
| fp8 KV cache | ❌ Blocked: forces FlashInfer, which hits a CCCL/CUDA version-skew bug |
| Draft-model speculative decoding | ❌ Net loss: 40% acceptance, **−18% decode speed at 8 concurrent** |
| `localhost` vs `127.0.0.1` | IPv6-first resolution added ~2s to **every** request (TTFT 2.7s → 0.69s) |

---

## 1. Bring-up: five blockers, in the order hit

1. **`--gpu-memory-utilization 0.90` failed.** The same card drives two monitors
   (~1.1GB reserved). Fix: `0.80`.
2. **Pinned memory is disabled by default on WSL2.** vLLM gates it off
   regardless of kernel version. Fix: `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.
3. **`torch.compile` crashed on the Windows-inherited PATH.** WSL2 imports
   about 86 Windows PATH entries. `torch/_dynamo/debug_utils.py` shells out to
   `nvcc --version` and catches only `FileNotFoundError`, but `execvp()` across
   that PATH returned `EACCES` (`PermissionError`). Fix:
   `appendWindowsPath = false` in `/etc/wsl.conf`.
4. **FlashInfer's JIT then couldn't find `nvcc`.** Fix: point `CUDA_HOME` at the
   pip-installed `nvidia/cu13` package directory.
5. **FlashInfer's vendored CCCL headers reject CUDA 13.3.** This is an upstream
   version-skew issue in vLLM 0.28.0's dependency chain, not specific to WSL.
   Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (falls back to vLLM's native sampler).

**Client side:** `localhost` resolved to IPv6 first and stalled about 2s before
falling back to the IPv4 address WSL2 forwards. Switching to `127.0.0.1` cut
TTFT from **2.7s to 0.69s**.

## 2. The benchmark bug that inflated my numbers

My first concurrency sweep looked great: TTFT stayed nearly flat as concurrency
rose. Then two identical runs reported `ttft_p50` of **1.877s** and **0.051s**
with nothing changed.

**Cause:** every request sent a byte-identical prompt. vLLM's automatic prefix
caching (on by default) cached those KV blocks after the first request, so every
later request was a free cache replay, not a real prefill.

**Fix:** prepend a random nonce to the **front** of each prompt. Prefix matching
starts at token 0, so this guarantees a full cache miss (see `_prompt()` in
`bench.py`).

**Corrected sweep** (`--levels 1,2,4,8,16,32 --reps 3 --tokens 256`, `max_num_seqs 32`):

| concurrency | ttft_p50 | ttft_p95 | tok/s per stream | aggregate tok/s | scaling |
|---|---|---|---|---|---|
| 1  | 0.038s | 0.039s | 73.65 | 73.78   | 1.0 |
| 2  | 0.064s | 0.096s | 69.25 | 137.34  | 1.86 |
| 4  | 0.118s | 0.130s | 70.47 | 270.79  | 3.67 |
| 8  | 0.210s | 0.215s | 67.64 | 508.15  | 6.89 |
| 16 | 0.399s | 0.407s | 61.06 | 874.26  | 11.85 |
| 32 | 0.696s | 0.967s | 48.14 | 1340.93 | **18.17** |

Real concurrency, but TTFT grows about 18x from 1 to 32 concurrent. Prefill is a
real cost. The original 23.2x scaling figure was an artifact.

## 3. Finding the real KV-cache ceiling via `/metrics`

`--max-num-seqs 32` is a config cap, not a capacity guarantee. vLLM's
`/metrics` endpoint reports the real budget directly:

- `kv_cache_size_tokens = 21952` (1,372 blocks × 16 tokens)
- `kv_cache_max_concurrency = 2.68` at the full `max_model_len` of 8,192

Stress test with ~2,000-token completions (~2,090 tokens per request), polling
`/metrics` every second:

| concurrency | tokens needed | % of KV budget | behavior |
|---|---|---|---|
| 8  | ~16,700 | 76%  | Clean: cache peaked ~79%, zero waiting, zero preemptions |
| 16 | ~33,400 | 152% | **Ceiling:** cache ~98%, running fell 16 → 10, preemptions 0 → 6 |
| 32 | ~66,900 | 305% | Thrashing: running oscillated 0–16, preemptions reached **43** |

At 32 concurrent, aggregate scaling collapsed to **2.57x**, almost exactly vLLM's
static prediction of 2.68x. The same server and config managed 18.17x with
short completions; only completion length changed.

**Why it hurts:** vLLM handles cache exhaustion with **preemption plus full
recompute**. An evicted sequence's KV cache is discarded and re-prefilled when
it's re-admitted. That's wasted GPU work, not just waiting, and it showed up as
per-stream decode dropping from 55.7 to 25.0 tok/s.

**Why it matters for agents:** short tool-calling turns never hit this. It
appears as soon as growing tool histories push context length up.

## 4. Tuning levers: what helped and what hurt

### ✅ `--watermark 0.05`: works
Same 2,000-token × 32-concurrent stress test, read from vLLM's periodic stats log:

```
Running=32  Waiting=0   cache=90.2%
Running=30  Waiting=2   cache=97.7%   <- admission backs off before 100%
Running=17  Waiting=15  cache=99.5%   <- peak backlog
Running=0   Waiting=0   cache=0.0%    <- fully drained, clean
```

It built a backlog but drained it completely, where the run without the
watermark thrashed and never recovered.

### ❌ fp8 KV cache: blocked on this stack
`--kv-cache-dtype fp8_e4m3` should roughly double KV capacity. In practice it
removes `FLASH_ATTN` from vLLM's candidate attention backends, so vLLM picks
`FLASHINFER`. FlashInfer then JIT-compiles prefill kernels against its vendored
CCCL headers and hits the same CUDA version-skew error as blocker #5. The
sampler workaround doesn't cover the attention path, so the engine crashes on
startup.

*Untested idea:* `VLLM_ATTENTION_BACKEND=TRITON_ATTN` might avoid this, since
Triton compiles through its own toolchain.

### ❌ Draft-model speculative decoding: net loss
`Qwen3-0.6B` drafting for `Qwen3-4B-AWQ`, 4 speculative tokens.

| concurrency | decode tok/s per stream (baseline) | with spec decode | ttft_p50 baseline | with spec decode |
|---|---|---|---|---|
| 4 | 70.47 | 64.62 (**−8.3%**) | 0.118s | 0.407s |
| 8 | 67.64 | 55.53 (**−17.9%**) | 0.210s | 0.258s |

- **Acceptance rate was 40.2%** and fell quickly by draft position:
  65.6% → 43.9% → 31.1% → 20.2%.
- At concurrency 1 the GPU has idle headroom to absorb wasted draft passes. At
  4 and above, the target's own batching already uses that headroom, so draft
  passes compete for compute.
- **It also costs KV capacity:** the draft weights shrank usable context so far
  that `max_model_len` had to drop to 2,048 (KV budget 2,368 tokens versus
  21,952 without the draft).
- Concurrency 1 isn't reported: first-hit CUDA-graph capture made it unreliable.

Note that native MTP heads don't exist for Qwen3, so draft-model speculation
is the only option for this model family.

## 5. Baseline config I'd recommend for this card

```bash
# environment (persist in ~/.bashrc)
export CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3-4B-AWQ \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --watermark 0.05 \
  --enable-auto-tool-choice --tool-call-parser hermes
# KV cache dtype: auto (fp16). No fp8, no speculative decoding (see above).
```

## Running the benchmark

```bash
pip install -r requirements.txt
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen3-4B-AWQ
export LLM_LABEL=4060ti-cuda

python bench.py --levels 1,2,4,8,16,32 --reps 3 --tokens 256   # concurrency scaling
python bench.py --levels 4,8,16,32 --tokens 2000               # KV-cache stress
```

Results are appended to `bench.jsonl`.

## Lessons

- **Distrust results that look too good.** The flat-TTFT result was the
  benchmark being wrong, not the hardware being great.
- **Read the server's own telemetry.** `/metrics` (`kv_cache_usage_perc`,
  `num_preemptions_total`, `num_requests_waiting_by_reason`) predicted the
  collapse before I measured it.
- **Tuning levers interact.** fp8 changes the attention backend, which changes
  the compile path, which hits a dependency bug. Speculative decoding trades
  away the KV capacity that was already the bottleneck. Measure, don't assume.

## Open questions

- Does `TRITON_ATTN` unblock fp8 KV cache on this stack?
- Exact preemption count with the watermark on, compared with 43 without it.
- Would `num_speculative_tokens=2`, or an EAGLE-style head, change the
  speculative-decoding result at low concurrency?

---

*Measured August 2026 on a single RTX 4060 Ti 8GB. Numbers are specific to this
hardware, driver, and vLLM version.*
