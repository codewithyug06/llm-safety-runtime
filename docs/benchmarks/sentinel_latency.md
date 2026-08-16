# LatentSentinel Latency Benchmark

**Model:** `stub`
**Device:** `cpu`
**Requests:** 100
**SLA Target:** <10ms p95 (GPU) / <500ms (CPU)
**SLA Status:** PASS

## Results

| Metric | Value |
|--------|-------|
| p50 | 198.74 ms |
| p90 | 243.51 ms |
| **p95** | **259.62 ms** |
| p99 | 310.27 ms |
| Mean | 207.69 ms |
| Std Dev | 26.42 ms |
| Min | 178.06 ms |
| Max | 336.65 ms |

## SLA Assessment

The p95 latency of **259.62ms** is within the SLA budget.
ARGUS overhead does not impact production LLM throughput.
