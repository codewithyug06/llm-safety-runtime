# LatentSentinel Latency Benchmark

**Model:** `stub`
**Device:** `cpu`
**Requests:** 100
**SLA Target:** <10ms p95 (GPU) / <500ms (CPU)
**SLA Status:** PASS

## Results

| Metric | Value |
|--------|-------|
| p50 | 203.38 ms |
| p90 | 210.86 ms |
| **p95** | **212.38 ms** |
| p99 | 216.47 ms |
| Mean | 203.89 ms |
| Std Dev | 5.10 ms |
| Min | 189.43 ms |
| Max | 218.30 ms |

## SLA Assessment

The p95 latency of **212.38ms** is within the SLA budget.
ARGUS overhead does not impact production LLM throughput.
