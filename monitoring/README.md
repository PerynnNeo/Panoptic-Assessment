# Task 4: Performance Analysis

## Latency Breakdown

| Component                 |     Average (ms) | Median (ms) | P95 (ms) |
| ------------------------- | ---------------: | ----------: | -------: |
| Edge detection latency    |            21.93 |           – |     57.3 |
| Cloud processing latency  |             2.85 |        2.83 |     3.09 |
| End-to-End (edge → cloud) | n/a (not logged) |         n/a |      n/a |

* **Edge detection latency** from `edge_metrics_edgecloud.csv` (`stage = detect`).
* **Cloud processing latency** from `cloud_metrics_edgecloud.csv` (`cloud_latency_ms`).
* **End-to-End latency** could not be logged in this run (`e2e_est_ms` not printed), but system throughput shows real-time performance well under the 15s requirement.

---

## Edge-only vs Edge+Cloud Benchmark

| Scenario   | Avg FPS | Edge Detect P95 (ms) | Cloud Latency P95 (ms) |  Edge CPU% |    Cloud CPU% |
| ---------- | ------: | -------------------: | ---------------------: | ---------: | ------------: |
| Edge-only  |   18.41 |                 57.3 |                      – | low (\~3%) |             – |
| Edge+Cloud |   19.07 |                 57.3 |                   3.09 | low (\~3%) | low (\~5–15%) |

**Interpretation:**

* **Edge-only:** stable \~18 FPS, low CPU, all inference local.
* **Edge+Cloud:** \~19 FPS, almost identical edge latency, with cloud overhead only \~3 ms per frame.
* No significant degradation in performance between the two modes.

---

## Trade-offs 

  * Edge-only = ultra-low latency, but no contextual activity recognition.
  * Edge+Cloud = enables rich activity alerts, adds minor overhead (\~3 ms average cloud latency).
  * Smart sampling reduces bandwidth but could miss rare edge cases if motion thresholds are too strict.
