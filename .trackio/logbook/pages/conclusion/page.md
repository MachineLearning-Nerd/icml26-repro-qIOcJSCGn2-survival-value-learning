# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_sxec_01", "created_at": "2026-07-17T19:04:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T19:04:05+00:00"}
-->
**C2 reproduced.** *SVL* (Tiofack et al.; `qIOcJSCGn2`) Proposition 4.1 — the goal-conditioned value
as a discounted sum of survival probabilities — is verified by three independent numpy computations
agreeing to 1.2e-12, with both scope negative controls (state-dependent reward, non-absorbing goal)
correctly breaking it.

- **C2 (value = discounted survival sum) — VERIFIED.** Bellman / survival-matrix / value-iteration agree to machine precision.

5/5 pytest tests pass. CPU only, exact.

## Scope & cost
| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | C2 Proposition 4.1 exact identity | + C3 offline-GCRL benchmarks (training) |
| Hardware | 4 vCPU CPU | GPU |
| Time | < 1 min | — |
| Cost | 0 | — |
| Outcome | C2 VERIFIED | — |

## Honest deviations
- C2 (Prop 4.1) is the exact identity, verified. C1 is the conceptual reframe (realized by C2);
  C3 (offline-GCRL benchmarks) needs training and is out of scope.
- Official `Simple-Robotics/hierarchical-survival-value-learning` (JAX) cross-checks the formula; core math is clean-room numpy.
