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

**Campaign status on 2026-07-23.** The current live judge score remains **4/12**. No new
scientific evidence completed, and no candidate score is estimated.

- **C1:** live verdict `toy`; exact released-scale continuation is **BLOCKED** because it requires
  forbidden GPU/T4 work.
- **C3:** live verdict `toy`; an exact-scope finite-dimensional MLE proof route is CPU-reachable
  and worth a bounded preflight, but no proof has run.
- **C4-C5:** live verdicts `inconclusive`; exact four-seed giant-task comparisons are
  **BLOCKED** by GPU requirements, and the public visual HSVL path is incomplete.
- **C6:** live verdict `inconclusive`; **BLOCKED** by missing exact flat-SVL/paper-matched CRL
  source and GPU-scale training.

The four route-audit child commits are `51a69d5` (C1), `70afb86` (C3), `a5c8685`
(C4-C5), and `06bcd6e` (C6). Four C1 local runs failed before evaluation during dependency
installation; therefore no route-audit `EVAL.md`, artifact, independent scientific audit, or
executed route negative control is claimed.

## Scope & cost
| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | C2 Proposition 4.1 exact identity plus route feasibility record | Full six-claim scientific evidence |
| Hardware | 4 vCPU CPU | GPU |
| Time | < 1 min | — |
| Cost | 0 | — |
| Outcome | C2 VERIFIED; live score 4/12 unchanged | — |

## Honest deviations
- C2 (Prop 4.1) is the exact identity and remains verified. C1 and C3-C6 were audited against
  their exact claim wording; route feasibility is not scientific verification.
- Official `Simple-Robotics/hierarchical-survival-value-learning` (JAX) cross-checks the formula; core math is clean-room numpy.
- No HF job was launched, no GPU was used, and the judged HF Space was not modified or synced.
