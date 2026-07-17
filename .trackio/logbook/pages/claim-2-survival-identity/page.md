# Claim 2 — Value = discounted sum of survival probabilities

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_sc2_01", "created_at": "2026-07-17T19:01:00+00:00", "title": "Claim & method"}
-->
**Claim (verbatim):** "Provides a closed-form identity expressing the goal-conditioned value
function as a discounted sum of survival probabilities." (Proposition 4.1)

For sparse per-step penalty `r=−1{not at goal}` and terminate-on-success (goal absorbing):
**V^π(s,g) = −Σ_{t≥0} γ^t S^π(t|s,g)**, S^π(t|s,g)=Pr(T^π(s,g)>t).

Three independent computations must agree:
- **(A) Bellman:** V=(I−γP_full)^{-1}r on the goal-absorbing chain.
- **(B) Survival:** V=−(I−γG)^{-1}𝟙 on the killed (non-goal) substochastic chain.
- **(C) Value iteration** to convergence.

---
<!-- trackio-cell
{"type": "code", "id": "cell_sc2_02", "created_at": "2026-07-17T19:01:10+00:00", "title": "Identity verifier", "command": ["python", "repro/src/run_svl.py"], "exit_code": 0}
-->
````bash
$ python repro/src/run_svl.py
````
max|V_A−V_B| and |V_A−V_C| over 6 seeds × {8,16,30} × {0.9,0.99} = **1.2e-12** (< 1e-10). All three
methods agree to machine precision.

**=> C2 VERIFIED.** Evidence: `outputs/svl_summary.json`.
