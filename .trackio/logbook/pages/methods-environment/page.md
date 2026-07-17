# Methods & environment

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_sm_01", "created_at": "2026-07-17T19:02:00+00:00", "title": "Setup"}
-->
**Finite MDP.** States S, actions A, transitions P, stochastic policy π, goal states (absorbing),
discount γ. Sparse per-step penalty r=−1{not at goal}.

- P^π(s'|s)=Σ_a π(a|s)P(s'|s,a). Goal-absorbing: P_full[goal]=e_goal.
- (A) Bellman V=(I−γP_full)^{-1}r.
- (B) Survival: G=substochastic transition matrix restricted to non-goal states;
  S(t|s)=(G^t𝟙)_s; V=−(I−γG)^{-1}𝟙.

**Environment.** Python 3.12, numpy/scipy, pytest. CPU only. 5/5 tests pass. Official code
`Simple-Robotics/hierarchical-survival-value-learning` (JAX, `survival.py`) implements the identity;
core math is clean-room numpy.

**Scope.** C2 (Prop 4.1) exact identity. C1 conceptual; C3 offline-GCRL benchmarks (training) out of scope.
