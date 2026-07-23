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

**Frozen baseline.** Experiment `b2cb1434-e685-476b-9597-d87cc959d177`, commit
`bd368be36cec3dbebfc1187fca1493445d4494b9`, run
`bed2a8ca-7821-4d05-977b-06bf590db98b`. The inherited command is
`bash repro/orx/run_evaluation.sh`. The resolved baseline packages were `numpy==1.26.4`,
`scipy==1.12.0`, and `pytest==8.4.2`.

**Route-audit source.** `https://ar5iv.labs.arxiv.org/html/2604.17551`, retrieved
2026-07-23, SHA-256
`69f0db8b7819cac262f4c91e945fc7a6d4b011e7d603fbfe3f0844fa5acecec0`.
Every route child pins the judge contract and relevant released source/configuration hashes.

**Execution limitation.** Four C1 local launches died before project evaluation while the fixed
runner installed its isolated environment. The OpenResearch logs contain at most the repository
clone line, and no route `EVAL.md` or text artifacts were produced. The other route children were
not launched after this repeated environmental failure. The last filesystem check showed 851 MiB
free.

**Scope.** C2 is exact and executed. C1 and C3-C6 entries elsewhere in this logbook are
source/protocol route audits only. No empirical training result, theorem proof, or unjudged score
is claimed.
