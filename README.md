# Repro — SVL: Goal-Conditioned RL as Survival Learning (qIOcJSCGn2)

Clean-room reproduction of *SVL: Goal-Conditioned Reinforcement Learning as Survival Learning*
(Tiofack, Schramm, Le Hellard, Carpentier; arXiv [2604.17551](https://arxiv.org/abs/2604.17551)),
for the [ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `qIOcJSCGn2`.

This repository now contains two distinct reproductions: an applied released-code experiment for
the Claim 1 time-to-goal distribution, and the original clean-room exact identity test for Claim 2.

## Results (all local CPU)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C1** time-to-goal is modeled as a probability distribution | **VERIFIED (applied, bounded)** | released PCS critic/likelihood on real OGBench PointMaze-large; held-out censored NLL **2.398 ± 0.041** vs **3.158 ± 0.006** for a goal-conditioned one-hazard geometric baseline (3 seeds); independently normalized probability mass to `1 ± 1.4e-15` with zero survival-monotonicity violations. |
| **C2** closed-form value = discounted sum of survival probabilities | **VERIFIED** | three independent methods (Bellman solve, survival matrix-geometric, value iteration) agree to **1.2e-12** across 36 random MDPs; both negative controls (state-dependent reward, non-absorbing goal) correctly break the identity. |

7/7 pytest tests pass. Claim 1 uses the full 1,000,000-valid-transition OGBench training
pool and a disjoint 100,000-valid-transition validation pool, with reduced model/training scale
fully disclosed in [the Claim 1 audit](docs/CLAIM1_OGBENCH_DISTRIBUTION_AUDIT.md). Claim 3
offline-GCRL policy performance remains out of scope and is not claimed.

## Reproduce
```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

# Claim 2 clean-room identity.
uv pip install numpy==1.26.4 scipy==1.12.0 pytest
python repro/src/run_svl.py

# Claim 1 released-code experiment; see the audit for exact dependency pins,
# data-download commands, thread limits, and full invocation.
python repro/src/claim1_ogbench_distribution.py --help
python repro/src/audit_claim1_outputs.py --results-dir outputs/claim1_ogbench

python -m pytest -q repro/tests
```

## Verification method (three independent routes + negative controls)
- **(A) Bellman:** `V = (I−γP_full)^{-1}r` on the goal-absorbing chain.
- **(B) Survival:** `V = −(I−γG)^{-1}𝟙` on the killed (non-goal) substochastic chain.
- **(C) Value iteration** to convergence.
- **Negative controls:** state-dependent penalty and non-absorbing goal both break the identity (its stated scope).

## Scope & honest disclosures
- C1 is direct distribution-learning evidence on a real navigation dataset, not a reproduction of
  the paper's policy-success table. The critic, horizon, bins, basis, batch, and update count are reduced.
- C2 (Proposition 4.1) is the exact identity and remains verified without modification.
- C3 offline-GCRL policy benchmarks require actor training/evaluation and are not claimed here.
- Official code `Simple-Robotics/hierarchical-survival-value-learning` (JAX, `survival.py`) implements the
  identity — cross-check reference; core math is clean-room numpy.

Logbook: https://huggingface.co/spaces/DineshAI/qIOcJSCGn2
