# Repro — SVL: Goal-Conditioned RL as Survival Learning (qIOcJSCGn2)

Clean-room reproduction of *SVL: Goal-Conditioned Reinforcement Learning as Survival Learning*
(Tiofack, Schramm, Le Hellard, Carpentier; arXiv [2604.17551](https://arxiv.org/abs/2604.17551)),
for the [ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `qIOcJSCGn2`.

**Proposition 4.1 (exact identity).** For a goal-conditioned MDP with sparse per-step penalty
`r=−1{not at goal}` and terminate-on-success (goal absorbing):
`V^π(s,g) = −Σ_{t≥0} γ^t·Pr(T^π(s,g) > t)`, the discounted sum of survival probabilities.

## Results (all CPU, exact / machine-precision)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C2** closed-form value = discounted sum of survival probabilities | **VERIFIED** | three independent methods (Bellman solve, survival matrix-geometric, value iteration) agree to **1.2e-12** across 36 random MDPs; both negative controls (state-dependent reward, non-absorbing goal) correctly break the identity. |

5/5 pytest tests pass. (C1 "reframe GCRL as survival learning" is the conceptual framing directly realized by C2's identity; C3 offline-GCRL benchmarks are empirical and out of scope.)

## Reproduce
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_svl.py    # identity + 3rd method + 2 negative controls
python -m pytest repro/tests/
```

## Verification method (three independent routes + negative controls)
- **(A) Bellman:** `V = (I−γP_full)^{-1}r` on the goal-absorbing chain.
- **(B) Survival:** `V = −(I−γG)^{-1}𝟙` on the killed (non-goal) substochastic chain.
- **(C) Value iteration** to convergence.
- **Negative controls:** state-dependent penalty and non-absorbing goal both break the identity (its stated scope).

## Scope & honest disclosures
- Only C2 (Proposition 4.1) is the exact identity — verified. C1 is the conceptual framing; C3 (offline-GCRL
  benchmarks) needs training and is out of scope.
- Official code `Simple-Robotics/hierarchical-survival-value-learning` (JAX, `survival.py`) implements the
  identity — cross-check reference; core math is clean-room numpy.

Logbook: https://huggingface.co/spaces/DineshAI/qIOcJSCGn2
