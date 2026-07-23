# Repro - SVL Survival Value Learning (qIOcJSCGn2)

Clean-room reproduction of *SVL: Goal-Conditioned RL as Survival Learning* (Tiofack et al.;
arXiv 2604.17551), OpenReview `qIOcJSCGn2`.

Proposition 4.1: under sparse per-step penalty + terminate-on-success, `V^π(s,g) = −Σ_t γ^t Pr(T^π(s,g)>t)`.

## Claims
| Claim | Statement | Verdict |
| --- | --- | --- |
| **C1** | Survival-learning framing at the released architecture and scale | **TOY; exact route BLOCKED** |
| **C2** | Closed-form value = discounted sum of survival probabilities | **VERIFIED** |
| **C3** | MLE/ERM consistency and asymptotic normality at the stated scope | **TOY; CPU proof route identified** |
| **C4** | HSVL vs HIQL on HumanoidMaze-giant | **INCONCLUSIVE; BLOCKED** |
| **C5** | HSVL vs HIQL on both AntMaze-giant and visual-AntMaze-giant | **INCONCLUSIVE; BLOCKED** |
| **C6** | Flat SVL vs re-evaluated CRL on HumanoidMaze-giant | **INCONCLUSIVE; BLOCKED** |

The current live judge score remains **4/12**. Route-audit recommendations are feasibility
findings only; they are not new judged verdicts and no candidate score is estimated.

## Pages
- [Claim 2 — survival identity](claim-2-survival-identity) · [Methods & environment](methods-environment)
- [First-round route audits](first-round-route-audits) · [Negative controls](negative-controls)
- [Conclusion](conclusion)

Three independent C2 methods + two executed C2 negative controls. The first-round route audit
records exact source/protocol constraints, one unexecuted audit-harness control, and four local
environment failures. No HF job, GPU, or judged-Space modification was made.
