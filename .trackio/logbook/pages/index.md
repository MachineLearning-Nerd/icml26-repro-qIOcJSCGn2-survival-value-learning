# Repro - SVL Survival Value Learning (qIOcJSCGn2)

Clean-room reproduction of *SVL: Goal-Conditioned RL as Survival Learning* (Tiofack et al.;
arXiv 2604.17551), OpenReview `qIOcJSCGn2`.

Proposition 4.1: under sparse per-step penalty + terminate-on-success, `V^π(s,g) = −Σ_t γ^t Pr(T^π(s,g)>t)`.

## Claims
| Claim | Statement | Verdict |
| --- | --- | --- |
| **C2** | Closed-form value = discounted sum of survival probabilities | **VERIFIED** |

(C1 = conceptual reframe realized by C2; C3 = offline-GCRL benchmarks, empirical, out of scope.)

## Pages
- [Claim 2 — survival identity](claim-2-survival-identity) · [Methods & environment](methods-environment)
- [Negative controls](negative-controls) · [Conclusion](conclusion)

Three independent methods + two negative controls. CPU only.
