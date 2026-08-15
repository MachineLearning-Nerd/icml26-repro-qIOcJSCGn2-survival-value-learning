# Claim-to-evidence map

This file separates paper claims from the experiments that can actually support them. A configuration, dependency repair, throughput probe, or route-feasibility note is not promoted to scientific evidence unless it measures the claim under the paper's scope.

## Evidence labels

- `VERIFIED`: the stated local test directly exercises the claim and its assumptions.
- `TOY`: a bounded or reduced experiment is informative but not faithful to the paper's reported scale.
- `INCONCLUSIVE`: the available result does not establish the claim either way.
- `BLOCKED`: the faithful route could not be executed with the available source, data, hardware, or evaluator.
- `NOT VERIFIED`: no independent claim-level artifact is present.

## C1 — survival-learning architecture and distributional value learning

**Paper path:** Sections 4.1–4.3 and Appendix Table 3. The faithful route requires the released twin-head width-512 HSVL critic, both hierarchical actors, the stated data/configuration, 1,000,000 updates, deterministic policy evaluation, and independent reconstruction of held-out survival outputs.

**What exists:** `baseline/frozen-four-of-twelve` contains a bounded PointMaze distribution audit, raw-output reconstruction, controls, and an exact-width T4 execution preflight. The audit used a reduced width, fewer updates, fewer bins, and no actor/policy evaluation. The preflight demonstrates execution feasibility only; it is not a paper result.

**Conclusion:** `TOY / FULL-SCALE BLOCKED`. The full C1 claim is not reproduced.

## C2 — Proposition 4.1

**Claim:** with a sparse per-step penalty and terminate-on-success,

```text
V^π(s,g) = −Σ[t ≥ 0] γ^t Pr(T^π(s,g) > t).
```

**Production path:**

1. `repro/src/svl.py` constructs a finite stochastic policy-induced MDP, makes goal states absorbing, and computes the Bellman value on the full chain.
2. The same file constructs the killed non-goal substochastic matrix and computes the discounted survival sum by a matrix solve.
3. `repro/src/run_svl.py` independently computes value iteration and compares all three routes over 6 seeds, 3 state sizes, and 2 discount factors.
4. The runner changes the reward to a state-dependent penalty and removes goal absorption as negative controls. Both controls must disagree with the identity.

**Observed result:** the current run reports `identity.ok=true`, both controls `ok=true`, and maximum agreement approximately `1.37e-12`.

**Conclusion:** `VERIFIED` for the stated finite-MDP/sparse-reward/terminate-on-success scope. This does not verify arbitrary dense-reward returns or neural hazard estimation.

## C3 — hazard MLE/ERM result

**Paper path:** Section 4.2, Lemma 4.2, and Equation 14. A faithful proof audit must state and use iid or suitable empirical-process assumptions, non-informative right censoring, correct specification or a KL target, identifiability, interiority/regularity, domination or a uniform law of large numbers, and nonsingular information for asymptotic normality.

**What exists:** `audit/claim-3-mle-erm-theorem` records the proof route and destructive controls that would remove identifiability or information assumptions. No independent theorem certificate was completed.

**Conclusion:** `ROUTE IDENTIFIED / NOT VERIFIED`. A small simulation or a successful neural training run would not by itself prove Lemma 4.2.

## C4 — HSVL versus HIQL on HumanoidMaze-giant

**Paper path:** Section 5.1, Table 1, and the corresponding appendix tables. Faithful evidence requires matched source/configuration, four seeds, the reported training budget, and 50 evaluation episodes per task/seed for HSVL and HIQL.

**What exists:** `audit/claims-4-5-benchmark-readiness`, `experiment/paper-scale-hsvl-humanoid-seed-*`, and `experiment/paper-scale-hiql-humanoid-seed-*` preserve configuration and readiness work. CPU integration, dependency, throughput, and runner branches do not contain the required paper-scale policy results.

**Conclusion:** `INCONCLUSIVE / BLOCKED`. No benchmark number is independently claimed.

## C5 — state and visual AntMaze-giant comparisons

**Paper path:** Section 5.1, Table 1, and appendix tables. The conjunction requires both state and visual tasks, matched HSVL/HIQL paths, four seeds, and the reported evaluation protocol.

**What exists:** the matching AntMaze and visual seed branches and `audit/claims-4-5-benchmark-readiness` preserve route definitions. The visual encoder/evaluation path and full accelerator campaign remain incomplete.

**Conclusion:** `INCONCLUSIVE / BLOCKED`. Table arithmetic witnesses are not policy-performance evidence.

## C6 — flat SVL versus re-evaluated CRL

**Paper path:** Section 5.2, Table 2, and Appendix Table 5. The comparison is intended to hold the flat DDPG+BC actor fixed while changing the critic, then compare four-seed success rates on HumanoidMaze-giant.

**What exists:** `audit/claim-6-flat-svl-crl` records the source and protocol audit. The pinned official release contains hierarchical HSVL but not the paper-matched flat SVL implementation and DDPG+BC actor, and the required accelerator-scale evaluation was not run.

**Conclusion:** `INCONCLUSIVE / BLOCKED`. The claim is not independently reproduced.

## Boundary

The paper's reported 81% HSVL result, Table 1 comparisons, Table 2 comparisons, and MLE asymptotics must not be described as reproduced by this repository. The only claim promoted on the default branch is Proposition 4.1 under its explicit assumptions.
