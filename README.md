# SVL: Goal-Conditioned Reinforcement Learning as Survival Learning

This repository is the ICML 2026 reproduction and evidence record for **SVL: Goal-Conditioned Reinforcement Learning as Survival Learning** by Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, and Justin Carpentier.

- Paper: [arXiv 2604.17551](https://arxiv.org/abs/2604.17551)
- Paper page: [Simple Robotics project page](https://simple-robotics.github.io/publications/survival-value-learning/)
- OpenReview: [qIOcJSCGn2](https://openreview.net/forum?id=qIOcJSCGn2)
- Official implementation: [Simple-Robotics/hierarchical-survival-value-learning](https://github.com/Simple-Robotics/hierarchical-survival-value-learning)
- Public reproduction logbook/artifacts: [Hugging Face Space](https://huggingface.co/spaces/DineshAI/qIOcJSCGn2)

The default branch is a small, CPU-runnable clean-room audit. The purpose-based branches preserve the larger source, feasibility, and campaign history without treating configuration or readiness work as successful paper reproduction.

## What the paper does

SVL treats the time required to reach a goal as a random variable. It models that time-to-goal distribution through survival and hazard functions, trains the hazard model with maximum likelihood on both successful event times and right-censored trajectories, and obtains the goal-conditioned value by a discounted sum of survival probabilities. HSVL combines that survival critic with a hierarchical actor trained by advantage-weighted regression.

Under the paper's sparse per-step penalty and terminate-on-success assumptions, Proposition 4.1 is

```text
V^π(s, g) = −Σ[t ≥ 0] γ^t Pr(T^π(s, g) > t).
```

The paper also reports three practical value estimators (finite horizon, piecewise-constant hazard, and piecewise-constant survival), a hazard MLE/ERM result, and offline OGBench comparisons against TD, Monte Carlo, contrastive, and hierarchical baselines.

## Claim status and how evidence is produced

| Claim | Paper-level statement | Local status | Evidence path |
| --- | --- | --- | --- |
| C1 | SVL's survival-learning framing and released-scale hazard architecture | **TOY / FULL-SCALE BLOCKED** | A bounded PointMaze distribution audit exists in [`baseline/frozen-four-of-twelve`](https://github.com/MachineLearning-Nerd/icml26-survival-value-learning/tree/baseline/frozen-four-of-twelve), but it used a reduced model and training budget; exact released-scale twin HSVL, actors, 1M updates, and policy evaluation were not completed. |
| C2 | Goal-conditioned value equals discounted survival probability | **VERIFIED** | [`repro/src/run_svl.py`](repro/src/run_svl.py) independently computes Bellman, killed-chain survival, and value-iteration values on finite random MDPs, then runs state-dependent-reward and non-absorbing-goal negative controls. |
| C3 | Hazard MLE/ERM has the stated consistency and asymptotic-normality behavior | **ROUTE IDENTIFIED / NOT VERIFIED** | [`audit/claim-3-mle-erm-theorem`](https://github.com/MachineLearning-Nerd/icml26-survival-value-learning/tree/audit/claim-3-mle-erm-theorem) records the needed iid, censoring, specification/KL, identifiability, domination, and information assumptions; no independent proof artifact was completed. |
| C4 | HSVL versus HIQL on HumanoidMaze-giant | **INCONCLUSIVE / BLOCKED** | [`audit/claims-4-5-benchmark-readiness`](https://github.com/MachineLearning-Nerd/icml26-survival-value-learning/tree/audit/claims-4-5-benchmark-readiness) and the method/seed branches preserve the exact four-seed route, but accelerator-scale training and 50-episode evaluation were not completed. |
| C5 | HSVL versus HIQL on state and visual AntMaze-giant | **INCONCLUSIVE / BLOCKED** | The same matched-data, four-seed, state-plus-visual route is recorded, but the released visual path and full paper-scale evaluation are incomplete. |
| C6 | Flat SVL versus re-evaluated CRL on HumanoidMaze-giant | **INCONCLUSIVE / BLOCKED** | [`audit/claim-6-flat-svl-crl`](https://github.com/MachineLearning-Nerd/icml26-survival-value-learning/tree/audit/claim-6-flat-svl-crl) records the route; the pinned release has hierarchical HSVL but not the paper-matched flat SVL and DDPG+BC actor. |

The historical campaign log records a live challenge state of 4/12 after C2 was accepted and the other routes remained toy, inconclusive, or blocked. That is campaign provenance, not an author-endorsed score and not a claim that the full paper was reproduced.

## Reproduce the verified result

The clean-room C2 audit needs only Python, NumPy, and SciPy. `pytest` is optional for the five historical unit tests.

```bash
python3 -m pip install numpy scipy
python3 repro/src/run_svl.py
python3 -m pytest repro/tests/
```

The runner writes [`outputs/svl_summary.json`](outputs/svl_summary.json). A successful run should report three-method agreement below `1e-10`; the current recorded run is approximately `1.4e-12`. The negative controls must fail, demonstrating that the theorem's sparse-reward and terminate-on-success assumptions are load-bearing.

The official JAX implementation is retained under [`upstream/`](upstream/) as provenance and an implementation cross-check. It is not silently substituted for the clean-room NumPy proof.

## Repository and branch guide

- [`docs/CLAIM_EVIDENCE.md`](docs/CLAIM_EVIDENCE.md) explains how each claim is produced, what counts as evidence, and what remains blocked.
- [`docs/BRANCH_AUDIT.md`](docs/BRANCH_AUDIT.md) maps every original branch to its clean final name, purpose, tip, and evidence boundary.
- [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md) records paper, official-code, local, and public-artifact provenance.
- [`docs/PUBLICATION_GATE.md`](docs/PUBLICATION_GATE.md) documents the fail-closed verification scope.
- [`STATUS.md`](STATUS.md) is the concise current status record.

The `audit/` branches contain claim and protocol audits; `experiment/` branches contain dependency, integration, throughput, and paper-scale seed work; `baseline/` branches preserve campaign snapshots; and `release/` contains the text-only logbook release. A branch name describes an experiment or audit, not a successful reproduction verdict.

## Citation

```bibtex
@article{tiofack2026svl,
  title={SVL: Goal-Conditioned Reinforcement Learning as Survival Learning},
  author={Tiofack, Franki Nguimatsia and Schramm, Fabian and Le Hellard, Th{\'e}otime and Carpentier, Justin},
  journal={arXiv preprint arXiv:2604.17551},
  year={2026},
  url={https://arxiv.org/abs/2604.17551}
}
```

## Thank you

Thank you to Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, and Justin Carpentier for developing SVL, releasing the HSVL implementation, and making the paper and benchmark design available for independent study. This repository is an independent reproduction/audit record; please cite the paper and consult the authors' official implementation for authoritative releases.
