# Status

## Identity

- Paper: *SVL: Goal-Conditioned Reinforcement Learning as Survival Learning*
- Authors: Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, Justin Carpentier
- arXiv: `2604.17551`
- OpenReview: `qIOcJSCGn2`
- Public campaign Space: <https://huggingface.co/spaces/DineshAI/qIOcJSCGn2>
- Repository status: clean-room C2 audit published; full paper-level reproduction is not claimed

## Claim matrix

| Claim | Status | Why |
| --- | --- | --- |
| C1 | `TOY / FULL-SCALE BLOCKED` | A reduced PointMaze distribution audit exists; the exact released twin HSVL, actors, 1M updates, and policy evaluation were not completed. |
| C2 | `VERIFIED` | Bellman solve, killed-chain survival solve, and value iteration agree to approximately `1.4e-12`; two negative controls break the identity. |
| C3 | `ROUTE IDENTIFIED / NOT VERIFIED` | The finite-dimensional MLE proof conditions are recorded, but no independent theorem artifact was completed. |
| C4 | `INCONCLUSIVE / BLOCKED` | HumanoidMaze-giant four-seed HSVL/HIQL evidence was not completed. |
| C5 | `INCONCLUSIVE / BLOCKED` | State and visual AntMaze-giant four-seed evidence was not completed. |
| C6 | `INCONCLUSIVE / BLOCKED` | The paper-matched flat-SVL and DDPG+BC comparison path is not present in the pinned release. |

## Reproduction state

- `python3 repro/src/run_svl.py` passes the C2 identity and both negative controls in the current environment.
- The repository's original campaign recorded 5/5 unit tests; `pytest` is not installed in the current base environment, so that suite was not rerun here.
- Full accelerator-scale benchmark claims remain explicitly unverified.
- No author-endorsed score or claim promotion is made.
