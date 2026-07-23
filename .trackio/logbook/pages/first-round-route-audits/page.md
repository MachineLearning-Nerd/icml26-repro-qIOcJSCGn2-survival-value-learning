# First-round route audits — C1, C3-C6

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_route_01", "created_at": "2026-07-23T11:23:18+00:00", "title": "Scope and source", "pinned": true, "pinned_at": "2026-07-23T11:23:18+00:00"}
-->
These are source/protocol feasibility audits only. They do not independently verify or falsify a
scientific claim, do not change the live judge verdicts, and do not estimate an unjudged score.

Primary paper source: `https://ar5iv.labs.arxiv.org/html/2604.17551`, retrieved 2026-07-23,
SHA-256 `69f0db8b7819cac262f4c91e945fc7a6d4b011e7d603fbfe3f0844fa5acecec0`.

| Route | Exact paper anchor | Child experiment | Commit |
| --- | --- | --- | --- |
| C1 | Sections 4.1/4.3; Appendix Table 3 | `73a136ac-0437-4d3e-a3c6-9feb74081eb9` | `51a69d5` |
| C3 | Section 4.2, Lemma 4.2 and Eq. 14 | `e31fdb0f-f8c6-409e-ad94-47459a0c1fc6` | `70afb86` |
| C4-C5 | Section 5.1, Table 1; Appendix Tables 3 and 6-8 | `5356d237-a6c6-4209-ad81-bb54737c900e` | `a5c8685` |
| C6 | Section 5.2, Table 2; Appendix Table 5 | `c46c2e3e-de83-4cda-97aa-6005f7fe4fa5` | `06bcd6e` |

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_route_02", "created_at": "2026-07-23T11:23:19+00:00", "title": "Route decisions"}
-->
## Route decisions

- **C1 — BLOCKED.** The exact route requires released width-512 twin-head HSVL, both actors,
  1,000,000 updates, deterministic evaluation, held-out raw outputs, independent reconstruction,
  and destructive controls. The frozen 9,778,698-parameter T4 preflight measured 21.451733 median
  and 6.321757 conservative updates/s, projecting 12.95-43.94 T4 hours. The bounded CPU model
  remains toy.
- **C3 — promote route only.** A proof at the actual finite-dimensional MLE scope can be CPU-only:
  explicitly state iid/non-informative censoring, specification or KL target, identifiability,
  interiority, domination/ULLN, and nonsingular-information conditions; then independently
  reconstruct consistency and asymptotic normality. Removing identifiability or nonsingular
  information must be destructive controls. A small Monte Carlo alone remains toy.
- **C4-C5 — BLOCKED.** Faithful evidence requires HSVL and HIQL, seeds 0-3, matched data/configs,
  1,000,000 state or 500,000 visual updates, and 50 episodes per task/seed. C5 requires both state
  and visual AntMaze-giant. The pinned HIQL visual route uses `impala_small` and
  `eval_on_cpu=0`; pinned HSVL constructs Identity encoders and exposes no visual encoder field.
- **C6 — BLOCKED.** The pinned HSVL release registers hierarchical HSVL only and does not contain
  flat SVL. The pinned OGBench CRL does not establish the paper-matched depth-6 DDPG+BC actor.
  Even with exact recovered source, the four-seed, two-method HumanoidMaze-giant comparison is
  accelerator-scale.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_route_03", "created_at": "2026-07-23T11:23:20+00:00", "title": "Execution and preservation"}
-->
## Execution and preservation

Four local C1 run ids were attempted:

- `e5787f97-eb1d-4ac8-bebd-b49b3b49815d`
- `4b2fd695-0c6d-4cca-8da7-e268639a1c9f`
- `e4d6e754-1100-442c-944f-efd4b65a71b4`
- `6933a487-1180-4f65-871e-d4d7b3950ef5`

All died before project evaluation during fixed-runner dependency setup. Available OpenResearch
logs contain at most the clone line. No route `EVAL.md`, raw outputs, text artifacts, or resolved
route-run environment exists. The other routes were not launched after the repeated environmental
failure.

No HF job or CPU-upgrade was launched, so there are no job ids, URLs, runtime/cost, or remote
result locations. No GPU was used. The judged Space revision
`db72d25f24ec51384021a3081e3aaad17f229572` was not modified. This logbook update is text-only
on an OpenResearch child branch and has not been synced to the Space.
