# Three-interpretation retry — C1, C4-C6

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_interp_01", "created_at": "2026-07-23T12:14:51+00:00", "title": "Scope and interpretation bush", "pinned": true, "pinned_at": "2026-07-23T12:14:51+00:00"}
-->
Three materially different readings of the paper were preregistered as sibling experiments:

1. **Literal protocol:** exact released architecture, task, method, comparator, four-seed training,
   and five-task × 50-episode evaluation.
2. **Official artifact witness:** independently evaluate immutable author checkpoints or
   reconstruct author raw four-seed outputs without retraining.
3. **Table arithmetic falsification:** exhaustively test whether the printed integer mean ± std
   entries can arise from four integer success counts at 250 episodes per seed.

Source: `https://ar5iv.labs.arxiv.org/html/2604.17551`, retrieved 2026-07-23,
SHA-256 `69f0db8b7819cac262f4c91e945fc7a6d4b011e7d603fbfe3f0844fa5acecec0`.

These retries do not estimate an unjudged score. The live judged state remains **4/12** at
revision `db72d25f24ec51384021a3081e3aaad17f229572`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_interp_02", "created_at": "2026-07-23T12:14:52+00:00", "title": "Runs and controls"}
-->
## Runs and controls

| Interpretation | Experiment / commit | Local run | Result |
| --- | --- | --- | --- |
| Literal attempt | `c336d8da-d907-496d-b5fb-9b86e9a095c8` / `20b582a` | `34e8ea20-1cbd-4f4e-9721-565d118e70a4` | `AUDIT_FAILED`: one paraphrased C1 regex; no scientific conclusion |
| Literal gate fix | `ebea9118-6d96-47fe-a726-7341205411f9` / `0266307` | `2bca9ff6-4df5-4d2f-a492-243d9af9933a` | `BLOCKED` |
| Official artifacts | `403a3965-2068-4415-8402-b7e40aa1ac3c` / `2bebeb4` | `0b659216-d0ed-4be9-ad12-debf63103974` | `DO_NOT_PROMOTE`; 0 eligible artifacts |
| Table arithmetic | `40bd5c1a-be09-4e4e-8972-2ed849f166d9` / `e5b4548` | `ee6b2379-bb2c-4c03-82ae-0400bef03df8` | `compatible_not_verified` |

Every completed retry passed 5/5 deterministic tests and the protected C2 regression. Integrity
controls passed: tampered source/file hashes were rejected; an invented artifact without a content
hash was rejected; impossible-mean and impossible-variance table controls were rejected. The
table route also passed an independent exact-Fraction auditor that does not import the primary
search.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_interp_03", "created_at": "2026-07-23T12:14:53+00:00", "title": "Honest claim boundary"}
-->
## Honest claim boundary

- **C1 remains toy.** Exact released-scale work is accelerator-bound; no eligible author
  checkpoint exists; table arithmetic is not applicable.
- **C2 remains verified.** Its wording, verifier, evidence page, and reachability were not changed.
- **C4 remains inconclusive.** The `81 ± 1` and `12 ± 4` entries both have integer-count witnesses,
  but that is not policy-performance evidence.
- **C5 remains inconclusive.** All four entries for both visual and state AntMaze conjuncts have
  witnesses; neither conjunction was reproduced.
- **C6 remains inconclusive.** `45 ± 2` and `7 ± 0` have witnesses; the actor identity and policy
  performance remain unmeasured.

The literal routes are **BLOCKED** because faithful execution requires forbidden GPU compute.
The CPU-reachable artifact and arithmetic routes cannot change a judge verdict, so no HF
CPU-upgrade was launched. There are no HF job IDs, URLs, charges, or remote result locations.
No GPU ran and no scientific evidence or score claim was changed by this text-only page.
