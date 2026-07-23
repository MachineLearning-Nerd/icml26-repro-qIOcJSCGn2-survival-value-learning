# qIOcJSCGn2 six-claim OpenResearch campaign prompt

You are operating the existing reproduction of *SVL: Goal-Conditioned Reinforcement Learning as
Survival Learning* (OpenReview `qIOcJSCGn2`, arXiv `2604.17551`). Your objective is to maximize the
live judge score, up to an honest 12/12, by independently **verifying or falsifying** each of the six
claims in `.openresearch/judge_contract.json`. A high score is an objective, never a conclusion.
Never predict or report a score that the live judge has not awarded.

## Immutable facts and source hierarchy

- The current judged HF Space is `DineshAI/qIOcJSCGn2` at exact revision
  `db72d25f24ec51384021a3081e3aaad17f229572`.
- Its live verdict is 4/12: C1 `toy`, C2 `verified`, C3 `toy`, and C4-C6 `inconclusive`.
- Preserve C2 without weakening its wording, verifier, evidence path, or page-tree reachability.
- Read exact claim wording from
  `https://ar5iv.labs.arxiv.org/html/2604.17551` first. The HTML fetched on 2026-07-23 for this
  audit had SHA-256 `69f0db8b7819cac262f4c91e945fc7a6d4b011e7d603fbfe3f0844fa5acecec0`.
  Fall back to arXiv native HTML and use the PDF only as a last resort. Record the URL, section,
  theorem/table anchor, retrieval date, and content hash used by every claim audit.
- There is no other public or judged `qIOcJSCGn2` reproduction to copy or use as claim evidence.
  Methodological examples only: `DineshAI/KUBkuPwGf4@b0c8fb9` demonstrates exhaustive independent
  theorem checks; `DineshAI/SBfIuo2Ets@dbdc542` demonstrates a real-data full-scale benchmark with
  destructive controls; `DineshAI/VQt4w3lElX@756d5f2` demonstrates honest source-backed
  falsification. Do not reuse their evidence, code, prose, or outputs.

## Non-negotiable constraints

1. Work only in child experiment branches. The baseline is frozen after its first run.
2. Keep the inherited command `bash repro/orx/run_evaluation.sh` unchanged. Vary committed
   code/configuration, not the command or environment.
3. CPU work may run locally first and then on HF Jobs with `cpu-upgrade`. GPU/T4 and all other GPU
   flavors are forbidden. If faithful evidence is genuinely GPU-bound, mark the claim `BLOCKED`
   with the measured reason; do not substitute a toy CPU experiment or silently reduce scale.
4. Before any HF job, list running/scheduling `DineshAI` jobs with an explicit cached token and
   prove the route is not a duplicate. Never cancel, signal, reprioritize, or overwrite another job.
5. Use the project `.venv` and `pip` installation performed by the fixed runner for bounded local
   checks. Pin versions and record the resolved environment for every scientific run.
6. Every run must write `EVAL.md` and text evidence under `.openresearch/artifacts/`. Remote job
   storage is ephemeral: persist checkpoints/results safely, but do not commit large archives,
   checkpoints, datasets, caches, Trackio-generated static assets, or credentials.
7. Never publish or modify the judged HF Space from this campaign. Stop at a verified text-only
   release candidate. Publication requires a separate human-approved release step.
8. Never replace the Space root `logbook.json` with `.trackio/logbook/logbook.json`. Before a release
   candidate is accepted, download the exact judged HF revision and prove its file set, claim nodes,
   and existing evidence references are a subset of the candidate. Existing claims must be
   unchanged or strengthened.
9. A passing self-written verifier is insufficient. Each upgraded claim needs a primary verifier,
   an independent auditor that reconstructs from raw outputs without importing the primary
   computation, at least one destructive/negative control, source/config/data hashes, and a
   fail-closed manifest.
10. Report failed, negative, and falsifying results. Never select tasks, seeds, checkpoints, or
    metrics after observing outcomes. Preregister them in the child description and committed
    config before launching.

## Claim-specific acceptance targets

- **C1:** remove every scale/component criticism named in the live verdict, or produce an exact
  falsification. A width/bins/update bump that remains below the released architecture is still toy.
- **C2:** already verified. Re-run its deterministic checks as a regression gate; otherwise leave it
  unchanged.
- **C3:** prove the stated MLE/ERM result at its actual mathematical scope under explicit regularity
  assumptions, or construct a valid-assumption counterexample. A small logistic-hazard Monte Carlo
  alone remains toy.
- **C4:** faithfully reproduce or falsify the four-seed HumanoidMaze-giant HSVL-vs-HIQL comparison
  with matched data, configs, evaluation episodes, and uncertainty. If GPU-bound under policy,
  record `BLOCKED`; do not compare against the paper's number alone.
- **C5:** faithfully reproduce or falsify both visual-AntMaze-giant and AntMaze-giant comparisons.
  One task cannot stand in for the conjunction. Apply the same preregistration and matched-baseline
  requirements as C4.
- **C6:** faithfully reproduce or falsify the flat SVL-vs-re-evaluated-CRL comparison on
  HumanoidMaze-giant with the same DDPG+BC actor and four-seed protocol. Source inspection alone is
  not performance evidence.

## Experiment-tree strategy

Use stacked bushes, not a flat fan:

1. Run the frozen baseline once. Confirm it reports the live 4/12 state and passes the protected C2
   regression checks.
2. First round: create sibling **route-audit** children for C1, C3, C4-C5, and C6. These children do
   source/protocol feasibility work only and must not claim scientific credit.
3. Promote only a route whose exact scope is CPU-reachable and whose preregistered evidence could
   change the corresponding judge verdict. Descend from that winner into a bounded preflight child,
   then into a paper-scale child only after the preflight passes.
4. Stop a route after three genuine failed/regressed attempts or when the exact work is GPU-bound.
5. End with one release-candidate child that combines only passed additive evidence. It must run all
   primary verifiers, independent auditors, negative controls, C2 regression tests, manifest/hash
   checks, and the judged-revision subset comparison.

For every completion, read the full `EVAL.md`, text artifacts, logs, and the three-dot Git diff from
the parent. Do not promote based on exit status alone.

## Required final report

Produce a self-contained report containing:

- exact evidence and verdict recommendation for C1-C6 (`verified`, `falsified`, `toy`,
  `inconclusive`, or `BLOCKED`);
- commands, environment, data/source/config hashes, seeds, scale, raw outputs, and independent audit;
- negative controls and any failed attempts;
- a clear distinction between current live 4/12, candidate evidence, and unjudged possibilities;
- CPU job IDs, URLs, runtime/cost, and persisted result locations;
- the exact candidate text-file allowlist and proof that the judged HF file/claim tree is preserved.

The correct terminal outcome may be less than 12/12. Honesty, reproducibility, and preservation of
the banked C2 claim take priority over a numerical target.
