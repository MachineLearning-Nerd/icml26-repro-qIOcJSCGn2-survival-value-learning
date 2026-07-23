# qIO C1-C3 distinct-route re-audit - 2026-07-20

This is a paper-local, read-only readiness audit. It did not run a test suite,
train or evaluate a model, signal a process, submit or alter an HF Job, or
publish a Space change.

## Exact official state

- Existing Space: `DineshAI/qIOcJSCGn2`
- Current and judged SHA: `757e63d6377d6a3789657d75e354306c95d6e43f`
- Official verdict observed at `2026-07-19T17:01:55+00:00`: **3/6**
- C1: `toy`
- C2: `verified`
- C3: `inconclusive`
- Overall quality: `medium`

The Space is still `RUNNING` at that exact SHA. Its executive summary and
conclusion contain stale 2/6 wording, but correcting that wording is a public
quality repair, not new C1 or C3 evidence and not a predicted score increase.
C2 must remain byte-identical in any future overlay; its public page SHA-256 is
`ac130af15a3f1e82ef62310fb8596f63030f1cacf6cb1974d40c030052ec7951`.

## Current claim evidence

### C1

The accepted evidence is a three-seed released PCS-path experiment on official
PointMaze-large data. Held-out censored NLL is `2.398 +/- 0.041`, versus
`3.158 +/- 0.006` for the repeated-hazard baseline. Probability normalization,
survival monotonicity, a shuffled-target control, raw predictions, and an
independent NumPy reconstruction are present.

- summary SHA-256:
  `49e48feb0eb1cf9ed5304ee0ce3f6cc3c7f83607e8b8a8faff5be5ed4d2d0435`
- independent audit SHA-256:
  `ef99588b060aa3631be80c0479cc8f621742fa1eb4d1f72ed15bed32f8dc2ca2`

The official `toy` boundary is still correct: width 64 instead of 512, 2,000
instead of 1,000,000 updates, 53 instead of 500 effective intervals, one head
instead of the released twin critic, and no actor or policy evaluation.

### C2

C2 is already verified by the exact discounted-survival identity and scope
controls. There is no honest score-improving C2 work. Its evidence and public
page should be preserved rather than rerun.

### C3

There is no accepted policy-performance evidence. The exact-width 100-update
PointMaze preflight proves only CUDA execution, checkpoint/resume, and model
inventory. It is one seed, 0.01% of the paper update count, and has no policy
evaluation or HIQL comparator.

## Protected local ownership and live HF state

PID `41861` remains alive at 0% CPU as
`run_full_agent_probe_queue.py --wait-pid 41276 ...`. It has no active qIO
child. It must exit naturally; moving or stopping this waiting queue would not
free meaningful CPU.

The live HF account snapshot contained eight unrelated `cpu-upgrade` Jobs and
one `t4-small` Job. The T4 is the protected Flux Job
`DineshAI/6a5dd080bee6ee1cf4ed215e`; there is no running qIO Job. No qIO GPU
work should launch until Flux is terminal and the account/ledger gates are
freshly rechecked.

## Zero-compute Table 1 audit

The official PDF SHA-256 is
`f468cfd9e6ea9e06e185a01e3bf81bc703565d55fa7db98c0b4211499c05ec72`.
Visual inspection of pages 1 and 8 confirms the abstract wording and Table 1
layout. Table 1 has 22 environment-level means.

Against the strongest of the six baseline means in each row, HSVL has 12 wins,
one tie, and nine losses. Its unweighted mean across rows is `56.2273`, versus
`56.7273` for the per-row strongest baseline, an average gap of `-0.5` points.
Against HIQL alone, HSVL has 16 wins, one tie, and five losses, with a mean
pairwise gap of `+10.5` points. Against CRL alone it has 17 wins, one tie, and
four losses, with a mean gap of `+19.9545` points.

The largest contradictions to a universal reading of the Table 1 caption
"outperforms six baselines across benchmark tasks" are:

| Environment | Strongest baseline | HSVL | Gap |
| --- | ---: | ---: | ---: |
| `cube-single-play-v0` | GCIQL `68` | `14` | `-54` |
| `puzzle-3x3-play-v0` | GCIQL `95` | `54` | `-41` |
| `antmaze-large-stitch-v0` | HIQL `67` | `31` | `-36` |
| `visual-antmaze-teleport-stitch-v0` | HIQL `37` | `25` | `-12` |
| `antmaze-medium-stitch-v0` | HIQL `94` | `83` | `-11` |

This is a legitimate literal-scope caveat, but it is **not** independent C3
reproduction or decisive falsification. The abstract's narrower pairwise
statement - matching or surpassing strong hierarchical TD and Monte Carlo
baselines while excelling on complex long-horizon tasks - is broadly consistent
with the same table, especially against HIQL and CRL and on giant navigate
tasks. Treating the authors' own reported table as empirical verification would
be circular. Therefore this zero-compute audit should inform a preregistered
decision rule, not be promoted as a two-point C3 result.

## Materially distinct score-seeking route

The best paper-local route that does not duplicate PID `41861` is a matched
AntMaze-giant campaign, not another PointMaze preflight:

1. Use `antmaze-giant-navigate-v0`, a reported Table 1 long-horizon task.
2. Run exact released HSVL and official OGBench HIQL for the same four frozen
   seeds, 1,000,000 updates, and 50 evaluation episodes.
3. Use identical train/validation bytes and a preregistered evaluator and seed
   manifest. Preserve every run and failure.
4. For HSVL, independently audit both survival heads on held-out censored data:
   normalization, monotonicity, NLL/calibration, twin disagreement, shuffled
   targets, and temporal reversal.
5. Decide C3 symmetrically from the matched seed-level distributions. A
   reproduced advantage supports the claim; failure to match HIQL is honest
   falsification evidence.

The target is materially distinct from the protected PointMaze 100-update
queue because it uses a different official dataset, a matched HIQL comparator,
paper-scale seeds/updates, policy evaluation, and an anchored Table 1 result.
It can also strengthen C1 through exact twin-survival diagnostics from the same
HSVL checkpoints.

The locally recovered official AntMaze-giant inputs are already pinned:

- train: 241,141,207 bytes,
  `387791c414cfba2d77eda64a47f3b6db7950bdf146ae901e4f08c2c2f9e0f3aa`
- validation: 24,052,175 bytes,
  `61d40e91899f496030fb4ee73ffd61c20dad502ab6cc8075744867097930ce71`

The paper reports HSVL `74 +/- 1` versus HIQL `65 +/- 5`. This is a smaller
margin than HumanoidMaze-giant, but AntMaze avoids a new licensed/data transfer
gate and is the most ready distinct C3 target in this paper directory.

## Readiness blockers

This route is not launch-ready:

- the official HIQL source/config gate was closed in the follow-up documented
  in `docs/ANTMAZE_GIANT_HIQL_SOURCE_CONFIG_READINESS_20260720.md`: OGBench
  tag `v1.2.1`, commit `1d4140997f60c52c6fb0702ec100dc988b18c548`,
  tree `0919ecd9e71ff437ed05875e921c94cf439107be`, and the complete official
  `impls` subtree are now preserved in a deterministic paper-local archive;
- exact resolved released HSVL and official HIQL settings are frozen together,
  but the paper's original four seed identities are not public. Candidate
  matched seeds `[0, 1, 2, 3]` are explicitly preregistered as new reproduction
  seeds, not recovered author seeds;
- the official HIQL requirements are range-based and have no lockfile, so a
  concrete hashed reproduction environment still must be created and disclosed;
- no matched terminal runner/importer/auditor chain exists;
- no target-specific dual throughput/resume measurement exists;
- headless evaluator compatibility is unmeasured for both methods;
- PID `41861` remains protected, Flux occupies T4, and no central reservation
  exists.

The released HSVL default also requests 800 logarithmic edges while paper
Table 3 says `K=500`. The prior preflight observed 499 unique intervals after
de-duplication. The route must record the requested edges, actual unique
intervals, and paper value explicitly rather than silently treating them as
identical.

## Cost boundary and useful work now

Current HF rates from `hf jobs hardware` are `$0.03/hour` for `cpu-upgrade`
and `$0.40/hour` for `t4-small`.

The only genuinely useful `cpu-upgrade` action would be a later bounded
readiness preflight after the HIQL bundle and exact protocol are frozen:

- unpack and hash both source trees and both AntMaze datasets;
- instantiate exact HSVL and HIQL on CPU;
- perform a tiny checkpoint/reload path for each;
- execute one headless evaluator episode per method;
- persist all outputs even on failure.

A 30-minute planning ceiling would cost `$0.015`; a one-hour hard ceiling would
cost `$0.03`. This can close dependency, serialization, and evaluator gates but
cannot measure GPU throughput and is not C1/C3 evidence. Because eight other
`cpu-upgrade` Jobs are currently active and the bundle is incomplete, launching
it now would add contention without answering the missing source/config gate.

After Flux releases T4, a separate target-specific dual T4 preflight is still
required before the long campaign. Existing PointMaze measurements give only
planning bounds: four HSVL plus four HIQL seeds with 25% contingency project to
`129.49` GPU-hours / `$51.80` at the median observed rate or `439.40`
GPU-hours / `$175.76` at the conservative rate. These are not AntMaze quotes;
the slower measured AntMaze method must set the actual reservation.

The HIQL source/config and candidate seed/evaluator manifest are now frozen.
Remaining zero-CPU work is the environment lock, evaluator implementation,
decision rule, and terminal artifact contract. The stale 2/6 public narrative
can also be prepared for an approval-gated 3/6 correction, but neither action
by itself changes claim credit.

## Decision

Do not move PID `41861`, launch another PointMaze probe, or start qIO T4 work
while Flux owns T4. Do not launch a generic CPU experiment. The exact matched
AntMaze-giant source/config protocol and immutable HIQL bundle now exist, but
the dependency/evaluator/terminal-contract gates above still block a preflight.
After those are closed, a tightly capped `cpu-upgrade`
environment/checkpoint/evaluator preflight is useful; the score-seeking work
remains the later matched four-seed T4 campaign under a fresh measured budget
and explicit approval.
