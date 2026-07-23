# C1/C3 claim-repair route audit

Date: 2026-07-20

This audit is bounded to the three anchored claims used by the official judge. It does not add
claims, publish results, launch a long experiment, or treat an execution preflight as scientific
claim evidence.

## Current judge contract and exact gaps

The fresh official verdict at Space SHA
`757e63d6377d6a3789657d75e354306c95d6e43f` is `3/6`:

| Claim | Verdict | Evidence accepted | Gap preventing full credit |
|---|---|---|---|
| C1: time-to-goal is modeled as a probability distribution | `toy` | Official PointMaze-large data; three seeds; PCS held-out censored NLL `2.398 +/- 0.041` versus repeated-hazard `3.158 +/- 0.006`; normalization error `1.34e-15`; zero monotonicity violations | Width 64 rather than 512, 2,000 rather than 1,000,000 updates, 53 rather than 500/800 bins, one head rather than the released twin ensemble, and no actor/policy evaluation |
| C2: discounted survival identity | `verified` | Exact computation and negative controls | None |
| C3: HSVL matches or surpasses hierarchical TD baselines on offline GCRL benchmarks and excels on long-horizon tasks | `inconclusive` | No performance evidence accepted | No trained hierarchical actors, HIQL comparison, reported long-horizon task, four-seed success rate, or policy evaluation |

The C1 real-data evidence remains valid and independently auditable. Its summary SHA-256 is
`49e48feb0eb1cf9ed5304ee0ce3f6cc3c7f83607e8b8a8faff5be5ed4d2d0435`; the independent
audit SHA-256 is `ef99588b060aa3631be80c0479cc8f621742fa1eb4d1f72ed15bed32f8dc2ca2`.

## The minimum exact-width HSVL preflight is already complete

HF Job `DineshAI/6a5dcda9bee6ee1cf4ed2148` completed on `t4-small` in 171 running
seconds. The local evidence is under `outputs/hf/qio-t4-preflight-v5-20260720/`; the job URL is
`https://huggingface.co/jobs/DineshAI/6a5dcda9bee6ee1cf4ed2148`.

The run used the official PointMaze-large input and exact released full-agent dimensions, performed
a fresh 0-to-10 CUDA stage followed by a separate-process 10-to-100 resume, and reloaded the final
checkpoint exactly. It observed 9,778,698 parameters:

- `modules_goal_rep`: 662,272
- `modules_high_actor`: 1,712,896
- `modules_low_actor`: 1,712,642
- `modules_value`: 5,690,888

The original packager looked for unprefixed Flax keys and incorrectly recorded both actors as
absent. The recorded normalizer changed only those two inventory booleans after confirming the
`modules_high_actor` and `modules_low_actor` keys and finite actor metrics; it did not change source,
checkpoints, protocol, or scientific values.

Observed feasibility evidence:

- peak GPU allocation: 11,459 MiB of 15,360 MiB (`74.6029%`)
- peak process-tree RSS: 1,897,746,432 bytes
- median steady throughput: `21.451733` updates/s
- conservative observed throughput: `6.321757` updates/s
- exact checkpoint reload: true
- final checkpoint SHA-256: `689ed10245e4926249316c79f7502ba692343c583e6fdfcc0de458020f606925`
- restored state SHA-256: `7bc393ffaac206b7deaa64de5e67634c4c0556aed1d0ff43e5dc258fbc25d9a0`

Primary local integrity anchors:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `preflight_report.json` | - | `9ccf813d81403f300908aff319b7c8f8142b19148c29c39e959d32b3ea61b6f8` |
| `preflight_integrity.json` | - | `ce681ac991e2c732f96f703aa83222b5aee5821ba87ac4d20ea94c692792f3a0` |
| `full_agent_probe/seed-0/COMPLETE.json` | - | `88ebda1a1c2df8570cc96cf6c04efa4c0cb24e9981af0cf2f1a5b2016e1f74b6` |
| `qio-full-agent-t4-return-v1.tar.gz` | 96,720,938 | `25911c0bc00a4200f3a1664d0d70c7a7b7dadf397dafcc7e2d36eaf40d7d41fa` |
| return integrity sidecar | 290 | `7b19f8cf80cced68e2fbc91235f53d603f1c1e176455515e5cd3e870dd551ab3` |
| training trace | - | `2433a3a6e12cea7a0731df5e559a805116b430ed62c848ed3e8abc64338c5e2d` |
| protocol | - | `9c54d123aa361d8d6fcfe77a1049445490b89a67ae2749a36e7bf460ac0b11ea` |

This closes the minimum HSVL architecture, memory, checkpoint, and resume feasibility gate. It does
not repair C1 or C3: it is only one seed, 100 updates, PointMaze, and has no policy evaluation.

## Source and artifact audit: no checkpoint shortcut

All 18 vendored executable/configuration files match official Git blobs at HSVL commit
`5f13cf22d397be42a87b7d35336db7662879d6db`; the official tree is
`7a38976b0a065d6ef7f755021853119dc2709487`. The repository has no tag, release, result bundle,
or model checkpoint. The paper's source YAML requests 800 logarithmic edges, which de-duplicate to
499 intervals, while Appendix Table 3 says `K=500`; this difference must remain disclosed.

The HSVL lockfile resolves `ogbench==1.2.1`. The official OGBench `v1.2.1` source commit is
`1d4140997f60c52c6fb0702ec100dc988b18c548`, and it contains the reference HIQL implementation
and exact per-task commands. Its GitHub release has no assets and the source tree contains no HIQL
checkpoint. Therefore neither C1 nor C3 can honestly be upgraded by downloading author weights.

## Materially distinct honest repair routes

### Route A: one shared full campaign for C1 and C3 (recommended scientific route)

Use one reported long-horizon Table 1 task and run exact HSVL and the official matched HIQL
comparator for four independent seeds. Evaluate every trained policy for 50 episodes per task. On
the trained HSVL checkpoints, additionally audit the twin survival distribution on held-out
censored examples: probability normalization, survival monotonicity, held-out NLL/calibration, and
a temporal negative control.

`humanoidmaze-giant-navigate-v0` is the most decisive target because the paper reports HSVL
`81 +/- 1` versus HIQL `12 +/- 4`. It directly exercises the complex long-horizon part of C3 and
uses `discount=0.999`, `subgoal_steps=100`, and `num_log_bins=500` in the released HSVL command.
The large reported margin lowers outcome risk, but its larger observation/data path means the
PointMaze throughput estimate cannot be assumed to transfer.

This is the only route that can close both judge criticisms with one coherent evidence set. Success
and failure must both be reported: a reproduced advantage supports C3, while no advantage is valid
falsification evidence rather than a reason to hide or restart the campaign.

### Route B: C1-only exact-scale distribution repair

Run the exact released full HSVL architecture for four seeds on official PointMaze-large for
1,000,000 updates, then add held-out censored-distribution audits and 50-episode policy evaluation.
This directly removes every scale/component item named in the C1 verdict while reusing the verified
data and preflight path. It does not address the reported long-horizon benchmark or matched HIQL gap,
so C3 should remain inconclusive.

This route is materially cheaper than Route A because it needs four HSVL seeds and no comparator,
but it risks the judge continuing to view PointMaze as outside the paper's headline benchmark set.

### Route C: direct falsification-first C3 campaign

Preregister the same Route A task, configs, four seeds, and 50-episode evaluator, but define the
decision rule symmetrically: report C3 as unsupported on the selected task if HSVL's seed-level
success distribution does not match or exceed HIQL within the preregistered uncertainty analysis.
This is not a cheaper run, but it is materially distinct in interpretation and prevents adaptive
task/seed selection after seeing results.

A Table 2 flat-critic reproduction with the same DDPG+BC actor and CRL comparator is a useful
mechanism audit and could falsify a narrower survival-critic advantage. It is not sufficient by
itself for anchored C3, which explicitly concerns the hierarchical TD comparison and long-horizon
benchmark performance.

### Routes that are not honest upgrades

- Repeating the 100-update HSVL preflight or doing a cosmetic width/step bump.
- Treating the paper's own table as independent reproduction evidence.
- Treating absence of checkpoints as falsification.
- Using source inspection alone to upgrade empirical C3.
- Running only HSVL and comparing it post hoc to a published HIQL mean when the judge explicitly
  identified the missing matched comparator.
- Selecting whichever task or seed looks best after observing results.

## Cost and time from the completed T4 measurement

At the recorded HF T4-small rate of `$0.40/hour`, PointMaze extrapolations are:

| Scope | Median-rate projection | Conservative observed-rate projection |
|---|---:|---:|
| One 1M-update HSVL seed | 12.95 h / $5.18 | 43.94 h / $17.58 |
| Four HSVL seeds | 51.80 h / $20.72 | 175.76 h / $70.30 |
| Four HSVL + four HIQL seeds, plus 25% setup/evaluation contingency | 129.49 h / $51.80 | 439.40 h / $175.76 |

The job recorded `$39.49` remaining at submission. Even the median matched-campaign projection does
not fit that snapshot, and the conservative projection is much larger. These are planning bounds,
not a quote: HIQL and HumanoidMaze throughput are unmeasured. A long run must not be launched until
the shared ledger is rechecked and target-specific measurements replace the assumptions.

## Minimum measured preflight still required before a long run

No further generic HSVL exact-width preflight is needed. Before Route A, one bounded target-specific
dual preflight must establish all of the following:

1. Pin and hash the chosen official train/validation datasets, HSVL commit, OGBench/HIQL commit, both
   fully resolved configs, seeds, and evaluator protocol.
2. On the intended accelerator, run exact HSVL from 0 to 10 updates and resume in a separate process
   to 100 updates on the chosen target. Record finite critic and both actor metrics, peak GPU/RAM,
   steady throughput, immutable checkpoint hashes, and exact reload.
3. Run official HIQL on the same target/data from 0 to 10 and separate-process resume to 100. Record
   the same feasibility, throughput, and checkpoint evidence. The existing HSVL preflight does not
   cover this comparator.
4. Execute the real MuJoCo evaluator for at least one bounded episode per task for both untrained or
   100-update agents under headless EGL. This is an environment-compatibility check only, not claim
   evidence. Confirm that success keys, task enumeration, episode termination, and output persistence
   work before paying for long training.
5. Recalculate the full four-plus-four-seed campaign from the slower measured method/target rate and
   actual evaluation duration. Require peak GPU memory below 95%, exact resume, finite metrics, and a
   total projection that fits the centrally verified budget with contingency.

Only after all five gates pass should the long jobs be scheduled. The preflight should persist its
artifacts even on failure; failure changes the plan but must not terminate the research campaign.

## Judge-ready next-action sequence

1. Record the completed HF preflight as feasibility-only; do not submit it as new claim evidence.
2. Select and preregister one Table 1 target. Prefer HumanoidMaze-giant for decisiveness, subject to
   the dual-preflight memory and budget gates.
3. Prepare the official HIQL comparator bundle pinned to OGBench commit
   `1d4140997f60c52c6fb0702ec100dc988b18c548`, alongside the already pinned HSVL source.
4. Run only the bounded dual preflight and evaluator smoke described above; preserve every failure.
5. Recheck the central ledger. If the four-plus-four plan does not fit, use Route B for a C1-only
   repair or acquire additional accelerator budget; do not reduce seeds/updates silently.
6. If funded, run all preregistered seeds with immutable checkpoints and no adaptive task selection.
7. Independently reconstruct success statistics and the HSVL held-out distribution diagnostics from
   raw artifacts, including negative controls and seed-level uncertainty.
8. Report the result whether it verifies or falsifies the claims. Only then update the public
   logbook and request a fresh judge verdict.

The persistent local queue at PID `41861` is outside this audit's control. It remains untouched and
must not be stopped merely because the HF feasibility gate completed first.
