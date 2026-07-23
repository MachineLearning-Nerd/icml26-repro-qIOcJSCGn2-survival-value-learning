# C1/C3 non-duplication and readiness audit

Date: 2026-07-20

This is a static, paper-local audit. It did not run training, evaluation, a test suite, an HF query,
or an HF mutation. It did not signal or modify any local process. No result is authorized for
publication by this document.

## Protected queue boundary

At inspection time, PID `41861` was alive and waiting as
`run_full_agent_probe_queue.py --wait-pid 41276 ...`. It had no child process. Its queued scientific
command is the existing exact-width 100-update full-agent probe. That work remains exclusively owned
by the existing queue and must not be duplicated, stopped, or reprioritized.

PID `41861` does **not** itself own the proposed update-100-to-1,000,000 C1 continuation or a matched
C3 campaign. Nevertheless, the frozen C1 contract deliberately refuses launch while PID `41861` or
another matching qIO process exists. The correct action is to wait for the existing queue to finish
naturally, then repeat all process, central-ledger, and live account gates.

## Claim decision matrix

| Route | What it could change | Scientific coverage | Present status |
|---|---|---|---|
| Exact C1 seed-0 continuation | Could strengthen C1 beyond `toy`, or produce negative/inconclusive evidence | One exact-scale PointMaze seed plus the existing three-seed reduced-scale distribution evidence | Components frozen; not launch-authorized |
| Four-seed exact PointMaze C1 | Strongest direct response to every C1 scale criticism | C1 only | Not bundled or funded |
| Four HSVL + four HIQL HumanoidMaze-giant seeds | Strongest joint C1/C3 route | Exact survival diagnostics and a matched reported long-horizon policy comparison | Not target-preflighted, data-pinned, or funded |
| Existing queued 100-update probe | Execution feasibility only | Neither C1 nor C3 claim repair | Already queued under PID `41861` |

A C1-only exact seed-0 execution can **honestly be worth doing** after its gates pass. It removes the
judge-cited width, update-count, effective-bin, twin-head, actor, and policy-evaluation gaps for one
seed and complements the already accepted three-seed reduced-scale PointMaze evidence. It cannot
guarantee a score increase: one exact seed is not a paper-scale multi-seed reproduction, PointMaze is
not the headline Table 1 comparison, and a failed scientific gate must remain negative or
inconclusive evidence. It cannot change C3.

## C1 exact seed-0 readiness

The bounded route resumes the immutable update-100 checkpoint to update 1,000,000 on
`pointmaze-large-navigate-v0`. It uses one `t4-small` with a 48-hour timeout and persistent mounted
storage.

Exact source/data/lineage needs already pinned:

- HSVL commit `5f13cf22d397be42a87b7d35336db7662879d6db`, tree
  `7a38976b0a065d6ef7f755021853119dc2709487`.
- OGBench commit `1d4140997f60c52c6fb0702ec100dc988b18c548`.
- PointMaze train data: 20,398,507 bytes,
  `82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61`.
- PointMaze validation data: 2,040,807 bytes,
  `19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4`.
- Update-100 checkpoint: 117,362,957 bytes,
  `689ed10245e4926249316c79f7502ba692343c583e6fdfcc0de458020f606925`.
- Restored-state identity:
  `7bc393ffaac206b7deaa64de5e67634c4c0556aed1d0ff43e5dc258fbc25d9a0`.

The completed T4 preflight measured 9,778,698 parameters, 11,459 MiB peak allocation out of
15,360 MiB, `21.451733` median updates/s, and `6.3217566` conservative observed updates/s. At the
recorded `$0.40/hour` rate, the remaining 999,900 updates project to:

| Bound | Time | Compute cost |
|---|---:|---:|
| Median observed rate | 12.9477 h | $5.1791 |
| Conservative observed rate | 43.9356 h | $17.5742 |
| Hard reservation ceiling | 48 h | $19.20 |

The runner, independent NumPy auditor, terminal verifier, source manifest, contract, and fail-closed
gate checker exist. The current frozen component identities are:

- runner: `90ea2ee78cd918d1e39cd2659ce7eeac09e433543a38f8c0a61dab36d8c4fce0`;
- auditor: `12274e0c550ce0795585c120e6be37ec6a4fb47b6818ba9b86dc97ba606e4198`;
- terminal verifier: `58c11f186c0075d9d1cc072b5b1a0212e372ebf0d941202e634f766c2ac92c4f`;
- immutable job source manifest:
  `8638e6fffb347a1998772d0771e291a41b6d7ebe1c67b0145098688ddd2ea851`;
- read-only gate checker:
  `79e696793a9aac9ab719a9d6cdfa8ae94928e6394fd54ad02e5a36d5f8b8405e`.

The terminal verifier independently reconstructs the 50-episode policy summary from raw episode
rows and recomputes survival evidence from raw held-out arrays. This is sufficient static evidence
for the compute payload and return validation, but not for submission.

### Missing operational pieces and live gates

- PID `41861` must exit naturally and no matching qIO job may exist.
- The central controller must create the exact `$19.20` reservation; this paper must not edit the
  shared ledger itself.
- A fresh immediate pre-submit account-wide `RUNNING` and `SCHEDULING` T4 check must pass.
- No submission wrapper currently closes the live-gate/submission race.
- No paper-local terminal-job plus persistent-bucket retriever/import command is frozen.
- No approval-gated existing-Space repair is prepared, and publication is outside this contract.

These are real readiness gaps. The runner alone must not be treated as a launch command.

## Strongest C3 route and exact unresolved needs

The strongest joint route is `humanoidmaze-giant-navigate-v0`: four exact HSVL seeds and four
official matched HIQL seeds, each trained for 1,000,000 updates and evaluated for 50 episodes under
the same preregistered protocol. The paper reports HSVL `81 +/- 1` versus HIQL `12 +/- 4`, making it
the most decisive reported long-horizon target. The released HSVL command requires
`discount=0.999`, `subgoal_steps=100`, and `num_log_bins=500`; batch size is 1024, width 512, critic
depth 3, actor depth 6, and subgoal representation dimension 256.

For HSVL, the held-out validation path must additionally record both survival heads, their mean,
normalization, monotonicity, censored NLL/calibration, twin disagreement, target-shuffle control,
and temporal-reversal control. All seeds and all failures must be reported.

This campaign is **not statically ready**. Before its cost can be called exact, it needs:

1. the official HumanoidMaze-giant train and validation files downloaded once, with exact byte sizes
   and SHA-256 identities pinned;
2. exact fully resolved HSVL and HIQL configurations plus evaluator and seed manifests;
3. target-specific HSVL 0-to-10 and separate-process 10-to-100 resume measurement;
4. the same measurement for official HIQL;
5. at least one bounded headless-EGL environment episode per method;
6. measured peak GPU/RAM, throughput, reload, evaluation duration, and failure persistence for both
   methods on the intended accelerator.

The current PointMaze-only extrapolation at `$0.40/hour`, including 25% contingency, is 129.49 GPU
hours / `$51.80` at the median rate or 439.40 GPU hours / `$175.76` at the conservative rate. These
figures are planning bounds, **not exact C3 cost requirements**: HumanoidMaze and HIQL throughput and
memory are unmeasured. A T4 must not be assumed to fit until the dual preflight passes. The prior
`$39.49` account snapshot is stale and below even the median projection; only a fresh central-budget
decision can authorize the campaign.

The locally present AntMaze-giant files are a fallback target and are already identifiable:

- train: 241,141,207 bytes,
  `387791c414cfba2d77eda64a47f3b6db7950bdf146ae901e4f08c2c2f9e0f3aa`;
- validation: 24,052,175 bytes,
  `61d40e91899f496030fb4ee73ffd61c20dad502ab6cc8075744867097930ce71`.

AntMaze-giant has lower data-preparation friction, but its reported gap (`74 +/- 1` versus
`65 +/- 5`) is much smaller. It is therefore not the strongest C3 evidence route unless the
HumanoidMaze dual preflight fails memory, runtime, environment, or budget gates.

## Safe next decision

Do no qIO compute while PID `41861` remains. After it exits naturally, refresh the shared process,
ledger, and account state. If exactly one C1 reservation up to `$19.20` is available, completing the
frozen seed-0 C1 path is the strongest bounded non-duplicative action. Treat any judge improvement
as possible, never guaranteed.

Do not schedule C3 from the PointMaze estimates. First fund and run only the dual target preflight;
then derive an exact four-plus-four campaign reservation from the slower measured method. If that
reservation cannot be funded, preserve C3 as `inconclusive` rather than weakening the registered
seed, update, comparator, or evaluation requirements.
