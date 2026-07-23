# Approach 2: source, paper, and artifact archaeology

Date: 2026-07-20

This is a read-only search for a cheaper, honest C1 or C3 claim-repair route after the full
four-HSVL plus four-HIQL campaign exceeded the recorded budget snapshot. It inspected only the
pinned official paper/source and existing local/official artifacts. It did not run a test suite,
launch or modify a job, touch PID `41861`, or publish anything.

The machine-readable record is `outputs/approach2_source_artifact_audit.json`.

## Outcome

No existing artifact can repair C1 or C3. There is no released HSVL checkpoint, HIQL checkpoint,
independent result bundle, or evaluator-only shortcut. The local exact checkpoint is a useful
resume point, but at 100 of 1,000,000 updates (`0.01%`) and with no policy evaluation it remains an
execution gate, not scientific evidence.

One bounded route fits the recorded budget snapshot: resume that exact seed-0 PointMaze-large
checkpoint to 1,000,000 updates and run a 50-episode deterministic policy evaluation plus a
held-out twin-distribution audit. This can seek a C1 upgrade only. It cannot repair C3, cannot
replace independent full-scale seeds, and is not authorized or counted as evidence by this audit.

## What the official artifacts contain

### Paper

The official arXiv preprint is preserved as `docs/official_paper.pdf`, SHA-256
`f468cfd9e6ea9e06e185a01e3bf81bc703565d55fa7db98c0b4211499c05ec72` (20 pages).
The relevant tables were also visually checked after rendering.

- Table 1 reports means and standard deviations over four seeds. The clearest long-horizon gaps are
  AntMaze-giant navigate, HSVL `74 +/- 1` versus HIQL `65 +/- 5`, and HumanoidMaze-giant navigate,
  HSVL `81 +/- 1` versus HIQL `12 +/- 4`.
- Table 2 is a controlled flat-critic comparison: SVL and CRL use the same DDPG+BC actor. That is a
  valuable mechanism ablation, but it is not anchored C3, which concerns hierarchical HSVL versus
  hierarchical TD/HIQL on long-horizon tasks.
- Table 3 requires 1,000,000 maze gradient steps, batch size 1024, width 512, critic depth 3, actor
  depth 6, and 50 evaluation episodes in the released config. The paper reports four seeds.
- Every table is an author result. Reusing a paper number would not produce independent
  reproduction evidence.

### HSVL source

The official repository still resolves to main commit
`5f13cf22d397be42a87b7d35336db7662879d6db`, tree
`7a38976b0a065d6ef7f755021853119dc2709487`. On 2026-07-20 the public repository had zero tags,
zero releases, zero GitHub Actions artifacts, and no checkpoint/result file in its tree. The project
page links the paper and this source repository, but no checkpoint, result archive, or W&B run.

All 18 vendored executable/config files match the pinned Git blobs. The source-manifest SHA-256 is
`e9a279e09fa5952002b8f37a2aa54c7e02a711fcff3f86c362101d6e406131a6`.

The source uses W&B only for metric logging. It declares project `hsvl` and group `rebutal`, but no
entity, run ID, artifact upload, or checkpoint upload. `save_agent` writes
`outputs/hsvl_rebutal/<run>/params_<step>.pkl` locally. The source contains no checkpoint load or
restore route. Thus even a public metrics run would not imply reusable weights.

Relevant source hashes:

| File | SHA-256 |
|---|---|
| `upstream/train.py` | `397dada4a1f6c106954f1387478ac4fc5b211093984067c03ee722c2fc9980c5` |
| `upstream/hsvl/config/main.yaml` | `0d584ef55e8ddc6c0b4ca19aa45b6571913b01c6bcbd56dc17fd4f71ee16ae65` |
| `upstream/hsvl/utils/common.py` | `3d89f3283b2dff6d058eb64991085be78ae6aa71cd14596465af49e18dffc5f4` |
| `upstream/hsvl/utils/flax_utils.py` | `b4b98e14e6dd8a7950ed8d353d3c1df648d35be51d5036f67ac67286fb7eeb9c` |

### HIQL/OGBench source

The HSVL lock resolves OGBench `1.2.1` at commit
`1d4140997f60c52c6fb0702ec100dc988b18c548`. Its reference HIQL implementation is source only.
The pinned tree has no checkpoint/result file, and releases `v1.2.1`, `v1.2.0`, and `v1.1.5` each
have zero assets. There is therefore no official matched HIQL checkpoint to evaluate cheaply.

## What can and cannot be inferred from the local checkpoint

The exact local checkpoint is
`outputs/hf/qio-t4-preflight-v5-20260720/full_agent_probe/seed-0/checkpoint-00000100.pkl`,
117,362,957 bytes, SHA-256
`689ed10245e4926249316c79f7502ba692343c583e6fdfcc0de458020f606925`. Its restored-state hash is
`7bc393ffaac206b7deaa64de5e67634c4c0556aed1d0ff43e5dc258fbc25d9a0`.

It proves exact resume/reload of the full 9,778,698-parameter architecture, including goal
representation, high and low actors, and twin critic. It does not prove distribution quality or
policy success because it is one seed, PointMaze-large, only 100 updates, and was not policy
evaluated. Deterministically evaluating it now would only characterize an undertrained preflight;
it could neither repair C1 nor honestly falsify the 1M-update/four-seed C3 claim.

The returned archive is 96,720,938 bytes, SHA-256
`25911c0bc00a4200f3a1664d0d70c7a7b7dadf397dafcc7e2d36eaf40d7d41fa`.

## Bounded candidate under the recorded cap: C1 only

Resume the exact seed-0 PointMaze-large checkpoint from update 100 to 1,000,000. At the measured
T4-small rate of `$0.40/hour`, the remaining 999,900 updates project to:

| Bound | Throughput | Time | Cost |
|---|---:|---:|---:|
| Median measured | 21.451733 updates/s | 12.95 h | $5.18 |
| Conservative observed | 6.321757 updates/s | 43.94 h | $17.57 |

Both fit the `$39.49` remaining-budget snapshot recorded when the preflight was submitted. That
snapshot may be stale, so the central budget and active-job ledger must be rechecked before any
launch. The existing queue and PID `41861` remain outside this route and must not be changed.

The final checkpoint would need all of the following before it could be submitted as C1 evidence:

1. Exact final checkpoint reload and immutable source/config/data/environment hashes.
2. Fifty deterministic policy-evaluation episodes with raw episode outcomes.
3. Held-out censored NLL and calibration for both survival heads.
4. Probability normalization and survival-monotonicity reconstruction.
5. Temporal and shuffled-target negative controls.
6. Clear combination with, but no substitution for, the existing three-seed reduced-scale audit.

If completed, this removes the judge-cited width, update-count, bin-count, twin-head, actor, and
policy-evaluation gaps for one exact-scale seed. It still has two explicit limitations: one
independent exact-scale seed and a non-headline PointMaze target. A judge upgrade is therefore
possible, not guaranteed.

## Fail-closed conclusion

- C1 is not repaired by anything found in this audit.
- C3 is not repaired or falsified by anything found in this audit.
- Paper tables are not independent evidence.
- The 100-update checkpoint is not claim evidence.
- The Table 2 flat-critic route does not answer anchored C3.
- No honest C3 repair under the recorded cap was identified without new matched HSVL/HIQL training
  evidence.
- The only bounded candidate is a one-seed exact-scale C1 continuation, which remains non-evidence
  until training and every preregistered audit complete.

No job or process was launched, stopped, signaled, reprioritized, or modified. No tests were run and
no publication was changed.
