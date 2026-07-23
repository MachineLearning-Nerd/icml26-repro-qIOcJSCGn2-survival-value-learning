# Status

- OpenReview ID: `qIOcJSCGn2`
- Public Space: `https://huggingface.co/spaces/DineshAI/qIOcJSCGn2`
- Repository HEAD before this uncommitted attempt: `37c43b05889e7aeda07c789db9ec38166161bc2f`

## Local evidence state

- **C1:** applied bounded verification completed on official OGBench PointMaze-large with the
  released PCS distribution path. Three seeds, disjoint validation, geometric baseline, shuffled
  target control, independent NumPy audit, and raw artifacts are complete. See
  `docs/CLAIM1_OGBENCH_DISTRIBUTION_AUDIT.md`.
- **C2:** prior exact-identity verification is preserved; all five original tests still pass.
- **C3:** untouched and not claimed.

## Verification and publication state

- Prior local tests: `15 passed`; no automated test suite was run for this research-only audit.
- Standalone artifact audit: `all_checks_pass: true`.
- Independent raw-output reconstruction reproduced every stored NLL exactly; all raw hashes,
  normalization, and monotonicity checks pass.
- Published Space SHA: `757e63d6377d6a3789657d75e354306c95d6e43f` (running, canonical six-page
  logbook, required challenge tags, rendered index/poster HTTP 200).
- Every canonical public logbook/static file matches the local source byte-for-byte. Three obsolete
  remote pages were removed explicitly; the public page tree contains only the six canonical pages.
- Public artifact: `reproduction-svl-pointmaze/repro-bundle:v2`, 39 files, 21,548,314 bytes,
  manifest digest `83b7a343906bb25cb9cb389db9ad29175f4a8e90ae58935857212829bf79fff5`.
  The downloaded public database and sampled summary/raw-prediction blobs match the local hashes.
- Official challenge validator passes under the current canonical naming profile. The historical
  Space slug is retained to preserve the judged submission.
- Fresh official verdict at `2026-07-19T17:01:55+00:00`: **3/6** at exact Space SHA
  `757e63d6377d6a3789657d75e354306c95d6e43f`, with C1 `toy`, C2 `verified`, and
  C3 `inconclusive` (overall quality: medium).
- The judge accepted the real OGBench distribution result, baseline gap, normalization, and
  monotonicity evidence. C1 remained `toy` specifically because the run used width 64 vs 512,
  2,000 vs 1,000,000 updates, 53 vs 500/800 bins, and one head vs the twin ensemble, with no
  actor or policy evaluation.
- Attempt 2 CPU feasibility gate (not a scientific attempt): the exact released twin critic has
  6,353,160 parameters and requested 800 bins de-duplicate to 499. A two-update synthetic
  batch-1024 CPU probe used about 787 MiB RSS and 0.772 seconds per steady update. That projects to
  about 8.9 days per 1,000,000-update seed or 26.8 days for three seeds before actors, evaluation,
  and a baseline. No official checkpoint or compatible faithful Metal path is available.
- State: **exact-width T4 execution feasibility passed; full scientific campaign remains unexecuted**.
  HF Job `DineshAI/6a5dcda9bee6ee1cf4ed2148` completed on `t4-small` in 171 running seconds.
  The exact released full agent performed a fresh 0-to-10 CUDA stage and a separate-process
  10-to-100 resume on official PointMaze-large data, reloaded the final checkpoint exactly, and
  observed all 9,778,698 parameters: goal representation 662,272, high actor 1,712,896, low actor
  1,712,642, and twin survival value modules 5,690,888. Peak GPU memory was 11,459 MiB of 15,360
  MiB (`74.6029%`); median steady throughput was `21.451733` updates/s and the conservative observed
  rate was `6.321757` updates/s. Evidence is in
  `outputs/hf/qio-t4-preflight-v5-20260720/`; `preflight_report.json` has SHA-256
  `9ccf813d81403f300908aff319b7c8f8142b19148c29c39e959d32b3ea61b6f8` and the returned archive
  has SHA-256 `25911c0bc00a4200f3a1664d0d70c7a7b7dadf397dafcc7e2d36eaf40d7d41fa`.
  This is execution/checkpoint evidence only: 100 updates, one seed, and no policy evaluation cannot
  upgrade C1 or C3.
- The persistent local queue session `19438` / PID `41861` remains alive and untouched. It still
  waits by native process inspection on the OAM-to-CVM queue PID `41276`; the completed HF preflight
  does not authorize stopping, signaling, or reprioritizing it.
- CPU-only protocol/provenance audit completed before the queued gate. All 18 vendored executable
  and configuration files match the official Git blobs at commit
  `5f13cf22d397be42a87b7d35336db7662879d6db`; the remote still has no newer commit, tag, release,
  or checkpoint. Both official PointMaze inputs match their SHA-256 identities. The standalone
  artifact `outputs/full_agent_protocol_audit.json` reports `all_checks_pass: true`, and the future
  checkpoint protocol is now bound to the verified source-manifest digest plus the local wrapper and
  auditor hashes. Completion validation rehashes the protocol, checkpoint manifest, active
  checkpoint, and exact restored state rather than trusting status fields alone. See
  `docs/FULL_AGENT_PROTOCOL_AUDIT.md`. This audit is intentionally not presented as claim evidence.
- Paper-scale C3 remains unchanged: one reported Table 1 task, 1,000,000 maze updates, four seeds,
  50 evaluation episodes per task, and a matched hierarchical-TD comparator (or exact reported
  target) are still required. Before that campaign, the chosen task's exact HSVL and HIQL paths
  still need a bounded dual throughput/resume preflight and a real headless-policy-evaluator smoke.
  The source YAML's 800 requested edges (499 unique intervals) versus Appendix Table 3's `K=500`
  is explicitly recorded rather than silently conflated.
- The separate CUDA handoff remains available as a Colab fallback without changing or replacing the
  waiting local queue. The same deterministic 22,528,563-byte upload archive was also used for the
  successful HF T4 preflight and is
  `outputs/colab_handoff/qio-full-agent-colab-input-v1.tar.gz` (SHA-256
  `d7f8e07b72f828d8f39100533510a60858dab8b7cc7debcc2e2409260da634ee`). It binds
  the official commit, all 18 Git blobs, both exact datasets, author lockfile, runner/auditor
  sources, and a fail-closed GPU protocol. The prescribed 10-update first invocation followed by a
  separate resume to 100 updates gives observed cross-process resume evidence. A deterministic
  return packer and JAX-free local verifier bind the active checkpoint, raw finite metrics,
  command/environment/GPU ledger, session chain, protocol, and archive hash. See
  `docs/COLAB_FULL_AGENT_HANDOFF.md`.
- The bounded repair audit and the remaining target-specific HSVL+HIQL preflight gates are recorded
  in `docs/CLAIM_REPAIR_ROUTE_AUDIT.md`. At the measured PointMaze rates, a four-HSVL plus four-HIQL
  campaign with 25% contingency projects to `$51.80` at the median rate or `$175.76` at the
  conservative observed rate, versus `$39.49` recorded remaining at preflight submission. No long
  run should launch until target-specific dual throughput/evaluation checks pass and the shared
  budget ledger is rechecked.
- A materially distinct source/paper/artifact archaeology pass is recorded in
  `docs/APPROACH2_SOURCE_ARTIFACT_ARCHAEOLOGY.md` and
  `outputs/approach2_source_artifact_audit.json`. It found no released HSVL or HIQL checkpoint,
  release asset, Actions artifact, independent result table, or evaluator-only C3 shortcut. The
  existing exact checkpoint is only update 100 of 1,000,000 with no policy evaluation, so it remains
  ineligible as claim evidence. The only bounded candidate found is to resume that exact seed-0
  PointMaze-large checkpoint to paper scale and perform the full policy/distribution audit for C1
  only: about `$5.18` median or `$17.57` at the conservative observed T4 rate, before a mandatory
  live budget recheck. This audit did not authorize or launch that continuation and leaves C3
  unchanged.
- The distinct AntMaze-giant route now has paper-local official HIQL
  source/config provenance: OGBench annotated tag `v1.2.1`, commit
  `1d4140997f60c52c6fb0702ec100dc988b18c548`, tree
  `0919ecd9e71ff437ed05875e921c94cf439107be`, 22 source members, and a
  deterministic 41,424-byte archive with SHA-256
  `71e1067d3a584e7b4659f9c4d57142d42e1bed59a657036e83755117e50801e6`.
  The matched task/config protocol is pinned in
  `repro/configs/antmaze_giant_hsvl_hiql_readiness_v1.json`. Candidate seeds
  `[0, 1, 2, 3]` are explicitly new reproduction seeds because the paper does
  not identify its original four values. This remains readiness work only:
  HSVL's paper-versus-release `K` discrepancy, the unpinned official HIQL
  environment, evaluator wrapper, terminal contracts, and later bounded
  preflights still block launch. See
  `docs/ANTMAZE_GIANT_HIQL_SOURCE_CONFIG_READINESS_20260720.md`.
- No GitHub commit or push was performed.
