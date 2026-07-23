# AntMaze-giant HIQL source and config readiness - 2026-07-20

## Outcome

The official HIQL source/config provenance for the proposed matched
`antmaze-giant-navigate-v0` route is now preserved paper-locally and bound by
hash. The exact task, official dataset bytes, released HSVL source/config, and
official OGBench HIQL command can be frozen honestly.

The route is still **not launch-ready**. The original four author seed values,
the exact paper-run HSVL bin count, and an exact official HIQL dependency
environment cannot be recovered from the public evidence. No command in this
document authorizes execution.

This was source/readiness work only. It did not run a test suite, train or
evaluate a model, launch or alter a job, signal a process, or publish anything.

## Official HIQL provenance

The upstream source is OGBench tag `v1.2.1`:

- repository: `https://github.com/seohongpark/ogbench.git`
- annotated tag object: `99f9686d4edb18c40ed607ff7d3270d14b27ace2`
- commit: `1d4140997f60c52c6fb0702ec100dc988b18c548`
- tree: `0919ecd9e71ff437ed05875e921c94cf439107be`

The preserved tree contains `LICENSE`, `README.md`, `pyproject.toml`, and the
complete official `impls` subtree: 22 source members in total. Its canonical
member-list digest is
`906ab3bed3a15fa359388f84b961532d6f98bfc2dbcef230e169d3041dd5a0c6`.

Paper-local bindings:

- source manifest: `repro/configs/ogbench_hiql_v1_2_1_source_manifest.json`,
  SHA-256 `a9ed9028701f70f861239760be761332b42dbb7bdcc3c8fa09109aed8fe52b83`
- deterministic source archive:
  `upstream/ogbench-hiql-v1.2.1-source.tar.gz`, 41,424 bytes, SHA-256
  `71e1067d3a584e7b4659f9c4d57142d42e1bed59a657036e83755117e50801e6`
- archive integrity sidecar:
  `upstream/ogbench-hiql-v1.2.1-source.tar.gz.integrity.json`, SHA-256
  `ed286e6aec15b56342eea063fa15d4328b0268ec691da51f9f530ef431379a8a`
- `impls/agents/hiql.py`: SHA-256
  `0cbceb5506e465f2be793dfd69be0244f7990a4d7335afe78c1f74a42b1aa43e`
- `impls/main.py`: SHA-256
  `375fad6375047ff59f3baea408ab7197784c1ec1457588a78a631025e434faee`
- `impls/hyperparameters.sh`: SHA-256
  `633f6e1f5311e42d0bfd6524cda7255120a0835678dcaa1227cfbf3f5cb1d9df`
- `impls/requirements.txt`: SHA-256
  `131375e19086af484cabd3c9a9e0c812f542bcb36e064f7970e3c80f33f796c1`

The official task command in `impls/hyperparameters.sh` is:

```text
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0
```

Its inherited defaults include 1,000,000 training steps, all tasks, and a
1,000,000-step save interval. The exact resolved agent and main settings are
recorded in `repro/configs/antmaze_giant_hsvl_hiql_readiness_v1.json`, SHA-256
`a5078a8cfefbfbf8f3cae0c8f14c29d85110bf296f3b97466263a3b0ad91b7b7`.

## Exact task and data binding

The route is frozen to `antmaze-giant-navigate-v0`, the Table 1 and Appendix
Table 6 task for which the paper reports HSVL `74 +/- 1` and HIQL `65 +/- 5`.

- train data: 241,141,207 bytes, SHA-256
  `387791c414cfba2d77eda64a47f3b6db7950bdf146ae901e4f08c2c2f9e0f3aa`
- validation data: 24,052,175 bytes, SHA-256
  `61d40e91899f496030fb4ee73ffd61c20dad502ab6cc8075744867097930ce71`

The official HSVL source remains bound to commit
`5f13cf22d397be42a87b7d35336db7662879d6db`, tree
`7a38976b0a065d6ef7f755021853119dc2709487`, and source-manifest SHA-256
`e9a279e09fa5952002b8f37a2aa54c7e02a711fcff3f86c362101d6e406131a6`.

## What can be frozen honestly

| Item | Status | Honest interpretation |
| --- | --- | --- |
| Environment/task | Frozen | Exactly `antmaze-giant-navigate-v0` |
| Train and validation bytes | Frozen | Exact local paths, sizes, URLs, and SHA-256 identities recorded |
| Official HIQL source | Frozen | Annotated tag, commit, tree, Git blobs, local source files, manifest, and deterministic archive bound |
| Official HIQL task command | Frozen | Exact upstream command plus inherited defaults recorded |
| Released HSVL source/config | Frozen | Exact commit, source manifest, YAML hashes, and released command recorded |
| Candidate matched training seeds | Preregistered, not recovered | `[0, 1, 2, 3]` may be used identically for both methods only as new reproduction seeds |
| Original four author seeds | Not recoverable | Paper and released repositories give a count but not the four identities |
| Exact paper-run HSVL `K` | Not recoverable | Appendix says `K=500`; released YAML requests 800 edges and released construction yielded 499 unique intervals in the prior static audit |
| Exact official HIQL environment | Not recoverable | Requirements use unpinned/range dependencies and no lockfile is released |

The candidate `[0, 1, 2, 3]` values must never be described as the authors'
seeds. They are a transparent new matched-seed choice for a future
reproduction. The proposed reset/actor episode seed formulas are likewise a
new deterministic evaluation contract and are not part of the authors'
released protocol.

## Remaining launch gates

1. Choose and disclose either the released HSVL 800-request/499-interval route
   or another justified resolution of the paper's `K=500` discrepancy.
2. Produce a fully resolved, hashed HIQL dependency lock. It must be labeled a
   reproduction-environment choice rather than recovered author provenance.
3. Implement a common evaluator wrapper that makes reset and actor seeds
   explicit while preserving official task and success semantics.
4. Bind matched resume/checkpoint, terminal importer, failure retention, and
   independent audit contracts for HSVL and HIQL.
5. Under separate authorization, run bounded target-specific CPU compatibility
   and dual-GPU throughput/resume preflights.
6. Freeze a symmetric C3 decision rule and recheck the shared accelerator
   budget before observing any paper-scale result.

Until all six gates are closed, the correct state is
`source_and_candidate_protocol_pinned_not_launch_ready`.
