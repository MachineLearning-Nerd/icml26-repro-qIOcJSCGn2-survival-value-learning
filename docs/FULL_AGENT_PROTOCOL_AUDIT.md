# Full-agent execution-gate protocol audit

## Outcome

The queued 100-update PointMaze run is now **fail-closed against the exact official source tree**.
All 18 executable/configuration files vendored from the authors' repository match the Git blob IDs
at commit `5f13cf22d397be42a87b7d35336db7662879d6db`; both official PointMaze files also match their
preregistered SHA-256 identities. The standalone result is
`outputs/full_agent_protocol_audit.json` (`all_checks_pass: true`).

This is a provenance and protocol result, not Claim 1 or Claim 3 performance evidence. It prevents
a later successful training artifact from being credited if the implementation or data have drifted.

## Primary-source audit

- Manuscript: *SVL: Goal-Conditioned Reinforcement Learning as Survival Learning*, arXiv
  `2604.17551v2` (29 May 2026). The inspected PDF is 644,407 bytes with SHA-256
  `f468cfd9e6ea9e06e185a01e3bf81bc703565d55fa7db98c0b4211499c05ec72`.
- Official code: `Simple-Robotics/hierarchical-survival-value-learning`. On 20 July 2026, remote
  `main` still resolved to the sole commit above; the repository had no tags, releases, or checkpoint
  artifacts. Its Git tree is `7a38976b0a065d6ef7f755021853119dc2709487`.
- Source inventory: every one of the 18 locally vendored tracked files matches its official Git blob.
  The only omitted remote file is `intro.png`, a non-executable README illustration.
- Data: official PointMaze train and validation files match SHA-256
  `82a73ed…d2c61` and `19e6b5…6ec4`, respectively.

The frozen source manifest is `repro/configs/upstream_source_manifest.json`. The future training
protocol includes the verified manifest digest, so source drift changes the protocol identity and
invalidates checkpoint reuse. It also binds the local execution-wrapper and provenance-auditor
SHA-256 values. The completion record stores hashes for the protocol artifact and active checkpoint
manifest; future queue invocations rehash that complete chain instead of trusting status fields.

## Gate versus paper-scale Claim 3

| Dimension | Queued gate | Paper protocol / Claim 3 requirement |
|---|---:|---:|
| Environment | official PointMaze-large | a reported Table 1 AntMaze/HumanoidMaze/visual task |
| Optimizer updates | 100 | 1,000,000 for maze tasks |
| Seeds | 1 | 4 |
| Policy evaluation | none | 50 episodes per task in released `main.yaml` |
| Architecture | released full twin critic + both actors | same |
| Width / representation | 512 / 256 | 512 / 256 |
| Batch / horizon | 1,024 / 10,000 | 1,024 / 10,000 for maze tasks |
| Temporal bins | released YAML requests 800; code de-duplicates to 499 | Appendix Table 3 reports 500 |

The `800` versus `500` wording is not hidden: the execution gate follows the released YAML exactly,
while the paper-scale target records Appendix Table 3 separately. The released geometric edge
construction produces 499 unique intervals from the 800-edge request.

## Remaining experimental gate

After the waiting queue releases, the 100-update run must complete, checkpoint, reload exactly, and
emit finite critic and actor metrics. That establishes only feasibility. Full Claim 3 evidence still
requires a reported benchmark task at 1,000,000 updates over four independent seeds, 50 evaluation
episodes per task, and a matched hierarchical-TD comparator or exact reported-result target. No such
performance result is claimed here.
