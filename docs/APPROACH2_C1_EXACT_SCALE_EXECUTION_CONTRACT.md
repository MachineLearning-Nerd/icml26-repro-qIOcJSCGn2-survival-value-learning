# Approach 2 C1 exact-scale execution contract

Status: **static, fail-closed, and not authorized to launch**.

This route resumes the returned seed-0 HSVL checkpoint from update 100 to
update 1,000,000 on `pointmaze-large-navigate-v0`, then records 50 deterministic
policy episodes and an independent held-out twin-survival audit. It is one
exact-scale seed for **C1 only**. It is not C3, not a three-seed reproduction,
and does not guarantee a judge-score increase.

The source of truth is
`repro/hf/qio_c1_exact_seed0_contract.json`. The accompanying
`repro/hf/check_qio_c1_exact_seed0_gates.py` is read-only and has no launch or
mutation operation. It exits nonzero until every gate is satisfied.

## Bound lineage

- Update-100 checkpoint: SHA-256
  `689ed10245e4926249316c79f7502ba692343c583e6fdfcc0de458020f606925`,
  117,362,957 bytes.
- Restored state: SHA-256
  `7bc393ffaac206b7deaa64de5e67634c4c0556aed1d0ff43e5dc258fbc25d9a0`.
- Active checkpoint manifest: SHA-256
  `65a50a2c154f0f19cd1d28739c7934f97a84950cbc2404c1ce77cd0a809c8680`.
- Protocol file: SHA-256
  `6b59171427fedd45c5034f74770859ef59f07dde9833adc784d04e64abfe6614`;
  canonical semantic identity
  `9c54d123aa361d8d6fcfe77a1049445490b89a67ae2749a36e7bf460ac0b11ea`.
- Frozen preflight runner: SHA-256
  `c54c972b3e02dd013cc972f8847b15b3bec28168458369c186c659eb424e29cd`.
- Returned archive: SHA-256
  `25911c0bc00a4200f3a1664d0d70c7a7b7dadf397dafcc7e2d36eaf40d7d41fa`,
  96,720,938 bytes. Its preflight report and integrity record are
  `9ccf813d81403f300908aff319b7c8f8142b19148c29c39e959d32b3ea61b6f8`
  and `ce681ac991e2c732f96f703aa83222b5aee5821ba87ac4d20ea94c692792f3a0`.
- Official source manifest: SHA-256
  `e9a279e09fa5952002b8f37a2aa54c7e02a711fcff3f86c362101d6e406131a6`;
  HSVL commit `5f13cf22d397be42a87b7d35336db7662879d6db`, tree
  `7a38976b0a065d6ef7f755021853119dc2709487`, and OGBench commit
  `1d4140997f60c52c6fb0702ec100dc988b18c548`.
- Train/validation data: SHA-256
  `82a73ed8de90ad2b8bf89069253e961918438d127d7f7c7f09c81e642d0d2c61`
  and `19e6b510c800b865d5c3f9e4335941a602731184fa683eb99face7c5405b6ec4`.

The update-100 return proves exact restore/reload, both actor modules, both
value heads, and the expected 9,778,698 parameters. It contains no policy
evaluation and is explicitly an execution gate rather than paper-scale
evidence.

## Exact gate sequence

1. Leave protected qIO PID 41861 and its observed dependency PID 41276
   untouched. Refuse while PID 41861 exists or another qIO full-agent or
   continuation process matches the contract; PID 41276 is recorded only to
   preserve its unrelated work and is never signalled.
2. The continuation/policy runner, independent NumPy twin-survival auditor,
   terminal return verifier, and immutable job source manifest have been
   implemented, statically inspected, and SHA-256 frozen in the contract. No
   metric, control, seed, or acceptance threshold may change after this freeze.
3. Have the central campaign controller add exactly one reservation row named
   `pending-qio-c1-exact-seed0-full-v1` to
   `../../icml-2026-reproduction-challenge/HF_JOBS_BUDGET.md`: `t4-small`,
   `reserved; not submitted`, at most 48 hours and at most USD 19.20. This
   paper-local work must not edit the central ledger.
4. Run the local read-only checker without the live flag. It must verify every
   source, data, protocol, checkpoint, completion, component, process, and
   reservation gate. It will still refuse because a live account check is
   intentionally absent.
5. At the immediate pre-submit boundary, run the checker once with
   `--check-live-hf`. It performs one read-only account-wide active-state gate
   using separate `RUNNING` and `SCHEDULING` queries, and refuses if **any** T4
   job is in either state, whether campaign-owned or external. It never stops
   or mutates a job. The eventual launcher must repeat the same read-only
   exclusion gate exactly once at its submit boundary to close the race;
   failure means defer without staging or submitting.
6. Submit only the hash-frozen seed-0 job on `t4-small`, with a 48-hour timeout
   and mounted persistent storage. No second same-flavor campaign job may be
   submitted. Never cancel, pause, kill, or mutate any account job.
7. Resume the exact update-100 state, atomically persist immutable checkpoint
   generations every 50,000 updates, and treat the mounted checkpoint as the
   source of truth. A timeout is a paused/resumable operational state, not a
   scientific failure; a new attempt needs a fresh reservation and all gates.
8. After update 1,000,000, exact-reload the terminal checkpoint. Only then run
   the frozen 50-episode deterministic policy protocol, held-out validation
   tuple audit, both twin heads, calibration, and both negative controls.
9. Return raw episode rows, raw twin logits/targets, independent metrics,
   manifests, checkpoint/session traces, completion proof, and `SHA256SUMS`.
   Import only after terminal status, bucket before/after inventory, complete
   hash verification, independent reconstruction of every policy-summary
   aggregate from the 50 raw episode rows, and independent local survival
   recomputation. Do not publish.

## Scientific interpretation boundary

The 32,768 validation tuples are selected with seed 20260719 and evaluated in
chunks of 8,192. Both heads and their mean must be reported. The joint-target
shuffle uses seed 20260721; temporal-hazard reversal is the second negative
control. Both controls are retained regardless of result.

A result is only a C1-support candidate if all artifacts and values are finite,
distribution normalization error is at most `1e-10`, survival monotonicity
increase is at most `1e-12`, and both controls worsen censored NLL. There is no
post hoc calibration threshold: raw calibration is reported. A failed
scientific gate is an inconclusive or negative C1 result, not an integrity
failure, and must not be discarded.

## Cost ceiling and present blockers

At USD 0.40/hour, the median update rate projects 12.9477 hours / USD 5.1791.
The conservative observed rate projects 43.9356 hours / USD 17.5742. The hard
reservation is 48 hours / **USD 19.20**, leaving about 4.06 hours beyond the
conservative projection.

The four frozen component identities are:

- continuation/policy runner:
  `90ea2ee78cd918d1e39cd2659ce7eeac09e433543a38f8c0a61dab36d8c4fce0`;
- independent NumPy auditor:
  `12274e0c550ce0795585c120e6be37ec6a4fb47b6818ba9b86dc97ba606e4198`;
- terminal return verifier:
  `58c11f186c0075d9d1cc072b5b1a0212e372ebf0d941202e634f766c2ac92c4f`;
- immutable job source manifest:
  `8638e6fffb347a1998772d0771e291a41b6d7ebe1c67b0145098688ddd2ea851`.

This was a static review only: no training, evaluation, test suite, or HF job
was run. The plan still refuses launch because protected local qIO work is
alive, the current ledger snapshot records an active FluxNet T4 job, and the
USD 19.20 qIO C1 reservation is absent. Those are launch blockers, not
permission to stop existing work.
