# Branch audit

All 60 original refs were inspected before renaming: `master` plus 59 `orx/*` branches. The final names describe purpose. Branch tips are preserved exactly before the later identity-normalization rewrite; the final remote tips are recorded by the publication verifier.

| Original branch | Final branch | Purpose and evidence boundary | Original tip |
| --- | --- | --- | --- |
| `orx/route-audit-c1-exact-released-architecture` | `audit/claim-1-exact-scale-readiness` | C1 exact released-architecture route feasibility. | `51a69d5feb9ffb2af742ed761742328a462ce074` |
| `orx/route-audit-c3-mle-erm-theorem` | `audit/claim-3-mle-erm-theorem` | C3 finite-dimensional MLE/ERM proof route and assumptions. | `70afb869e8e0bc53d248d9e72f06e1c8f405e15c` |
| `orx/route-audit-c6-flat-svl-crl` | `audit/claim-6-flat-svl-crl` | C6 flat-SVL versus CRL route feasibility. | `06bcd6ebbe43a9f528057f0360548b01a755f542` |
| `orx/route-audit-c4-c5-giant-benchmarks` | `audit/claims-4-5-benchmark-readiness` | C4-C5 matched benchmark route feasibility. | `a5c868575c4bb148c28fb58ddf3ccf5f96a85f07` |
| `orx/interpretation-a-literal-protocol-gate-fix` | `audit/interpretation-literal-gate-fix` | Repair of the literal-protocol audit gate; remains blocked. | `02663074c1e1fb6b398ca553a5d50d3d22863ff7` |
| `orx/interpretation-a-literal-paper-protocol` | `audit/interpretation-literal-protocol` | Literal paper protocol attempt; blocked before faithful evaluation. | `20b582a4c733112967aa188a63b1e2a81c422d47` |
| `orx/interpretation-b-official-artifact-witness` | `audit/interpretation-official-artifact-witness` | Search for immutable author artifacts/checkpoints; no eligible artifact promoted. | `2bebeb40d6780010f6667b189d6532c65755e66f` |
| `orx/interpretation-c-table-arithmetic-falsification` | `audit/interpretation-table-arithmetic` | Exact table mean/variance arithmetic compatibility audit; not policy evidence. | `e5b4548f7e4c3cf9939f27048b205018f978ac60` |
| `orx/round-1-route-audit-logbook` | `audit/round-1-route-feasibility` | First-round feasibility audit for C1 and C3-C6; route decisions only, no new scientific result. | `8a36dda170cf2e7f5afa494904501fb6370eca27` |
| `orx/three-interpretation-retry-logbook` | `audit/three-interpretation-retry` | Literal-protocol, official-artifact, and table-arithmetic interpretation retries with integrity controls. | `c264506c7091adff01ed896038237643598fd8f6` |
| `orx/frozen-live-4-of-12-six-claim-baseline` | `baseline/frozen-four-of-twelve` | Frozen campaign snapshot: bounded C1 PointMaze evidence, exact-width T4 preflight, source/protocol archaeology; full-scale claims remain blocked. | `bd368be36cec3dbebfc1187fca1493445d4494b9` |
| `orx/qio-six-claim-seed-20260723` | `baseline/six-claim-seed-20260723` | Six-claim campaign seed and live-score baseline; C2 verified, C1/C3-C6 not promoted. | `bd368be36cec3dbebfc1187fca1493445d4494b9` |
| `orx/hf-cpu-dependency-repair-diagnostic` | `experiment/hf-cpu-dependency-repair` | HF CPU dependency repair diagnostic for the official training stack. | `fb70c0469efc50481bb7da036ea72d9e296afda2` |
| `orx/hf-cpu-full-scale-campaign-preflight` | `experiment/hf-cpu-full-scale-preflight` | HF CPU full-scale campaign preflight and failure-surface logging. | `f8639d900136e3db09888483eb8742026600f555` |
| `orx/hf-cpu-hashable-config-diagnostic` | `experiment/hf-cpu-hashable-config` | Static-config hashability repair diagnostic. | `e9f84fca9fc4212e6ecb3c54d488fd7100af45a1` |
| `orx/hf-cpu-integration-crl-humanoid` | `experiment/hf-cpu-integration-crl-humanoid` | CPU integration configuration for crl-humanoid; infrastructure/readiness only. | `00469d6652d8cc0243b7258c24f4760ed42400e8` |
| `orx/hf-cpu-integration-flat-svl-humanoid` | `experiment/hf-cpu-integration-flat-svl-humanoid` | CPU integration configuration for flat-svl-humanoid; infrastructure/readiness only. | `af7673c2df01b7fa1292698b2286e38093c5767c` |
| `orx/hf-cpu-integration-hiql-ant` | `experiment/hf-cpu-integration-hiql-ant` | CPU integration configuration for hiql-ant; infrastructure/readiness only. | `1374fa2eecf5fed035a85847fe36e4b338ba9c1f` |
| `orx/hf-cpu-integration-hiql-humanoid` | `experiment/hf-cpu-integration-hiql-humanoid` | CPU integration configuration for hiql-humanoid; infrastructure/readiness only. | `ff071f880552f8daad1fc35057f8c1666d282cd9` |
| `orx/hf-cpu-integration-hsvl-humanoid` | `experiment/hf-cpu-integration-hsvl-humanoid` | CPU integration configuration for hsvl-humanoid; infrastructure/readiness only. | `90ff127875fd7c275af3710078dc3d7095f3f6a7` |
| `orx/hf-cpu-integration-visual-hiql-ant` | `experiment/hf-cpu-integration-visual-hiql-ant` | CPU integration configuration for visual-hiql-ant; infrastructure/readiness only. | `3e591f96cfd1342d5a2f650248b9acc8915a0662` |
| `orx/hf-cpu-integration-visual-hsvl-ant` | `experiment/hf-cpu-integration-visual-hsvl-ant` | CPU integration configuration for visual-hsvl-ant; infrastructure/readiness only. | `ab07dbae487b46a791eb9cb384d8287f6431ea49` |
| `orx/hf-cpu-method-import-diagnostic` | `experiment/hf-cpu-method-import` | Method-import smoke diagnostic. | `6c7a3f82f3c987f7cdcfc2efd3d34fe2f3033854` |
| `orx/hf-cpu-gl-import-repair-diagnostic` | `experiment/hf-cpu-opengl-repair` | OpenGL/import runtime repair diagnostic. | `2cb61d9992ef00c319cbc3aa21d9ea811df9d98a` |
| `orx/hf-cpu-resumable-training-runner-integration` | `experiment/hf-cpu-resumable-runner` | Resumable training-runner integration. | `4ec3f98ab995e6f02d9766a49c1443ef366525d3` |
| `orx/hf-cpu-paper-scale-throughput-benchmark` | `experiment/hf-cpu-throughput-benchmark` | Paper-scale CPU throughput benchmark; no claim-level result. | `d7ea59a0fbdc18dd95a31e1a5982f99a20160d25` |
| `orx/paper-scale-crl-humanoid-seed-0` | `experiment/paper-scale-crl-humanoid-seed-0` | Paper-scale crl-humanoid-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `f92b9bd8316990a7d7b7e498ba697d4d26c071a8` |
| `orx/paper-scale-crl-humanoid-seed-1` | `experiment/paper-scale-crl-humanoid-seed-1` | Paper-scale crl-humanoid-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `08c2bb18144db0dff98c272455f7682a8a5233e1` |
| `orx/paper-scale-crl-humanoid-seed-2` | `experiment/paper-scale-crl-humanoid-seed-2` | Paper-scale crl-humanoid-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `50b089d58a6d6ce37bfa6706e5dce6528e3c39ff` |
| `orx/paper-scale-crl-humanoid-seed-3` | `experiment/paper-scale-crl-humanoid-seed-3` | Paper-scale crl-humanoid-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `f889a7101cbe62672b9fc8398c05c26a8fb592a9` |
| `orx/paper-scale-flat-svl-humanoid-seed-0` | `experiment/paper-scale-flat-svl-humanoid-seed-0` | Paper-scale flat-svl-humanoid-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `a39f080d824ffb919adde4781297370459b3cb48` |
| `orx/paper-scale-flat-svl-humanoid-seed-1` | `experiment/paper-scale-flat-svl-humanoid-seed-1` | Paper-scale flat-svl-humanoid-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `f84634ebeae9ace909a107544027742889b8708d` |
| `orx/paper-scale-flat-svl-humanoid-seed-2` | `experiment/paper-scale-flat-svl-humanoid-seed-2` | Paper-scale flat-svl-humanoid-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `aaf954d3018aad16eda909e4ccaf281e2747b9f2` |
| `orx/paper-scale-flat-svl-humanoid-seed-3` | `experiment/paper-scale-flat-svl-humanoid-seed-3` | Paper-scale flat-svl-humanoid-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `b22f687820b6b2cf5bdeaf9a7944c9cee3cf8730` |
| `orx/paper-scale-hiql-ant-seed-0` | `experiment/paper-scale-hiql-ant-seed-0` | Paper-scale hiql-ant-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `dda12885678814ce38731a50ab41b6f723f46a90` |
| `orx/paper-scale-hiql-ant-seed-1` | `experiment/paper-scale-hiql-ant-seed-1` | Paper-scale hiql-ant-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `4840918ffb97527b41d9860c5c428586539cec7e` |
| `orx/paper-scale-hiql-ant-seed-2` | `experiment/paper-scale-hiql-ant-seed-2` | Paper-scale hiql-ant-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `fe7aaac0ecb3a7269f0eacf6fbb2b9508f5139b9` |
| `orx/paper-scale-hiql-ant-seed-3` | `experiment/paper-scale-hiql-ant-seed-3` | Paper-scale hiql-ant-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `aea8eccea8aea0c0b8f3f6f5310d004267b7661e` |
| `orx/paper-scale-hiql-humanoid-seed-0` | `experiment/paper-scale-hiql-humanoid-seed-0` | Paper-scale hiql-humanoid-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `4eb2a73af746e49999b1c94a7ad72d74fc15ecbd` |
| `orx/paper-scale-hiql-humanoid-seed-1` | `experiment/paper-scale-hiql-humanoid-seed-1` | Paper-scale hiql-humanoid-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `db706d2b959134ef1ee60d649533295d3666176b` |
| `orx/paper-scale-hiql-humanoid-seed-2` | `experiment/paper-scale-hiql-humanoid-seed-2` | Paper-scale hiql-humanoid-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `664bc420d1283dacd37aa9c55fb3eaca03013d83` |
| `orx/paper-scale-hiql-humanoid-seed-3` | `experiment/paper-scale-hiql-humanoid-seed-3` | Paper-scale hiql-humanoid-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `42a3aa5f713993e260c90a3b2fcd43c215223786` |
| `orx/paper-scale-hsvl-ant-seed-0` | `experiment/paper-scale-hsvl-ant-seed-0` | Paper-scale hsvl-ant-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `e41d84584d7b86b65e4e114a635e35a4586f4e21` |
| `orx/paper-scale-hsvl-ant-seed-1` | `experiment/paper-scale-hsvl-ant-seed-1` | Paper-scale hsvl-ant-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `414e5b97de80d6ea91ad06b33a02084b44525e03` |
| `orx/paper-scale-hsvl-ant-seed-2` | `experiment/paper-scale-hsvl-ant-seed-2` | Paper-scale hsvl-ant-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `d82f9b45e6ceccd34c71ccfdd48b15de3478f99c` |
| `orx/paper-scale-hsvl-ant-seed-3` | `experiment/paper-scale-hsvl-ant-seed-3` | Paper-scale hsvl-ant-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `f559aa7e5ea6b774fc60e91c6f1ba4db5e501cb7` |
| `orx/paper-scale-hsvl-humanoid-seed-0` | `experiment/paper-scale-hsvl-humanoid-seed-0` | Paper-scale hsvl-humanoid-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `7435de2ffb9f2ec5faed190916f20dbe815c5064` |
| `orx/paper-scale-hsvl-humanoid-seed-1` | `experiment/paper-scale-hsvl-humanoid-seed-1` | Paper-scale hsvl-humanoid-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `50ef6ee1917710305fff9b38a1d0608ee895f1c2` |
| `orx/paper-scale-hsvl-humanoid-seed-2` | `experiment/paper-scale-hsvl-humanoid-seed-2` | Paper-scale hsvl-humanoid-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `890f4c3604d6dcff3e116d1966b1ca91c417b3a1` |
| `orx/paper-scale-hsvl-humanoid-seed-3` | `experiment/paper-scale-hsvl-humanoid-seed-3` | Paper-scale hsvl-humanoid-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `63905748c245d9426719d5530fd36c6e39b77dbc` |
| `orx/paper-scale-visual-hiql-ant-seed-0` | `experiment/paper-scale-visual-hiql-ant-seed-0` | Paper-scale visual-hiql-ant-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `6436769dc1f993989c31f1abd28bc2b0e475597f` |
| `orx/paper-scale-visual-hiql-ant-seed-1` | `experiment/paper-scale-visual-hiql-ant-seed-1` | Paper-scale visual-hiql-ant-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `809971f16cc72c575c4c2c01147be14308bb61be` |
| `orx/paper-scale-visual-hiql-ant-seed-2` | `experiment/paper-scale-visual-hiql-ant-seed-2` | Paper-scale visual-hiql-ant-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `3b9753aa52588b1e101f6719d9f8a8b4d73f528b` |
| `orx/paper-scale-visual-hiql-ant-seed-3` | `experiment/paper-scale-visual-hiql-ant-seed-3` | Paper-scale visual-hiql-ant-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `6e87260f251f8f2f0a020fe77dea4b13270bea19` |
| `orx/paper-scale-visual-hsvl-ant-seed-0` | `experiment/paper-scale-visual-hsvl-ant-seed-0` | Paper-scale visual-hsvl-ant-seed-0 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `79089d8bf41e77b61aa1644957f71384a1191972` |
| `orx/paper-scale-visual-hsvl-ant-seed-1` | `experiment/paper-scale-visual-hsvl-ant-seed-1` | Paper-scale visual-hsvl-ant-seed-1 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `06d4065a9111d55669be5ec67b4753b683140930` |
| `orx/paper-scale-visual-hsvl-ant-seed-2` | `experiment/paper-scale-visual-hsvl-ant-seed-2` | Paper-scale visual-hsvl-ant-seed-2 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `ad10c765f71e1339333e13ef9a12a693a974f1de` |
| `orx/paper-scale-visual-hsvl-ant-seed-3` | `experiment/paper-scale-visual-hsvl-ant-seed-3` | Paper-scale visual-hsvl-ant-seed-3 configuration/seed branch; retained as experiment provenance, not a completed reproduction. | `cfbfb86d0958df683bc2aa54a7a67d86e434b4e0` |
| `master` | `main` | Default clean-room baseline; C2 source and original five-test harness. | `37c43b05889e7aeda07c789db9ec38166161bc2f` |
| `orx/hf-logbook-text-only-release-candidate` | `release/hf-logbook-text-only` | Text-only public logbook release candidate. | `377bc4ce8159911d13ad1998ccdd0f6e7479d579` |

## Naming rules

- `main` is the clean default branch.
- `baseline/` preserves campaign snapshots.
- `audit/` records claim, protocol, interpretation, and readiness audits.
- `experiment/` records infrastructure, integration, throughput, and paper-scale seed work.
- `release/` records the text-only logbook release.

A branch label is a navigation aid, not a scientific verdict. The claim matrix in [`docs/CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md) is authoritative for what is verified.
