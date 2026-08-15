# Source and provenance audit

## Paper

- Title: *SVL: Goal-Conditioned Reinforcement Learning as Survival Learning*
- Authors: Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, Justin Carpentier
- arXiv: [2604.17551](https://arxiv.org/abs/2604.17551)
- OpenReview: [qIOcJSCGn2](https://openreview.net/forum?id=qIOcJSCGn2)
- Project page: [Simple Robotics](https://simple-robotics.github.io/publications/survival-value-learning/)

The paper page and arXiv record are the primary sources for the title, author list, equations, estimator descriptions, and reported benchmark claims. The OpenReview identifier is retained for challenge provenance.

## Official implementation

- Repository: [Simple-Robotics/hierarchical-survival-value-learning](https://github.com/Simple-Robotics/hierarchical-survival-value-learning)
- Pinned source commit: `5f13cf22d397be42a87b7d35336db7662879d6db`
- Local copy: [`upstream/`](../upstream/)
- Relevant implementation: [`upstream/hsvl/utils/survival.py`](../upstream/hsvl/utils/survival.py)

The official implementation is retained as a provenance and cross-check source. The default-branch C2 certificate is clean-room NumPy and does not import JAX or the official training agent.

## Local evidence

- Clean-room identity code: [`repro/src/svl.py`](../repro/src/svl.py)
- Reproduction runner: [`repro/src/run_svl.py`](../repro/src/run_svl.py)
- Existing unit-test source: [`repro/tests/test_svl.py`](../repro/tests/test_svl.py)
- Current summary: [`outputs/svl_summary.json`](../outputs/svl_summary.json)
- Public campaign logbook: [DineshAI/qIOcJSCGn2](https://huggingface.co/spaces/DineshAI/qIOcJSCGn2)

## Provenance rules

1. Official paper and code are cited, not rewritten as local authorship.
2. Local NumPy evidence is labeled clean-room and scoped to the exact identity.
3. Historical OpenResearch/Hugging Face artifacts are preserved as campaign provenance, not silently promoted to fresh results.
4. GPU feasibility, dependency repair, configuration, and seed branches are labeled readiness or experiment work until they contain a faithful claim-level result.
