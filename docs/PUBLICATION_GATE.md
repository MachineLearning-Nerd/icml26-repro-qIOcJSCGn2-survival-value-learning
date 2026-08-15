# Publication gate

The repository gate is intentionally scoped. It checks that the public default branch describes the paper honestly and that the one promoted claim has machine-readable evidence. It is not an official ICML challenge score and it does not certify the paper's accelerator-scale benchmark claims.

## Run

```bash
python3 repro/src/verify_results.py
python3 repro/src/publication_gate.py --skip-producers
```

## Required checks

- final project name and paper identifier are present;
- citation, author thank-you, claim matrix, source audit, and branch audit are present;
- `outputs/svl_summary.json` reports the identity and both negative controls as true;
- the default branch has no tracked `.trackio` or stale logbook state;
- all final branch names are purpose-based and no `master`/`orx/*` remote ref remains;
- the configured remote points to `MachineLearning-Nerd/icml26-survival-value-learning`;
- reachable commits use `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`;
- no private workspace path or old repository URL is active in the default-branch documentation;
- the clean-room runner remains executable in the current Python environment.

`SCOPED_PASS` means the documentation and C2 evidence surface is publishable with the stated limitations. It does not mean C1 or C3–C6 are reproduced.
