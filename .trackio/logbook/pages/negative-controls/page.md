# Negative controls

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_sn_01", "created_at": "2026-07-17T19:03:00+00:00", "title": "State-dependent reward"}
-->
**State-dependent penalty breaks the identity.** A non-sparse reward r(s)=−(s+1) gives
max|V_A − V_B| = 31.2 ≫ 0 — the survival identity requires the constant per-step penalty, so it
correctly fails here. (This is the theorem's stated scope: it holds for sparse penalty, not dense/shaped.)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_sn_02", "created_at": "2026-07-17T19:03:10+00:00", "title": "Non-absorbing goal"}
-->
**Non-absorbing goal breaks the identity.** Without terminate-on-success (goal not absorbing),
max|V_A(non-absorbing) − V_B| = 12.3 ≫ 0 — the survival framing requires the absorbing goal, so it
correctly fails. This confirms the identity's terminate-on-success precondition is load-bearing.
