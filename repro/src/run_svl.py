#!/usr/bin/env python3
"""Verify the SVL survival-value identity (qIOcJSCGn2, Proposition 4.1).

V^pi(s,g) = -sum_t gamma^t Pr(T^pi(s,g) > t)   (sparse per-step penalty, terminate-on-success).

Three independent computations must agree:
  (A) Bellman solve (I - gamma P_full)^{-1} r
  (B) survival matrix-geometric -(I - gamma G)^{-1} 1
  (C) value iteration to convergence

Negative controls (the identity's stated scope):
  (1) state-dependent (non-sparse) penalty -> identity FAILS
  (2) non-absorbing goal (no terminate-on-success) -> identity FAILS
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import svl


def value_iteration(mdp, r, tol=1e-13, maxit=100000):
    """(C) value iteration V <- r + gamma P_full V (3rd independent route)."""
    Ppi = svl.policy_transition(mdp)
    P_full = svl.full_absorbing(Ppi, mdp["goal"]); g = mdp["gamma"]
    V = np.zeros(mdp["nS"])
    for _ in range(maxit):
        Vn = r + g * P_full @ V
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    return V


def main():
    print("=" * 74)
    print("SVL survival-value identity (qIOcJSCGn2, Prop 4.1) -- V = -sum_t gamma^t S(t)")
    print("=" * 74)
    res = {}

    # ---- identity across random MDPs / gamma / size ----
    print("\n(A) Bellman == (B) survival == (C) value-iteration, across MDPs:")
    ident_ok = True; max_diff = 0.0
    for seed in range(6):
        for nS in [8, 16, 30]:
            for gamma in [0.9, 0.99]:
                mdp = svl.random_mdp(nS=nS, nA=3, n_goal=2, gamma=gamma, seed=seed)
                VA = svl.bellman_value(mdp); VB = svl.survival_value(mdp)
                r = -np.ones(nS)
                for s in mdp["goal"]:
                    r[s] = 0.0
                VC = value_iteration(mdp, r)
                nogoal = [s for s in range(nS) if s not in mdp["goal"]]
                d = max(np.max(np.abs(VA[nogoal] - VB[nogoal])),
                        np.max(np.abs(VA[nogoal] - VC[nogoal])))
                max_diff = max(max_diff, d)
                ident_ok &= d < 1e-10
    print(f"  max|V_A - V_B| and |V_A - V_C| over 6 seeds x {{8,16,30}} x {{0.9,0.99}} = {max_diff:.2e}")
    print(f"  -> three independent methods agree to <1e-10: {ident_ok}")
    res["identity"] = dict(ok=bool(ident_ok), max_diff=float(max_diff))

    # ---- negative control 1: state-dependent (non-sparse) penalty -> identity fails ----
    print("\nNegative control 1: state-dependent penalty r(s)=-(s+1) breaks the identity")
    mdp = svl.random_mdp(nS=12, nA=3, n_goal=2, gamma=0.95, seed=1)
    nogoal = [s for s in range(12) if s not in mdp["goal"]]
    # Method A with non-sparse reward; Method B always assumes constant -1 penalty
    Ppi = svl.policy_transition(mdp); P_full = svl.full_absorbing(Ppi, mdp["goal"])
    r_dep = -np.array([s + 1 for s in range(12)], dtype=float)
    for s in mdp["goal"]:
        r_dep[s] = 0.0
    VA_dep = np.linalg.solve(np.eye(12) - 0.95 * P_full, r_dep)
    VB = svl.survival_value(mdp)   # uses -1 penalty (sparse)
    ctrl1 = np.max(np.abs(VA_dep[nogoal] - VB[nogoal])) > 0.1
    print(f"  max|V_A(state-dep r) - V_B(sparse)| = {np.max(np.abs(VA_dep[nogoal]-VB[nogoal])):.4f} (>0.1 => fails): {ctrl1}")
    res["neg_control_nonsparse"] = dict(ok=bool(ctrl1))

    # ---- negative control 2: non-absorbing goal (no terminate-on-success) -> fails ----
    print("\nNegative control 2: non-absorbing goal (no terminate-on-success) breaks the identity")
    mdp2 = svl.random_mdp(nS=12, nA=3, n_goal=2, gamma=0.95, seed=2)
    Ppi2 = svl.policy_transition(mdp2)
    # goal NOT absorbing: agent can leave goal. V_A uses raw Ppi2 (non-absorbing) with r=-1 on non-goal.
    r2 = -np.ones(12)
    for s in mdp2["goal"]:
        r2[s] = 0.0
    VA_nonabs = np.linalg.solve(np.eye(12) - 0.95 * Ppi2, r2)
    VB2 = svl.survival_value(mdp2)   # assumes absorbing goal
    nogoal2 = [s for s in range(12) if s not in mdp2["goal"]]
    ctrl2 = np.max(np.abs(VA_nonabs[nogoal2] - VB2[nogoal2])) > 0.1
    print(f"  max|V_A(non-absorbing) - V_B(absorbing)| = {np.max(np.abs(VA_nonabs[nogoal2]-VB2[nogoal2])):.4f} (>0.1 => fails): {ctrl2}")
    res["neg_control_nonabsorbing"] = dict(ok=bool(ctrl2))

    verified = bool(ident_ok and ctrl1 and ctrl2)
    print("\n" + "=" * 74)
    print(f"C2 SURVIVAL-VALUE IDENTITY: {'VERIFIED' if verified else 'PARTIAL'}")
    print("=" * 74)
    out = os.path.join(HERE, "..", "..", "outputs", "svl_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
