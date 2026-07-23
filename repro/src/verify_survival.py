#!/usr/bin/env python3
"""Survival Value Learning (arXiv:2604.17551), Proposition 4.1: under the sparse goal reward
r(s,a,g) = -1{ ||phi(s')-g|| >= eps }, the goal-conditioned value function equals the NEGATIVE
discounted sum of the survival function,

    V^pi(s,g) = - sum_{t=0}^inf gamma^t S^pi(t | s,g),   S^pi(t|s,g) = Pr(T^pi(s,g) > t),

where T is the (random) time-to-goal. We verify this identity EXACTLY on a small stochastic MDP with
multiple paths to the goal (a multimodal time-to-goal distribution), computing:
  (A) V^pi by value iteration on the sparse reward (independent of survival),
  (B) the survival curve S(t) from the absorbing Markov chain (transient-block row sums of P^t),
  (C) -sum_t gamma^t S(t),
and checking (A) == (C). Deterministic.
"""
import numpy as np, json, hashlib

def main():
    R = {"claim": "SurvivalValueLearning_Prop4.1", "paper": "arXiv:2604.17551"}
    gamma = 0.95

    # ---- small stochastic MDP with TWO paths of different length (multimodal hitting time) ----
    # states: S(0) -> either a short branch [1->2->G] or a long branch [3->4->5->6->G]; G=goal absorbing.
    # from S, go up-branch w.p. 0.5 (short) or down-branch w.p. 0.5 (long). Each step also has 0.15
    # chance to "slip" back one state on the long branch -> spreads the second mode.
    names = ["S", "u1", "u2", "d1", "d2", "d3", "d4", "G"]
    idx = {n: i for i, n in enumerate(names)}
    n = len(names); G = idx["G"]
    P = np.zeros((n, n))
    P[idx["S"], idx["u1"]] = 0.5; P[idx["S"], idx["d1"]] = 0.5      # branch choice
    P[idx["u1"], idx["u2"]] = 1.0
    P[idx["u2"], G] = 1.0                                          # short path: S->u1->u2->G (len 3)
    # long path with slip-back
    def step(a, b): P[idx[a], idx[b]] += 1.0
    P[idx["d1"], idx["d2"]] = 1.0
    P[idx["d2"], idx["d3"]] = 0.85; P[idx["d2"], idx["d1"]] = 0.15  # slip
    P[idx["d3"], idx["d4"]] = 0.85; P[idx["d3"], idx["d2"]] = 0.15
    P[idx["d4"], G] = 0.85;         P[idx["d4"], idx["d3"]] = 0.15
    P[G, G] = 1.0                                                   # goal absorbing
    assert np.allclose(P.sum(1), 1.0)

    # ---- (A) value iteration on sparse reward r = -1 for every non-goal state ----
    r = -np.ones(n); r[G] = 0.0
    V = np.zeros(n)
    for _ in range(20000):
        Vn = r + gamma * (P @ V)
        Vn[G] = 0.0                                               # goal terminal, value 0
        if np.max(np.abs(Vn - V)) < 1e-12: V = Vn; break
        V = Vn
    R["V_valueiteration_S"] = round(float(V[idx["S"]]), 6)

    # ---- (B) survival S(t) = Pr(T>t | start=S): transient-block row sum of P^t ----
    trans = [i for i in range(n) if i != G]
    Q = P[np.ix_(trans, trans)]                                   # transient sub-matrix
    s0 = trans.index(idx["S"])
    Tmax = 400; S = []
    Qt = np.eye(len(trans))
    for t in range(Tmax):
        S.append(float(Qt[s0].sum()))                            # Pr(still not absorbed at time t)
        Qt = Qt @ Q
    S = np.array(S)
    # time-to-goal pmf p(T=t) = S(t-1)-S(t); find modes
    pmf = np.concatenate([[1 - S[0]], S[:-1] - S[1:]])
    modes = [t for t in range(1, Tmax - 1) if pmf[t] > pmf[t - 1] and pmf[t] > pmf[t + 1] and pmf[t] > 1e-3]
    R["time_to_goal_modes"] = modes
    R["multimodal_time_to_goal"] = len(modes) >= 2

    # ---- (C) -sum_t gamma^t S(t) ----
    disc = gamma ** np.arange(Tmax)
    V_survival = -float(np.sum(disc * S))
    R["V_survival_identity_S"] = round(V_survival, 6)

    # ---- (D) Monte-Carlo cross-check of the hitting-time distribution ----
    rng = np.random.default_rng(0); K = 40000; hit = []
    for _ in range(K):
        s = idx["S"]; t = 0
        while s != G and t < Tmax:
            s = rng.choice(n, p=P[s]); t += 1
        hit.append(t)
    hit = np.array(hit)
    V_mc = -float(np.mean([np.sum(gamma ** np.arange(h)) for h in hit]))   # -E[sum_{k<T} gamma^k]
    R["V_montecarlo_S"] = round(V_mc, 4)

    R["identity_abs_error"] = round(abs(V[idx["S"]] - V_survival), 8)
    R["prop4.1_identity_holds"] = R["identity_abs_error"] < 1e-6
    R["montecarlo_agrees"] = abs(V_mc - V[idx["S"]]) < 0.05

    R["verdict"] = "supports" if (R["prop4.1_identity_holds"] and R["multimodal_time_to_goal"]
                                  and R["montecarlo_agrees"]) else "inconclusive"

    print("claim: " + R["claim"])
    print("Sparse-reward MDP with a short and a (slip-prone) long path -> multimodal time-to-goal.")
    print()
    print(f"(A) V^pi(S) by value iteration on r=-1{{not goal}} : {R['V_valueiteration_S']}")
    print(f"(C) -sum_t gamma^t S(t) (survival identity)        : {R['V_survival_identity_S']}")
    print(f"    -> |V_VI - V_survival| = {R['identity_abs_error']}  Prop 4.1 identity holds: {R['prop4.1_identity_holds']}")
    print(f"(D) Monte-Carlo V (40k rollouts)                   : {R['V_montecarlo_S']} (agrees: {R['montecarlo_agrees']})")
    print(f"time-to-goal distribution modes at t={R['time_to_goal_modes']} -> multimodal: {R['multimodal_time_to_goal']}")
    print(f"verdict: {R['verdict']}")

    def _np(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError
    import os; os.makedirs("outputs", exist_ok=True)
    open("outputs/survival_results.json", "w").write(json.dumps(R, indent=2, default=_np))
    print("RESULTS_SHA256=" + hashlib.sha256(json.dumps(R, sort_keys=True, default=_np).encode()).hexdigest())
    return 0 if R["verdict"] == "supports" else 1

if __name__ == "__main__":
    raise SystemExit(main())
