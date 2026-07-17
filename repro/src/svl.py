#!/usr/bin/env python3
"""Clean-room Survival Value Learning identity (ICML 2026, SVL;
Tiofack, Schramm, Le Hellard, Carpentier; arXiv 2604.17551; OpenReview qIOcJSCGn2).

Proposition 4.1 (exact identity). For a goal-conditioned MDP with sparse per-step penalty
r(s_t,g) = -1{not at goal} and TERMINATE-ON-SUCCESS (goal states absorbing):

    V^pi(s,g) = - sum_{t>=0} gamma^t * S^pi(t | s,g),     S^pi(t|s,g) = Pr(T^pi(s,g) > t)

where T^pi(s,g) is the first-hitting time of the goal. (V is negative because the penalty
is -1 per non-goal step.)

Two independent numpy computations:
  (A) BELLMAN solve on the full absorbing chain:  V_A = (I - gamma P_full)^{-1} r,
      P_full = goal-absorbing transition matrix, r = -1 on non-goal, 0 on goal.
  (B) SURVIVAL matrix-geometric on the killed (non-goal) chain:
      G = substochastic transition matrix restricted to non-goal states;
      S(t|s) = (G^t 1)_s ;  V_B(s) = - sum_t gamma^t (G^t 1)_s = -((I - gamma G)^{-1} 1)_s.

Identity verified iff V_A (non-goal) == V_B to machine precision.
"""
from __future__ import annotations
import numpy as np


def random_mdp(nS=12, nA=3, n_goal=2, gamma=0.95, seed=0):
    """Random MDP, random stochastic policy, goal states (absorbing)."""
    rng = np.random.default_rng(seed)
    P = rng.random((nS, nA, nS))
    P = P / P.sum(2, keepdims=True)
    pi = rng.random((nS, nA)); pi = pi / pi.sum(1, keepdims=True)
    goal = list(rng.choice(nS, size=min(n_goal, nS), replace=False))
    return dict(nS=nS, nA=nA, P=P, pi=pi, goal=goal, gamma=gamma)


def policy_transition(mdp):
    """P^pi(s'|s) = sum_a pi(a|s) P(s'|s,a)."""
    return np.einsum("sa,saq->sq", mdp["pi"], mdp["P"])


def full_absorbing(Ppi, goal):
    """Goal-absorbing transition matrix (goal states self-loop)."""
    P = Ppi.copy()
    for g in goal:
        P[g, :] = 0.0; P[g, g] = 1.0
    return P


def killed_matrix(Ppi, goal):
    """Substochastic 'killed' transition matrix restricted to non-goal states
    (transitions into goal are dropped -> rows sum to <= 1)."""
    nogoal = [s for s in range(Ppi.shape[0]) if s not in goal]
    G = Ppi[np.ix_(nogoal, nogoal)].copy()
    return G, nogoal


def bellman_value(mdp):
    """(A) V_A = (I - gamma P_full)^{-1} r, r=-1 on non-goal, 0 on goal."""
    nS = mdp["nS"]; g = mdp["gamma"]
    Ppi = policy_transition(mdp)
    P_full = full_absorbing(Ppi, mdp["goal"])
    r = -np.ones(nS)
    for s in mdp["goal"]:
        r[s] = 0.0
    A = np.eye(nS) - g * P_full
    return np.linalg.solve(A, r)


def survival_value(mdp):
    """(B) V_B(s) = -((I - gamma G)^{-1} 1)_s on non-goal states; goal states get 0."""
    g = mdp["gamma"]
    Ppi = policy_transition(mdp)
    G, nogoal = killed_matrix(Ppi, mdp["goal"])
    n = len(nogoal)
    V = np.zeros(mdp["nS"])
    Vg = -np.linalg.solve(np.eye(n) - g * G, np.ones(n))
    for i, s in enumerate(nogoal):
        V[s] = Vg[i]
    return V


if __name__ == "__main__":
    mdp = random_mdp(seed=0)
    VA = bellman_value(mdp); VB = survival_value(mdp)
    nogoal = [s for s in range(mdp["nS"]) if s not in mdp["goal"]]
    print("V_A (Bellman)      :", np.round(VA, 4))
    print("V_B (survival)     :", np.round(VB, 4))
    print("max|V_A - V_B| (non-goal):", np.max(np.abs(VA[nogoal] - VB[nogoal])))
