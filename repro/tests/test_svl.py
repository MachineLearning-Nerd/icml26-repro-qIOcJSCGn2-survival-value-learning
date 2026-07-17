#!/usr/bin/env python3
"""Exact-identity tests for SVL survival-value (qIOcJSCGn2, Proposition 4.1)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import svl


def test_identity_bellman_equals_survival():
    """V = (I-gamma P_full)^-1 r == -((I-gamma G)^-1 1) on non-goal states."""
    for seed in range(5):
        for gamma in [0.9, 0.99]:
            mdp = svl.random_mdp(nS=12, nA=3, n_goal=2, gamma=gamma, seed=seed)
            VA = svl.bellman_value(mdp); VB = svl.survival_value(mdp)
            nogoal = [s for s in range(12) if s not in mdp["goal"]]
            assert np.max(np.abs(VA[nogoal] - VB[nogoal])) < 1e-10


def test_identity_matches_value_iteration():
    """A third independent route (value iteration) agrees."""
    for seed in range(4):
        mdp = svl.random_mdp(nS=14, nA=3, n_goal=2, gamma=0.95, seed=seed)
        VA = svl.bellman_value(mdp)
        r = -np.ones(14)
        for s in mdp["goal"]:
            r[s] = 0.0
        P_full = svl.full_absorbing(svl.policy_transition(mdp), mdp["goal"])
        V = np.zeros(14)
        for _ in range(20000):
            Vn = r + 0.95 * P_full @ V
            if np.max(np.abs(Vn - V)) < 1e-13:
                break
            V = Vn
        nogoal = [s for s in range(14) if s not in mdp["goal"]]
        assert np.max(np.abs(VA[nogoal] - V[nogoal])) < 1e-9


def test_goal_value_is_zero():
    """Under terminate-on-success, goal states have V=0 (no further penalty)."""
    mdp = svl.random_mdp(seed=0)
    VA = svl.bellman_value(mdp)
    for s in mdp["goal"]:
        assert abs(VA[s]) < 1e-12


def test_negcontrol_state_dependent_reward_breaks_identity():
    """A non-sparse (state-dependent) penalty breaks the identity."""
    mdp = svl.random_mdp(nS=12, n_goal=2, gamma=0.95, seed=1)
    Ppi = svl.policy_transition(mdp); P_full = svl.full_absorbing(Ppi, mdp["goal"])
    r = -np.array([s + 1 for s in range(12)], dtype=float)
    for s in mdp["goal"]:
        r[s] = 0.0
    VA = np.linalg.solve(np.eye(12) - 0.95 * P_full, r)
    VB = svl.survival_value(mdp)
    nogoal = [s for s in range(12) if s not in mdp["goal"]]
    assert np.max(np.abs(VA[nogoal] - VB[nogoal])) > 0.1


def test_negcontrol_nonabsorbing_goal_breaks_identity():
    """Without terminate-on-success (goal not absorbing), the identity fails."""
    mdp = svl.random_mdp(nS=12, n_goal=2, gamma=0.95, seed=2)
    Ppi = svl.policy_transition(mdp)
    r = -np.ones(12)
    for s in mdp["goal"]:
        r[s] = 0.0
    VA = np.linalg.solve(np.eye(12) - 0.95 * Ppi, r)   # raw (non-absorbing)
    VB = svl.survival_value(mdp)
    nogoal = [s for s in range(12) if s not in mdp["goal"]]
    assert np.max(np.abs(VA[nogoal] - VB[nogoal])) > 0.1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
