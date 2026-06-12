r"""
Sanity tests for the mathematics and parameter recovery.

Compatible with ``pytest`` (``pytest -q``) and also runnable directly
(``python tests/test_model.py``) if pytest is not installed.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.synthetic_data import generate_synthetic_league  # noqa: E402
from football_mle import (  # noqa: E402
    fit_dixon_coles,
    fit_maher,
    multiclass_brier,
    pack_parameters,
    prepare_data,
    rps,
    score_matrix_from_rates,
    simulate_tournament,
    tau_correction,
    time_decay_weights,
)
from football_mle.likelihood import compute_rates, unpack_parameters  # noqa: E402
from football_mle.prediction import outcome_probabilities  # noqa: E402


def test_pack_unpack_roundtrip() -> None:
    attack = np.array([1.0, 0.8, 1.2])
    defense = np.array([0.9, 1.1, 1.0])
    theta = pack_parameters(attack, defense, 1.3, rho=-0.1)
    a, d, g, r = unpack_parameters(theta, 3, with_rho=True)
    assert np.allclose(a, attack)
    assert np.allclose(d, defense)
    assert g == 1.3
    assert r == -0.1


def test_tau_values() -> None:
    lam, mu, rho = 1.4, 1.1, -0.12
    assert np.isclose(tau_correction(0, 0, lam, mu, rho), 1 - lam * mu * rho)
    assert np.isclose(tau_correction(0, 1, lam, mu, rho), 1 + lam * rho)
    assert np.isclose(tau_correction(1, 0, lam, mu, rho), 1 + mu * rho)
    assert np.isclose(tau_correction(1, 1, lam, mu, rho), 1 - rho)
    assert np.isclose(tau_correction(2, 0, lam, mu, rho), 1.0)
    assert np.isclose(tau_correction(3, 2, lam, mu, rho), 1.0)


def test_neutral_venue_drops_home_advantage() -> None:
    attack = np.array([1.2, 0.9])
    defense = np.array([0.8, 1.1])
    home_idx = np.array([0])
    away_idx = np.array([1])
    gamma = 1.3
    lam_home, _ = compute_rates(attack, defense, gamma, home_idx, away_idx, np.array([False]))
    lam_neutral, _ = compute_rates(attack, defense, gamma, home_idx, away_idx, np.array([True]))
    assert np.isclose(lam_home[0] / lam_neutral[0], gamma)


def test_score_matrix_sums_to_one() -> None:
    matrix = score_matrix_from_rates(1.7, 1.1, rho=-0.12, max_goals=12)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-9)
    ph, pd_, pa = outcome_probabilities(matrix)
    assert np.isclose(ph + pd_ + pa, 1.0, atol=1e-9)
    assert ph > pa  # higher home rate should win more


def test_rps_edge_cases() -> None:
    assert np.isclose(rps([1.0, 0.0, 0.0], 0), 0.0)
    assert np.isclose(rps([0.0, 0.0, 1.0], 0), 1.0)
    assert np.isclose(rps([0.5, 0.3, 0.2], 0), 0.145)
    assert rps([0.8, 0.1, 0.1], 1) < rps([0.8, 0.1, 0.1], 2)


def test_brier_cases() -> None:
    assert np.isclose(multiclass_brier([1.0, 0.0, 0.0], 0), 0.0)
    assert np.isclose(multiclass_brier([0.0, 0.0, 1.0], 0), 2.0)
    assert np.isclose(multiclass_brier([0.5, 0.3, 0.2], 0), 0.38)


def test_time_decay() -> None:
    dates = ["2024-01-01", "2024-04-01", "2024-07-01"]
    ref = "2024-07-01"
    assert np.allclose(time_decay_weights(dates, ref, 0.0), 1.0)
    weights = time_decay_weights(dates, ref, 0.01)
    assert weights[0] < weights[1] < weights[2]
    assert np.isclose(weights[-1], 1.0)


def test_parameter_recovery_maher() -> None:
    league = generate_synthetic_league(n_teams=16, seed=123)
    data = prepare_data(league.matches, xi=0.0, teams=league.teams)
    model = fit_maher(data)
    assert model.converged
    assert np.corrcoef(model.attack, league.true_attack)[0, 1] > 0.8
    assert np.corrcoef(model.defense, league.true_defense)[0, 1] > 0.8
    assert np.isclose(model.attack.mean(), 1.0, atol=1e-6)
    assert abs(model.home_advantage - league.true_home_advantage) < 0.2


def test_dixon_coles_not_worse_than_maher() -> None:
    league = generate_synthetic_league(n_teams=16, seed=2024)
    data = prepare_data(league.matches, xi=0.0, teams=league.teams)
    maher = fit_maher(data)
    dc = fit_dixon_coles(data)
    assert dc.converged
    assert dc.log_likelihood >= maher.log_likelihood - 1e-4
    assert -0.3 < dc.rho < 0.1


def test_standard_errors_available() -> None:
    league = generate_synthetic_league(n_teams=12, seed=5)
    data = prepare_data(league.matches, xi=0.0, teams=league.teams)
    model = fit_dixon_coles(data, compute_se=True)
    assert model.attack_se is not None and model.attack_se.shape == model.attack.shape
    assert model.rho_se is not None and model.rho_se > 0
    assert np.all(model.attack_se >= 0)


def test_tournament_simulation_probabilities() -> None:
    league = generate_synthetic_league(n_teams=8, seed=7)
    data = prepare_data(league.matches, xi=0.0, teams=league.teams)
    model = fit_dixon_coles(data)
    teams = league.teams[:8]
    rows = []
    for group in (teams[:4], teams[4:]):
        for a, b in combinations(group, 2):
            rows.append({"home": a, "away": b, "neutral": True})
    fixtures = pd.DataFrame(rows)
    result = simulate_tournament(
        model, fixtures, n_simulations=2000, qualifiers_per_group=2, best_third_places=0, seed=1
    )
    probs = result.probabilities
    assert len(probs) == 8
    assert np.isclose(probs["p_champion"].sum(), 1.0, atol=1e-9)
    assert np.isclose(probs["p_advance"].sum(), 4.0, atol=1e-9)  # 2 groups x 2 qualifiers
    assert (probs["p_champion"] >= 0).all() and (probs["p_champion"] <= 1).all()


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  [OK] {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
