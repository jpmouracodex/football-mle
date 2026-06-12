r"""
Synthetic-league generator for demonstration and testing.

Simulates a double round-robin season from known "true" parameters, sampling
each score **directly from the Dixon-Coles distribution**. This lets us check
empirically that the MLE recovers the generating parameters.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script directly (without installing the package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_mle.likelihood import score_matrix_from_rates  # noqa: E402


@dataclass
class SyntheticLeague:
    """Bundle of simulated data and the true parameters."""

    matches: pd.DataFrame
    teams: list[str]
    true_attack: np.ndarray
    true_defense: np.ndarray
    true_home_advantage: float
    true_rho: float


def _sample_score(
    home_rate: float,
    away_rate: float,
    rho: float,
    rng: np.random.Generator,
    max_goals: int = 15,
) -> tuple[int, int]:
    """Sample a score (home_goals, away_goals) from the Dixon-Coles matrix."""
    matrix = score_matrix_from_rates(home_rate, away_rate, rho=rho, max_goals=max_goals, normalize=True)
    flat = matrix.ravel()
    flat = flat / flat.sum()
    index = rng.choice(flat.size, p=flat)
    row, col = divmod(int(index), max_goals + 1)
    return row, col


def generate_synthetic_league(
    n_teams: int = 20,
    home_advantage: float = 1.35,
    rho: float = -0.12,
    sigma_attack: float = 0.35,
    sigma_defense: float = 0.30,
    start_date: str = "2024-08-10",
    days_between_rounds: int = 4,
    seed: int = 42,
) -> SyntheticLeague:
    r"""Generate a synthetic round-robin league (home and away).

    Attack/defense are drawn on a log-normal scale and attack is normalized to
    mean 1 (the model's identifiability constraint). Each round gets a date, so
    time decay can be exercised.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {i + 1:02d}" for i in range(n_teams)]

    attack = np.exp(rng.normal(0.0, sigma_attack, size=n_teams))
    attack = attack / attack.mean()  # enforce mean(attack) = 1
    defense = np.exp(rng.normal(0.0, sigma_defense, size=n_teams))

    fixtures = list(permutations(range(n_teams), 2))  # every (i, j), i != j
    rng.shuffle(fixtures)

    base_date = pd.Timestamp(start_date)
    matches_per_round = n_teams // 2
    records = []
    for k, (i, j) in enumerate(fixtures):
        round_index = k // matches_per_round
        date = base_date + pd.Timedelta(days=round_index * days_between_rounds)
        home_rate = attack[i] * defense[j] * home_advantage
        away_rate = attack[j] * defense[i]
        home_goals, away_goals = _sample_score(home_rate, away_rate, rho, rng)
        records.append(
            {
                "date": date,
                "home": teams[i],
                "away": teams[j],
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    matches = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return SyntheticLeague(
        matches=matches,
        teams=teams,
        true_attack=attack,
        true_defense=defense,
        true_home_advantage=home_advantage,
        true_rho=rho,
    )


if __name__ == "__main__":
    league = generate_synthetic_league()
    destination = Path(__file__).resolve().parents[1] / "sample_data" / "synthetic_league.csv"
    destination.parent.mkdir(exist_ok=True)
    league.matches.to_csv(destination, sep=";", decimal=",", index=False, encoding="utf-8")
    print(f"{len(league.matches)} matches generated and saved to:\n  {destination}")
    print(league.matches.head(10).to_string(index=False))
