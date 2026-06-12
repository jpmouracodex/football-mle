r"""
Stage 4 — Score matrix and outcome simulation.

From the estimated parameters, build the bivariate score matrix
:math:`P(X=x, Y=y)` (home vs away goals), already with the Dixon-Coles
correction applied to the low-score cells. From it, derive the aggregated
**1X2** market probabilities: home win (1), draw (X) and away win (2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import FitResult
from .likelihood import Vector, score_matrix_from_rates

__all__ = ["MatchPrediction", "score_matrix", "outcome_probabilities", "predict_match"]


@dataclass
class MatchPrediction:
    """Full prediction of a fixture."""

    home: str
    away: str
    neutral: bool
    matrix: Vector
    prob_home: float
    prob_draw: float
    prob_away: float
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float

    def summary(self) -> str:
        venue = " (neutral venue)" if self.neutral else ""
        score = f"{self.most_likely_score[0]}-{self.most_likely_score[1]}"
        return (
            f"{self.home} (home) x {self.away} (away){venue}\n"
            f"  Expected goals : {self.expected_home_goals:.2f} - "
            f"{self.expected_away_goals:.2f}\n"
            f"  1 (home) : {self.prob_home:.3f}  ({100 * self.prob_home:4.1f}%)\n"
            f"  X (draw) : {self.prob_draw:.3f}  ({100 * self.prob_draw:4.1f}%)\n"
            f"  2 (away) : {self.prob_away:.3f}  ({100 * self.prob_away:4.1f}%)\n"
            f"  Most likely score: {score} ({100 * self.most_likely_score_prob:.1f}%)"
        )


def score_matrix(
    fit_result: FitResult,
    home: str,
    away: str,
    max_goals: int = 10,
    neutral: bool = False,
    normalize: bool = True,
) -> Vector:
    r"""Score matrix of shape ``(max_goals+1, max_goals+1)``.

    Rows index home-team goals, columns index away-team goals. Rates come from the
    model parameters and the Dixon-Coles correction (:math:`\rho`) is applied to
    cells ``(0,0)``, ``(0,1)``, ``(1,0)`` and ``(1,1)``.
    """
    home_rate, away_rate = fit_result.rates(home, away, neutral=neutral)
    return score_matrix_from_rates(
        home_rate, away_rate, rho=fit_result.rho, max_goals=max_goals, normalize=normalize
    )


def outcome_probabilities(matrix: Vector) -> tuple[float, float, float]:
    """Aggregate the score matrix into ``(P(home), P(draw), P(away))``.

    * Home win: home goals > away goals (lower triangle).
    * Draw: main diagonal.
    * Away win: home goals < away goals (upper triangle).
    """
    prob_home = float(np.tril(matrix, k=-1).sum())
    prob_draw = float(np.trace(matrix))
    prob_away = float(np.triu(matrix, k=1).sum())
    return prob_home, prob_draw, prob_away


def predict_match(
    fit_result: FitResult,
    home: str,
    away: str,
    max_goals: int = 10,
    neutral: bool = False,
) -> MatchPrediction:
    """Full fixture prediction: matrix, 1X2, expected goals and modal score."""
    home_rate, away_rate = fit_result.rates(home, away, neutral=neutral)
    matrix = score_matrix_from_rates(
        home_rate, away_rate, rho=fit_result.rho, max_goals=max_goals, normalize=True
    )
    prob_home, prob_draw, prob_away = outcome_probabilities(matrix)

    modal_index = int(np.argmax(matrix))
    row, col = np.unravel_index(modal_index, matrix.shape)

    return MatchPrediction(
        home=home,
        away=away,
        neutral=neutral,
        matrix=matrix,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        # Expected goals = Poisson mean = the rate itself (the Dixon-Coles
        # correction changes this negligibly).
        expected_home_goals=float(home_rate),
        expected_away_goals=float(away_rate),
        most_likely_score=(int(row), int(col)),
        most_likely_score_prob=float(matrix[row, col]),
    )
