r"""
Stage 5 — Validation (Ranked Probability Score and multi-class Brier) plus
temporal cross-validation and optimal time-decay selection.

To assess the **calibration** of the 1X2 probabilities on a held-out set, two
proper scoring rules are used:

Ranked Probability Score (RPS)
    Recommended by Constantinou & Fenton (2012) for football because it respects
    the natural **order** of outcomes (home > draw > away):

    .. math::

        \mathrm{RPS} = \frac{1}{r-1}
        \sum_{i=1}^{r-1}\Big(\sum_{j=1}^{i}(p_j - o_j)\Big)^2 \in [0, 1].

Multi-class Brier
    Mean squared distance between the probability vector and the outcome
    indicator, **without** using the class order:
    :math:`\mathrm{BS} = \sum_{j} (p_j - o_j)^2 \in [0, 2]`.

Both are proper; **lower is better**. The category order used is
:data:`OUTCOME_ORDER` = ``("home", "draw", "away")``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .data import half_life_from_xi, prepare_data
from .fitting import FitResult
from .prediction import predict_match

__all__ = [
    "OUTCOME_ORDER",
    "outcome_index",
    "indicator_vector",
    "rps",
    "multiclass_brier",
    "match_log_loss",
    "EvaluationResult",
    "evaluate",
    "temporal_cross_validation",
    "optimize_xi",
]

# Ordinal order of outcomes (matters for the RPS).
OUTCOME_ORDER: tuple[str, str, str] = ("home", "draw", "away")


def outcome_index(home_goals: int, away_goals: int) -> int:
    """Map a score to the outcome index: 0=home, 1=draw, 2=away."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def indicator_vector(index: int, n_classes: int = 3) -> npt.NDArray[np.float64]:
    """One-hot vector of the observed outcome."""
    vector = np.zeros(n_classes, dtype=np.float64)
    vector[index] = 1.0
    return vector


def rps(probabilities: npt.ArrayLike, actual_index: int) -> float:
    r"""Ranked Probability Score of a **single** match.

    ``probabilities`` is :math:`(p_{\text{home}}, p_{\text{draw}}, p_{\text{away}})`
    in :data:`OUTCOME_ORDER`; ``actual_index`` is the observed outcome index.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    o = indicator_vector(actual_index, len(p))
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    # The last cumulative term is always (1 - 1) = 0; sum only r-1 terms.
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(p) - 1))


def multiclass_brier(probabilities: npt.ArrayLike, actual_index: int) -> float:
    """Multi-class Brier score of a match (sum of squared residuals)."""
    p = np.asarray(probabilities, dtype=np.float64)
    o = indicator_vector(actual_index, len(p))
    return float(np.sum((p - o) ** 2))


def match_log_loss(probabilities: npt.ArrayLike, actual_index: int, epsilon: float = 1e-15) -> float:
    r"""Log-loss (cross-entropy) of a match: :math:`-\ln p_{\text{actual}}`."""
    p = np.asarray(probabilities, dtype=np.float64)
    return float(-np.log(np.clip(p[actual_index], epsilon, 1.0)))


@dataclass
class EvaluationResult:
    """Aggregated metrics of a model on a test set."""

    model: str
    n_matches: int
    mean_rps: float
    mean_brier: float
    mean_log_loss: float
    accuracy: float

    def summary(self) -> str:
        return (
            f"Validation - {self.model}  (n = {self.n_matches})\n"
            f"  Mean RPS      : {self.mean_rps:.4f}   (lower is better)\n"
            f"  Mean Brier    : {self.mean_brier:.4f}   (lower is better)\n"
            f"  Mean log-loss : {self.mean_log_loss:.4f}   (lower is better)\n"
            f"  1X2 accuracy  : {self.accuracy:.3f}"
        )


def evaluate(
    fit_result: FitResult,
    test_df: pd.DataFrame,
    max_goals: int = 10,
) -> EvaluationResult:
    """Evaluate a fitted model on a canonical-schema test set.

    For each test match, predict the 1X2 vector, compare to the actual outcome
    and accumulate RPS, Brier, log-loss and accuracy. Matches with teams unseen
    in training are skipped. The ``neutral`` column is honored if present.
    """
    rps_values: list[float] = []
    brier_values: list[float] = []
    log_loss_values: list[float] = []
    correct = 0
    n = 0

    has_neutral = "neutral" in test_df.columns
    known = set(fit_result.teams)
    for _, match in test_df.iterrows():
        home, away = match["home"], match["away"]
        if home not in known or away not in known:
            continue

        neutral = bool(match["neutral"]) if has_neutral else False
        prediction = predict_match(fit_result, home, away, max_goals=max_goals, neutral=neutral)
        probabilities = np.array([prediction.prob_home, prediction.prob_draw, prediction.prob_away])
        actual = outcome_index(int(match["home_goals"]), int(match["away_goals"]))

        rps_values.append(rps(probabilities, actual))
        brier_values.append(multiclass_brier(probabilities, actual))
        log_loss_values.append(match_log_loss(probabilities, actual))
        correct += int(np.argmax(probabilities) == actual)
        n += 1

    if n == 0:
        raise ValueError("No test match could be evaluated (unknown teams?).")

    return EvaluationResult(
        model=fit_result.model,
        n_matches=n,
        mean_rps=float(np.mean(rps_values)),
        mean_brier=float(np.mean(brier_values)),
        mean_log_loss=float(np.mean(log_loss_values)),
        accuracy=correct / n,
    )


# ---------------------------------------------------------------------------
# Temporal cross-validation and optimal time-decay selection
# ---------------------------------------------------------------------------
def temporal_cross_validation(
    df: pd.DataFrame,
    fit_function: Callable[..., FitResult],
    *,
    xi: float = 0.0,
    n_splits: int = 4,
    min_train_fraction: float = 0.5,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    The data is sorted by date; the tail ``1 - min_train_fraction`` is split into
    ``n_splits`` contiguous test blocks. For each block, the model is trained on
    all earlier matches and scored on the block. Returns one row per fold.
    """
    if df["date"].isna().any():
        raise ValueError("temporal_cross_validation requires a valid 'date' column.")

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    start = int(n * min_train_fraction)
    edges = np.linspace(start, n, n_splits + 1).astype(int)

    rows = []
    for k in range(n_splits):
        train = df.iloc[: edges[k]]
        test = df.iloc[edges[k] : edges[k + 1]]
        if len(test) == 0 or len(train) == 0:
            continue
        data = prepare_data(train, xi=xi, reference_date=train["date"].max())
        fit_result = fit_function(data)
        evaluation = evaluate(fit_result, test, max_goals=max_goals)
        rows.append(
            {
                "fold": k,
                "n_train": len(train),
                "n_test": evaluation.n_matches,
                "mean_rps": evaluation.mean_rps,
                "mean_brier": evaluation.mean_brier,
                "accuracy": evaluation.accuracy,
            }
        )
    return pd.DataFrame(rows)


def optimize_xi(
    df: pd.DataFrame,
    xi_grid: Sequence[float],
    fit_function: Callable[..., FitResult],
    *,
    n_splits: int = 4,
    min_train_fraction: float = 0.5,
    max_goals: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Select the time-decay rate ``xi`` that minimizes cross-validated RPS.

    Runs :func:`temporal_cross_validation` for each candidate ``xi`` and returns
    ``(best_xi, table)`` where ``table`` has the mean RPS per ``xi`` (with the
    corresponding half-life in days).
    """
    rows = []
    for xi in xi_grid:
        folds = temporal_cross_validation(
            df,
            fit_function,
            xi=xi,
            n_splits=n_splits,
            min_train_fraction=min_train_fraction,
            max_goals=max_goals,
        )
        rows.append(
            {
                "xi": xi,
                "half_life_days": half_life_from_xi(xi),
                "mean_rps": float(folds["mean_rps"].mean()),
                "mean_brier": float(folds["mean_brier"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    best_xi = float(table.loc[table["mean_rps"].idxmin(), "xi"])
    return best_xi, table
