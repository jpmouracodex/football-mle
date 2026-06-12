r"""
football_mle — Goal prediction by Maximum Likelihood Estimation (MLE).

A from-scratch implementation of the Maher (1982) and Dixon-Coles (1997) models
(extending the applied Poisson model of Vizzoni, 2018) for football result
prediction, featuring:

* time-decay weighting (Dixon & Coles, 1997);
* MLE via ``scipy.optimize.minimize`` (SLSQP) rather than a black-box GLM;
* neutral-venue support (home advantage only for host nations) — for leagues
  *and* international tournaments;
* the :math:`\rho`/:math:`\tau` low-score correction;
* the bivariate score matrix and 1X2 probabilities;
* validation by Ranked Probability Score and multi-class Brier, temporal
  cross-validation and optimal time-decay selection;
* a Monte-Carlo tournament simulator (e.g. the 2026 World Cup);
* free, key-less data loaders (:mod:`football_mle.sources`).

Typical flow
------------
>>> from football_mle import prepare_data, fit_dixon_coles, predict_match
>>> from football_mle.sources import fetch_league
>>> df = fetch_league("Premier League (England)")
>>> model = fit_dixon_coles(prepare_data(df))
>>> print(predict_match(model, "Arsenal", "Chelsea").summary())
"""
from __future__ import annotations

from .data import (
    CANONICAL_COLUMNS,
    MatchData,
    half_life_from_xi,
    load_csv,
    prepare_data,
    standardize_dataframe,
    time_decay_weights,
    xi_from_half_life,
)
from .dixon_coles import dixon_coles_negative_log_likelihood, fit_dixon_coles
from .fitting import FitResult, fit
from .likelihood import (
    compute_rates,
    negative_log_likelihood,
    pack_parameters,
    score_matrix_from_rates,
    tau_correction,
)
from .maher import fit_maher, maher_negative_log_likelihood
from .prediction import MatchPrediction, outcome_probabilities, predict_match, score_matrix
from .simulation import (
    TournamentResult,
    derive_groups,
    official_groups,
    simulate_match,
    simulate_tournament,
)
from .validation import (
    OUTCOME_ORDER,
    EvaluationResult,
    evaluate,
    multiclass_brier,
    optimize_xi,
    outcome_index,
    rps,
    temporal_cross_validation,
)

__version__ = "2.0.0"

__all__ = [
    # data / Stage 1
    "CANONICAL_COLUMNS",
    "MatchData",
    "load_csv",
    "standardize_dataframe",
    "prepare_data",
    "time_decay_weights",
    "xi_from_half_life",
    "half_life_from_xi",
    # core
    "pack_parameters",
    "compute_rates",
    "tau_correction",
    "negative_log_likelihood",
    "score_matrix_from_rates",
    # models / Stages 2-3
    "fit_maher",
    "maher_negative_log_likelihood",
    "fit_dixon_coles",
    "dixon_coles_negative_log_likelihood",
    "fit",
    "FitResult",
    # prediction / Stage 4
    "MatchPrediction",
    "score_matrix",
    "outcome_probabilities",
    "predict_match",
    # validation / Stage 5
    "rps",
    "multiclass_brier",
    "outcome_index",
    "evaluate",
    "EvaluationResult",
    "temporal_cross_validation",
    "optimize_xi",
    "OUTCOME_ORDER",
    # simulation
    "simulate_match",
    "simulate_tournament",
    "derive_groups",
    "official_groups",
    "TournamentResult",
    "__version__",
]
