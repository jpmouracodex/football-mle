r"""
Mathematical core — expected rates, the :math:`\tau` correction and the
(weighted, negative) log-likelihood.

Model (multiplicative parametrization of Maher, 1982 / Dixon-Coles, 1997; see also
the applied treatment in Vizzoni, 2018). For a match between home team :math:`i`
and away team :math:`j`, goals are **conditionally** independent Poisson variables:

.. math::

    X_{ij} \sim \mathrm{Poisson}(\lambda_{ij}), \qquad
    Y_{ij} \sim \mathrm{Poisson}(\mu_{ij}),

with rates

.. math::

    \lambda_{ij} = \alpha_i\,\beta_j\,\gamma^{\,\mathbb{1}[\text{not neutral}]},
    \qquad
    \mu_{ij}     = \alpha_j\,\beta_i ,

where :math:`\alpha` is the **attack** strength (higher = scores more),
:math:`\beta` the **defense** factor (higher = concedes more, i.e. weaker
defense) and :math:`\gamma > 1` the **home advantage**. The home advantage is
applied only to non-neutral matches, so the same model serves domestic leagues
and neutral-venue international tournaments.

The Dixon-Coles correction replaces independence by a local dependence on
low scores through the factor :math:`\tau` (see :func:`tau_correction`).

Identifiability
---------------
Rates depend only on the products :math:`\alpha_i\beta_j`. The transformation
:math:`(\alpha, \beta) \mapsto (c\,\alpha,\ \beta/c)` leaves every rate
unchanged — there is thus **one** scale degree of freedom, fixed by the classic
Maher/Dixon-Coles constraint :math:`\tfrac{1}{n}\sum_i \alpha_i = 1`.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from .data import MatchData

__all__ = [
    "Vector",
    "pack_parameters",
    "unpack_parameters",
    "compute_rates",
    "tau_correction",
    "poisson_log_pmf",
    "negative_log_likelihood",
    "negative_log_likelihood_gradient",
    "maher_negative_log_likelihood",
    "dixon_coles_negative_log_likelihood",
    "score_matrix_from_rates",
    "TAU_FLOOR",
]

Vector = npt.NDArray[np.float64]

# Numerical floor to avoid log(<=0) if rho drives tau to non-positive values.
TAU_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# Parameter vector packing
# ---------------------------------------------------------------------------
# Layout: theta = [ attack_0..attack_{n-1} | defense_0..defense_{n-1} | gamma | (rho) ]
def pack_parameters(
    attack: npt.ArrayLike,
    defense: npt.ArrayLike,
    home_advantage: float,
    rho: float | None = None,
) -> Vector:
    """Concatenate the named parameters into the flat ``theta`` the optimizer needs."""
    parts = [
        np.asarray(attack, dtype=np.float64).ravel(),
        np.asarray(defense, dtype=np.float64).ravel(),
        np.array([home_advantage], dtype=np.float64),
    ]
    if rho is not None:
        parts.append(np.array([rho], dtype=np.float64))
    return np.concatenate(parts)


def unpack_parameters(
    theta: Vector, n_teams: int, with_rho: bool
) -> tuple[Vector, Vector, float, float]:
    """Inverse of :func:`pack_parameters`.

    Returns ``(attack, defense, home_advantage, rho)``; ``rho`` is ``0.0`` when
    ``with_rho`` is ``False`` (Maher model).
    """
    attack = theta[:n_teams]
    defense = theta[n_teams : 2 * n_teams]
    home_advantage = float(theta[2 * n_teams])
    rho = float(theta[2 * n_teams + 1]) if with_rho else 0.0
    return attack, defense, home_advantage, rho


# ---------------------------------------------------------------------------
# Expected rates and Dixon-Coles correction
# ---------------------------------------------------------------------------
def compute_rates(
    attack: Vector,
    defense: Vector,
    home_advantage: float,
    home_index: npt.NDArray[np.int64],
    away_index: npt.NDArray[np.int64],
    neutral: npt.NDArray[np.bool_] | None = None,
) -> tuple[Vector, Vector]:
    r"""Per-match expected rates :math:`(\lambda, \mu)`.

    :math:`\lambda = \alpha_{\text{home}}\,\beta_{\text{away}}\,\gamma` and
    :math:`\mu = \alpha_{\text{away}}\,\beta_{\text{home}}`, with :math:`\gamma`
    applied only where the match is **not** at a neutral venue.
    """
    if neutral is None:
        home_factor = home_advantage
    else:
        home_factor = np.where(neutral, 1.0, home_advantage)
    home_rate = attack[home_index] * defense[away_index] * home_factor
    away_rate = attack[away_index] * defense[home_index]
    return home_rate, away_rate


def tau_correction(
    home_goals: npt.ArrayLike,
    away_goals: npt.ArrayLike,
    home_rate: npt.ArrayLike,
    away_rate: npt.ArrayLike,
    rho: float,
) -> Vector:
    r"""Dixon-Coles (1997) dependence factor :math:`\tau_{\lambda,\mu}(x, y)`.

    .. math::

        \tau(x,y) =
        \begin{cases}
            1 - \lambda\mu\rho & (x,y) = (0,0) \\
            1 + \lambda\rho     & (x,y) = (0,1) \\
            1 + \mu\rho         & (x,y) = (1,0) \\
            1 - \rho            & (x,y) = (1,1) \\
            1                   & \text{otherwise.}
        \end{cases}

    The parameter :math:`\rho` models the tactical/psychological dependence at
    low scores: :math:`\rho < 0` raises the probability of ``0-0``/``1-1`` and
    lowers that of ``1-0``/``0-1`` relative to independence. Accepts scalars or
    arrays (including ``meshgrid`` grids), so it is reused to build the score
    matrix.
    """
    x = np.asarray(home_goals)
    y = np.asarray(away_goals)
    lam = np.asarray(home_rate, dtype=np.float64)
    mu = np.asarray(away_rate, dtype=np.float64)

    shape = np.broadcast(x, y, lam, mu).shape
    conditions = [
        (x == 0) & (y == 0),
        (x == 0) & (y == 1),
        (x == 1) & (y == 0),
        (x == 1) & (y == 1),
    ]
    choices = [
        1.0 - lam * mu * rho,
        1.0 + lam * rho,
        1.0 + mu * rho,
        np.full(shape, 1.0 - rho),
    ]
    return np.select(conditions, choices, default=1.0)


def poisson_log_pmf(k: npt.ArrayLike, rate: npt.ArrayLike) -> Vector:
    r"""Log Poisson pmf: :math:`k\ln\theta - \theta - \ln k!`.

    Uses :func:`scipy.special.gammaln` for :math:`\ln k! = \ln\Gamma(k+1)`,
    stable for large counts.
    """
    k = np.asarray(k, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    return k * np.log(rate) - rate - gammaln(k + 1.0)


# ---------------------------------------------------------------------------
# Log-likelihood (negative, weighted)
# ---------------------------------------------------------------------------
def negative_log_likelihood(theta: Vector, data: MatchData, with_rho: bool) -> float:
    r"""Weighted **negative** log-likelihood (the MLE objective).

    .. math::

        -\ell(\theta) = -\sum_{k} w_k \Big[
            \ln \tau(x_k, y_k)
            + \ln \mathrm{Pois}(x_k \mid \lambda_k)
            + \ln \mathrm{Pois}(y_k \mid \mu_k)
        \Big],

    where :math:`w_k` are the time-decay weights. The :math:`\ln\tau` term enters
    only when ``with_rho`` is ``True`` (Dixon-Coles); with ``with_rho=False`` it
    reduces to the Maher (pure Poisson) model. Minimizing this maximizes the
    likelihood.
    """
    n = data.n_teams
    attack, defense, home_advantage, rho = unpack_parameters(theta, n, with_rho)
    home_rate, away_rate = compute_rates(
        attack, defense, home_advantage, data.home_index, data.away_index, data.neutral
    )

    log_prob = poisson_log_pmf(data.home_goals, home_rate)
    log_prob = log_prob + poisson_log_pmf(data.away_goals, away_rate)

    if with_rho:
        tau = tau_correction(data.home_goals, data.away_goals, home_rate, away_rate, rho)
        log_prob = log_prob + np.log(np.maximum(tau, TAU_FLOOR))

    return float(-np.sum(data.weights * log_prob))


def _tau_log_derivatives(
    home_goals: npt.NDArray[np.int64],
    away_goals: npt.NDArray[np.int64],
    home_rate: Vector,
    away_rate: Vector,
    rho: float,
) -> tuple[Vector, Vector, Vector]:
    r"""Partial derivatives of :math:`\ln\tau` w.r.t. :math:`\lambda`, :math:`\mu`, :math:`\rho`.

    Non-zero only on the four low-score cells; zero elsewhere (where :math:`\tau=1`).
    """
    lam, mu = home_rate, away_rate
    tau = np.maximum(tau_correction(home_goals, away_goals, lam, mu, rho), TAU_FLOOR)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(mu)
    d_rho = np.zeros_like(lam)

    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    d_lam[m00] = -mu[m00] * rho / tau[m00]
    d_mu[m00] = -lam[m00] * rho / tau[m00]
    d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]

    d_lam[m01] = rho / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]

    d_mu[m10] = rho / tau[m10]
    d_rho[m10] = mu[m10] / tau[m10]

    d_rho[m11] = -1.0 / tau[m11]
    return d_lam, d_mu, d_rho


def negative_log_likelihood_gradient(theta: Vector, data: MatchData, with_rho: bool) -> Vector:
    r"""Analytic gradient of :func:`negative_log_likelihood` w.r.t. ``theta``.

    Derived from :math:`\partial_\lambda \ell = x/\lambda - 1 + \partial_\lambda\ln\tau`
    (and likewise for :math:`\mu`) via the chain rule. Because each rate is linear
    in the relevant :math:`\alpha`, :math:`\beta` and :math:`\gamma`, the
    per-team sums collapse to weighted residuals gathered with ``np.bincount`` —
    so the whole gradient costs one vectorized pass, regardless of the number of
    teams. Supplying it to the optimizer removes the :math:`O(p)` finite-difference
    cost per iteration (decisive for hundreds of international teams).
    """
    n = data.n_teams
    attack, defense, home_advantage, rho = unpack_parameters(theta, n, with_rho)
    home_goals, away_goals = data.home_goals, data.away_goals
    x = home_goals.astype(np.float64)
    y = away_goals.astype(np.float64)
    w = data.weights
    home_rate, away_rate = compute_rates(
        attack, defense, home_advantage, data.home_index, data.away_index, data.neutral
    )

    if with_rho:
        d_lam, d_mu, d_rho = _tau_log_derivatives(home_goals, away_goals, home_rate, away_rate, rho)
    else:
        d_lam = d_mu = 0.0

    # G = rate * d(loglik)/d(rate); for the Poisson part rate*(x/rate - 1) = x - rate.
    g_home = x - home_rate + home_rate * d_lam
    g_away = y - away_rate + away_rate * d_mu
    wg_home = w * g_home
    wg_away = w * g_away

    sum_attack = np.bincount(data.home_index, weights=wg_home, minlength=n) + np.bincount(
        data.away_index, weights=wg_away, minlength=n
    )
    sum_defense = np.bincount(data.away_index, weights=wg_home, minlength=n) + np.bincount(
        data.home_index, weights=wg_away, minlength=n
    )

    grad_attack = -sum_attack / attack
    grad_defense = -sum_defense / defense
    grad_home_advantage = -np.sum(wg_home * (~data.neutral)) / home_advantage

    parts = [grad_attack, grad_defense, np.array([grad_home_advantage])]
    if with_rho:
        grad_rho = -np.sum(w * d_rho)
        parts.append(np.array([grad_rho]))
    return np.concatenate(parts)


def maher_negative_log_likelihood(theta: Vector, data: MatchData) -> float:
    """Stage 2 — Maher (pure Poisson) negative log-likelihood."""
    return negative_log_likelihood(theta, data, with_rho=False)


def dixon_coles_negative_log_likelihood(theta: Vector, data: MatchData) -> float:
    """Stage 3 — Dixon-Coles negative log-likelihood (with ``rho``)."""
    return negative_log_likelihood(theta, data, with_rho=True)


# ---------------------------------------------------------------------------
# Bivariate score matrix from rates (reused in Stage 4 and the simulator)
# ---------------------------------------------------------------------------
def score_matrix_from_rates(
    home_rate: float,
    away_rate: float,
    rho: float = 0.0,
    max_goals: int = 10,
    normalize: bool = True,
) -> Vector:
    r"""Matrix :math:`P(X=x, Y=y)` for :math:`x, y \in \{0, \dots, \text{max\_goals}\}`.

    Combines the outer product of the Poisson marginals with the :math:`\tau`
    correction on the four low-score cells. If ``normalize`` is ``True`` the
    matrix is rescaled to sum to 1 (truncation at ``max_goals`` and the
    Dixon-Coles correction slightly change the total mass).
    """
    counts = np.arange(max_goals + 1)
    home_marginal = np.exp(poisson_log_pmf(counts, home_rate))
    away_marginal = np.exp(poisson_log_pmf(counts, away_rate))
    matrix = np.outer(home_marginal, away_marginal)

    grid_home, grid_away = np.meshgrid(counts, counts, indexing="ij")
    matrix = matrix * tau_correction(grid_home, grid_away, home_rate, away_rate, rho)

    matrix = np.maximum(matrix, 0.0)  # tau<0 in extreme cells must not create negative mass
    total = matrix.sum()
    if normalize and total > 0:
        matrix = matrix / total
    return matrix
