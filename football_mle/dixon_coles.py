r"""
Stage 3 — Dixon-Coles model (1997).

Extends Maher with the dependence parameter :math:`\rho`, which corrects the
**joint** probability of the four low-score outcomes (``0-0``, ``1-0``, ``0-1``,
``1-1``) through the factor :math:`\tau` (see
:func:`football_mle.likelihood.tau_correction`). This captures the
tactical/psychological dependence that the pure-Poisson independence assumption
misses — empirically there are more low-scoring draws than independence predicts.

The likelihood is Maher's plus the term :math:`\ln\tau(x_k, y_k)` per match. The
optimization reuses the :mod:`football_mle.fitting` harness and can be warm-started
from the Maher solution to speed up and stabilize convergence.
"""
from __future__ import annotations

from .data import MatchData
from .fitting import FitResult, fit
from .likelihood import dixon_coles_negative_log_likelihood, pack_parameters, tau_correction
from .maher import fit_maher

__all__ = ["dixon_coles_negative_log_likelihood", "tau_correction", "fit_dixon_coles"]


def fit_dixon_coles(
    data: MatchData,
    *,
    warm_start_with_maher: bool = True,
    rho_bounds: tuple[float, float] = (-0.25, 0.25),
    rho_init: float = -0.05,
    home_advantage_init: float = 1.35,
    method: str = "auto",
    maxiter: int = 1000,
    compute_se: bool = False,
    verbose: bool = False,
) -> FitResult:
    r"""Fit the Dixon-Coles model by MLE (attack, defense, :math:`\gamma`, :math:`\rho`).

    Parameters
    ----------
    warm_start_with_maher:
        If ``True`` (default), fit Maher first and use its solution as the
        starting point. Since Dixon-Coles contains Maher as a special case
        (:math:`\rho = 0`), this guarantees the final log-likelihood is **not**
        lower than Maher's.
    rho_bounds:
        Search box for :math:`\rho`. Real-data values typically sit near
        ``-0.13``; the default ``(-0.25, 0.25)`` keeps :math:`\tau > 0` on usual rates.
    """
    x0 = None
    if warm_start_with_maher:
        base = fit_maher(
            data, home_advantage_init=home_advantage_init, method=method, maxiter=maxiter
        )
        x0 = pack_parameters(base.attack, base.defense, base.home_advantage, rho=rho_init)

    return fit(
        data,
        with_rho=True,
        model_name="Dixon-Coles",
        x0=x0,
        rho_bounds=rho_bounds,
        rho_init=rho_init,
        home_advantage_init=home_advantage_init,
        method=method,
        maxiter=maxiter,
        compute_se=compute_se,
        verbose=verbose,
    )
