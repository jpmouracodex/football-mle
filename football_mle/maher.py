r"""
Stage 2 — Pure Poisson model (Maher, 1982).

Each score is modeled as two **independent** Poisson variables whose rates depend
on the team attack, the opponent defense and the home advantage:

.. math::

    \lambda = \alpha_{\text{home}}\,\beta_{\text{away}}\,\gamma, \qquad
    \mu = \alpha_{\text{away}}\,\beta_{\text{home}}.

Estimation is by maximum likelihood with the identifiability constraint
:math:`\frac1n\sum_i \alpha_i = 1`. This is the baseline the Dixon-Coles
correction (Stage 3) builds upon. Relative to a textbook Poisson GLM (which
estimates the same attacks/defenses via ``glm(... family = poisson)``), here the
likelihood is built and optimized explicitly, enabling time weighting and the
:math:`\rho` dependence term.
"""
from __future__ import annotations

from .data import MatchData
from .fitting import FitResult, fit
from .likelihood import maher_negative_log_likelihood

__all__ = ["maher_negative_log_likelihood", "fit_maher"]


def fit_maher(
    data: MatchData,
    *,
    home_advantage_init: float = 1.35,
    method: str = "auto",
    maxiter: int = 1000,
    compute_se: bool = False,
    verbose: bool = False,
) -> FitResult:
    """Fit the Maher (pure Poisson) model by MLE.

    Returns a :class:`FitResult` with :math:`\\alpha`, :math:`\\beta`,
    :math:`\\gamma` and ``rho = 0``.
    """
    return fit(
        data,
        with_rho=False,
        model_name="Maher (pure Poisson)",
        home_advantage_init=home_advantage_init,
        method=method,
        maxiter=maxiter,
        compute_se=compute_se,
        verbose=verbose,
    )
