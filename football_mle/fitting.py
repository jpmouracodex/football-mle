r"""
Maximum Likelihood Estimation (MLE) engine.

This is the shared optimization harness for Stages 2 (Maher) and 3 (Dixon-Coles).
Estimation uses :func:`scipy.optimize.minimize` with the **SLSQP** method, which
handles simultaneously:

* **bounds** — positivity of attack, defense and home advantage;
* an **equality constraint** :math:`\tfrac{1}{n}\sum_i \alpha_i = 1`, the
  Maher/Dixon-Coles identifiability condition.

Optionally, asymptotic **standard errors** are derived from the observed
information matrix (numerical Hessian of the negative log-likelihood at the
optimum), using a pseudo-inverse to absorb the constrained scale direction.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from .data import MatchData
from .likelihood import (
    Vector,
    negative_log_likelihood,
    negative_log_likelihood_gradient,
    pack_parameters,
    unpack_parameters,
)

__all__ = ["FitResult", "fit"]

# Above this many teams the SLSQP dense QP (O(p^3) per iteration) becomes the
# bottleneck, so we switch to limited-memory L-BFGS-B.
_SLSQP_MAX_TEAMS = 50


@dataclass
class FitResult:
    r"""Estimated parameters and diagnostics of a fitted model.

    Attributes
    ----------
    model:
        Model name (``"Maher (pure Poisson)"`` or ``"Dixon-Coles"``).
    teams:
        Team list in index order.
    attack, defense:
        Vectors :math:`\alpha` and :math:`\beta` (one value per team).
    home_advantage:
        Home-field multiplier :math:`\gamma`.
    rho:
        Dixon-Coles dependence parameter (``0.0`` in the Maher model).
    log_likelihood:
        Value of the (weighted) log-likelihood at the optimum.
    attack_se, defense_se, home_advantage_se, rho_se:
        Asymptotic standard errors (``None`` unless ``compute_se=True``).
    """

    model: str
    teams: list[str]
    attack: Vector
    defense: Vector
    home_advantage: float
    rho: float
    log_likelihood: float
    converged: bool
    n_evaluations: int
    message: str
    attack_se: Vector | None = None
    defense_se: Vector | None = None
    home_advantage_se: float | None = None
    rho_se: float | None = None
    scipy_result: OptimizeResult | None = field(default=None, repr=False)

    # -- access by name ------------------------------------------------------
    def index(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError as err:
            raise KeyError(f"Team {team!r} is not in the model. Known: {self.teams}") from err

    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        r"""Expected :math:`(\lambda, \mu)` for a fixture.

        Set ``neutral=True`` to drop the home advantage (e.g. tournament games on
        neutral ground).
        """
        i, j = self.index(home), self.index(away)
        gamma = 1.0 if neutral else self.home_advantage
        home_rate = float(self.attack[i] * self.defense[j] * gamma)
        away_rate = float(self.attack[j] * self.defense[i])
        return home_rate, away_rate

    # -- reports -------------------------------------------------------------
    def ratings_table(self) -> pd.DataFrame:
        """Table ``team | attack | defense`` (+ SE columns), sorted by attack."""
        table = pd.DataFrame(
            {"team": self.teams, "attack": self.attack, "defense": self.defense}
        )
        if self.attack_se is not None:
            table["attack_se"] = self.attack_se
            table["defense_se"] = self.defense_se
        return table.sort_values("attack", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        """Readable text summary of parameters and diagnostics."""
        table = self.ratings_table()
        gamma_line = f"  Home advantage (gamma): {self.home_advantage:.4f}"
        if self.home_advantage_se is not None:
            gamma_line += f"  (SE {self.home_advantage_se:.4f})"
        rho_line = f"  Dependence (rho)      : {self.rho:+.4f}"
        if self.rho_se is not None:
            rho_line += f"  (SE {self.rho_se:.4f})"
        lines = [
            f"Model: {self.model}",
            gamma_line,
            rho_line,
            f"  Log-likelihood        : {self.log_likelihood:.2f}",
            f"  Converged             : {self.converged} ({self.message})",
            "",
            "  (attack: higher = scores more | defense: lower = concedes less)",
            "  " + table.to_string(index=False).replace("\n", "\n  "),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimization harness
# ---------------------------------------------------------------------------
def _mean_attack_constraint(n: int) -> dict[str, object]:
    """Equality constraint ``mean(attack) - 1 = 0`` with analytic jacobian."""

    def function(theta: Vector) -> float:
        return float(np.mean(theta[:n]) - 1.0)

    def jacobian(theta: Vector) -> Vector:
        g = np.zeros_like(theta)
        g[:n] = 1.0 / n
        return g

    return {"type": "eq", "fun": function, "jac": jacobian}


def _numerical_hessian(func, x: Vector, rel_step: float = 1e-4) -> Vector:
    """Central-difference Hessian of a scalar function (observed information)."""
    n = x.size
    h = rel_step * (np.abs(x) + 1.0)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h[i]; xpp[j] += h[j]
            xpm[i] += h[i]; xpm[j] -= h[j]
            xmp[i] -= h[i]; xmp[j] += h[j]
            xmm[i] -= h[i]; xmm[j] -= h[j]
            value = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4.0 * h[i] * h[j])
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _standard_errors(theta_hat: Vector, data: MatchData, with_rho: bool) -> Vector:
    """Asymptotic SEs from the inverse observed-information matrix.

    The mean-attack constraint makes the Hessian singular along the scale
    direction; a pseudo-inverse yields finite SEs on the identified subspace and
    zero variance along the (fixed) constrained direction.
    """
    hessian = _numerical_hessian(
        lambda t: negative_log_likelihood(t, data, with_rho), theta_hat
    )
    covariance = np.linalg.pinv(hessian)
    variances = np.clip(np.diag(covariance), 0.0, None)
    return np.sqrt(variances)


def fit(
    data: MatchData,
    *,
    with_rho: bool,
    model_name: str,
    x0: Vector | None = None,
    rho_bounds: tuple[float, float] = (-0.25, 0.25),
    home_advantage_init: float = 1.35,
    rho_init: float = -0.05,
    method: str = "auto",
    maxiter: int = 1000,
    ftol: float = 1e-9,
    compute_se: bool = False,
    verbose: bool = False,
) -> FitResult:
    r"""Estimate parameters by MLE via ``scipy.optimize.minimize`` (SLSQP).

    Parameters
    ----------
    data:
        Vectorized matches (see :func:`football_mle.data.prepare_data`).
    with_rho:
        Include the Dixon-Coles :math:`\rho` (Stage 3) if ``True``; otherwise fit
        the Maher model (Stage 2).
    model_name:
        Descriptive label stored in the result.
    x0:
        Optional initial ``theta`` (e.g. to warm-start Dixon-Coles from Maher).
    rho_bounds:
        Search box for :math:`\rho`; kept modest so :math:`\tau > 0` on usual rates.
    method:
        ``"auto"`` (default) uses SLSQP with the explicit constraint for small
        squads and L-BFGS-B (constraint via renormalization) beyond
        ``_SLSQP_MAX_TEAMS``; force with ``"slsqp"`` or ``"lbfgsb"``.
    compute_se:
        If ``True``, also compute asymptotic standard errors (extra cost,
        :math:`O(p^2)` likelihood evaluations).

    Returns
    -------
    FitResult
    """
    n = data.n_teams

    if x0 is None:
        attack0 = np.ones(n)
        defense0 = np.ones(n)
        parts = [attack0, defense0, [home_advantage_init]]
        if with_rho:
            parts.append([rho_init])
        x0 = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])

    bounds: list[tuple[float, float]] = [(1e-3, 50.0)] * (2 * n)  # attack and defense
    bounds.append((1e-3, 50.0))  # home advantage
    if with_rho:
        bounds.append(rho_bounds)

    use_lbfgsb = method == "lbfgsb" or (method == "auto" and n > _SLSQP_MAX_TEAMS)

    with warnings.catch_warnings():
        # SLSQP occasionally probes slightly outside the bounds during the line
        # search; scipy clips and warns. Harmless for the result.
        warnings.filterwarnings(
            "ignore", message="Values in x were outside bounds", category=RuntimeWarning
        )
        if use_lbfgsb:
            # L-BFGS-B scales to thousands of parameters. It carries no equality
            # constraint, so mean(attack)=1 is enforced afterwards by the
            # (likelihood-invariant) renormalization below — same optimum.
            result = minimize(
                fun=negative_log_likelihood,
                x0=x0,
                args=(data, with_rho),
                jac=negative_log_likelihood_gradient,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "maxfun": 100_000, "ftol": 1e-10, "gtol": 1e-7},
            )
        else:
            result = minimize(
                fun=negative_log_likelihood,
                x0=x0,
                args=(data, with_rho),
                jac=negative_log_likelihood_gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=[_mean_attack_constraint(n)],
                options={"maxiter": maxiter, "ftol": ftol, "disp": verbose},
            )

    attack, defense, home_advantage, rho = unpack_parameters(result.x, n, with_rho)
    attack = np.array(attack, dtype=np.float64)
    defense = np.array(defense, dtype=np.float64)

    # Final normalization: enforce mean(attack)=1 exactly without changing the
    # rates (hence the likelihood), since (attack, defense) -> (attack/c, defense*c)
    # leaves every product invariant.
    c = float(np.mean(attack))
    if c > 0:
        attack = attack / c
        defense = defense * c

    attack_se = defense_se = home_advantage_se = rho_se = None
    if compute_se:
        theta_hat = pack_parameters(
            attack, defense, home_advantage, rho if with_rho else None
        )
        se = _standard_errors(theta_hat, data, with_rho)
        attack_se = se[:n]
        defense_se = se[n : 2 * n]
        home_advantage_se = float(se[2 * n])
        rho_se = float(se[2 * n + 1]) if with_rho else None

    return FitResult(
        model=model_name,
        teams=list(data.teams),
        attack=attack,
        defense=defense,
        home_advantage=float(home_advantage),
        rho=float(rho),
        log_likelihood=float(-result.fun),
        converged=bool(result.success),
        n_evaluations=int(result.get("nfev", -1)),
        message=str(result.message),
        attack_se=attack_se,
        defense_se=defense_se,
        home_advantage_se=home_advantage_se,
        rho_se=rho_se,
        scipy_result=result,
    )
