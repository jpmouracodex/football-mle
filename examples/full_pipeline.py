r"""
End-to-end pipeline covering the project's 5 stages.

Runs:
  1. data structuring and time-decay weighting;
  2. Maher (pure Poisson) MLE fit;
  3. Dixon-Coles (with rho) MLE fit;
  4. score matrix + 1X2 probabilities of a fixture;
  5. out-of-sample validation by RPS / Brier, comparing the models.

Usage:
    python examples/full_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_mle import (  # noqa: E402
    evaluate,
    fit_dixon_coles,
    fit_maher,
    half_life_from_xi,
    predict_match,
    prepare_data,
    score_matrix,
    xi_from_half_life,
)
from examples.synthetic_data import generate_synthetic_league  # noqa: E402

WIDTH = 78


def header(text: str) -> None:
    print("\n" + "=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def main() -> None:
    # -- STAGE 1: data + time decay -----------------------------------------
    header("STAGE 1 — Data structuring and time decay")
    league = generate_synthetic_league(n_teams=20, seed=7)
    matches = league.matches
    print(f"Synthetic league: {len(matches)} matches, {len(league.teams)} teams.")
    print(matches.head(5).to_string(index=False))

    # Chronological train/test split (last ~15% of rounds for testing).
    cutoff = matches["date"].quantile(0.85)
    train = matches[matches["date"] <= cutoff].copy()
    test = matches[matches["date"] > cutoff].copy()
    print(f"\nTrain: {len(train)} matches | Test: {len(test)} matches (cutoff {cutoff.date()}).")

    xi = xi_from_half_life(180)
    reference = train["date"].max()

    # Two views of the SAME training set: undecayed (weights 1) and decayed.
    # The Maher vs Dixon-Coles comparison always uses the same weights so the
    # log-likelihoods are comparable (the nesting LL(DC) >= LL(Maher) only holds
    # under identical weights).
    data_plain = prepare_data(train, xi=0.0, teams=league.teams)
    data_decay = prepare_data(train, reference_date=reference, xi=xi, teams=league.teams)
    print(f"\nxi (half-life {half_life_from_xi(xi):.0f} days) = {xi:.5f} per day")
    print(
        f"Decay weights: min={data_decay.weights.min():.3f} (oldest match) … "
        f"max={data_decay.weights.max():.3f} (most recent)."
    )

    # -- STAGE 2: Maher -----------------------------------------------------
    header("STAGE 2 — Maher (pure Poisson) MLE")
    maher = fit_maher(data_plain, compute_se=True)
    print(maher.summary())

    attack_corr = np.corrcoef(maher.attack, league.true_attack)[0, 1]
    defense_corr = np.corrcoef(maher.defense, league.true_defense)[0, 1]
    print(f"\nParameter recovery (corr. with truth): attack={attack_corr:.3f}, defense={defense_corr:.3f}")
    print(f"Home advantage: estimated={maher.home_advantage:.3f} | true={league.true_home_advantage:.3f}")

    # -- STAGE 3: Dixon-Coles ----------------------------------------------
    header("STAGE 3 — Dixon-Coles (rho correction) MLE")
    dixon_coles = fit_dixon_coles(data_plain, compute_se=True)  # SAME weights as Maher
    print(dixon_coles.summary())
    print(f"\nrho: estimated={dixon_coles.rho:+.4f} | true={league.true_rho:+.4f}")

    gain = dixon_coles.log_likelihood - maher.log_likelihood
    print(
        "Nesting (same data/weights): Dixon-Coles contains Maher at rho=0, so "
        f"LL(DC) >= LL(Maher).\n"
        f"  LL(Maher) = {maher.log_likelihood:.2f} | LL(DC) = {dixon_coles.log_likelihood:.2f} | "
        f"gain = {gain:+.2f} (1 extra parameter)."
    )

    # -- STAGE 4: score matrix + 1X2 ----------------------------------------
    header("STAGE 4 — Score matrix and 1X2")
    ranking = dixon_coles.ratings_table()
    home = ranking.iloc[0]["team"]
    away = ranking.iloc[-1]["team"]
    prediction = predict_match(dixon_coles, home, away, max_goals=10)
    print(prediction.summary())

    matrix = score_matrix(dixon_coles, home, away, max_goals=5)
    print("\nScore matrix P(home=row, away=col), up to 5 goals (%):")
    labels = [str(k) for k in range(matrix.shape[0])]
    table = pd.DataFrame(100 * matrix, index=labels, columns=labels)
    print(table.round(2).to_string())

    # -- STAGE 5: out-of-sample validation ----------------------------------
    header("STAGE 5 — Validation (RPS / Brier) on the test set")
    dixon_coles_decay = fit_dixon_coles(data_decay)  # time-decay variant
    eval_maher = evaluate(maher, test)
    eval_dc = evaluate(dixon_coles, test)
    eval_dc_decay = evaluate(dixon_coles_decay, test)
    for result in (eval_maher, eval_dc, eval_dc_decay):
        label = "Dixon-Coles + time-decay" if result is eval_dc_decay else result.model
        print(result.summary().replace(result.model, label, 1))
        print()

    best = min((eval_maher, eval_dc, eval_dc_decay), key=lambda e: e.mean_rps)
    print(f"Lowest RPS (best 1X2 calibration) on this test: {best.model}.")
    print(
        "\nCritical reading:\n"
        "  * Dixon-Coles' gain concentrates in the SCORE distribution (and the\n"
        "    log-likelihood); on the aggregated 1X2 market rho's effect is small,\n"
        "    since it only redistributes mass between 0-0/1-1 and 1-0/0-1.\n"
        "  * In this synthetic league strengths are CONSTANT in time, so time\n"
        "    decay does not help (no drift to exploit). Its value shows up on real\n"
        "    data, where teams change form across the season."
    )


if __name__ == "__main__":
    main()
