"""
Free national-team data from the martj42/international_results dataset.

This GitHub CSV (no API key) holds every men's international since 1872 with a
``neutral`` flag, and — remarkably — already lists the **2026 World Cup group
fixtures**, from which the groups can be reconstructed.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from ._http import read_csv_url

__all__ = [
    "INTERNATIONAL_RESULTS_URL",
    "fetch_international_results",
    "played_matches",
    "world_cup_2026_fixtures",
]

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

_RENAME = {
    "home_team": "home",
    "away_team": "away",
    "home_score": "home_goals",
    "away_score": "away_goals",
}


def fetch_international_results() -> pd.DataFrame:
    """Download all international results (including unplayed scheduled fixtures).

    Returns columns ``date, home, away, home_goals, away_goals, neutral,
    tournament``. Unplayed fixtures have ``NaN`` goals.
    """
    raw = read_csv_url(INTERNATIONAL_RESULTS_URL, encoding="utf-8")
    df = raw.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["neutral"] = df["neutral"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    for column in ("home_goals", "away_goals"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[["date", "home", "away", "home_goals", "away_goals", "neutral", "tournament"]]


def played_matches(
    df: pd.DataFrame,
    *,
    since: str | _dt.date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only played matches (non-null scores), optionally from ``since`` onward.

    The result is a canonical-schema training table (``int`` goals, ``neutral``).
    """
    out = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if since is not None:
        out = out[out["date"] >= pd.Timestamp(since)]
    out["home_goals"] = out["home_goals"].astype("int64")
    out["away_goals"] = out["away_goals"].astype("int64")
    return out[["date", "home", "away", "home_goals", "away_goals", "neutral"]].reset_index(drop=True)


def world_cup_2026_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 2026 World Cup **group-stage** fixtures (``home, away, neutral``).

    Robust to future additions of knockout fixtures: matches are taken in date
    order and each team is capped at its 3 group games, which isolates exactly the
    group stage (12 groups × 6 = 72 matches).
    """
    mask = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)
    wc = df.loc[mask].sort_values("date")

    appearances: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for _, row in wc.iterrows():
        home, away = row["home"], row["away"]
        if appearances.get(home, 0) < 3 and appearances.get(away, 0) < 3:
            rows.append(
                {"date": row["date"], "home": home, "away": away, "neutral": bool(row["neutral"])}
            )
            appearances[home] = appearances.get(home, 0) + 1
            appearances[away] = appearances.get(away, 0) + 1

    return pd.DataFrame(rows, columns=["date", "home", "away", "neutral"]).reset_index(drop=True)
