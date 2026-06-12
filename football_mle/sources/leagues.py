"""
Free club-league data from football-data.co.uk (no API key required).

Each league maps to a division code; seasons use the site's ``YYZZ`` convention
(e.g. ``"2526"`` for 2025-26). The CSVs carry ``Date, HomeTeam, AwayTeam, FTHG,
FTAG`` (among many betting-odds columns), which the canonical loader extracts.
"""
from __future__ import annotations

import datetime as _dt
import warnings

import pandas as pd

from ..data import standardize_dataframe
from ._http import read_csv_url

__all__ = [
    "LEAGUES",
    "list_leagues",
    "season_code",
    "current_season_code",
    "recent_seasons",
    "fetch_league",
]

_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Display name -> football-data.co.uk division code.
LEAGUES: dict[str, str] = {
    "Premier League (England)": "E0",
    "Championship (England)": "E1",
    "La Liga (Spain)": "SP1",
    "Serie A (Italy)": "I1",
    "Bundesliga (Germany)": "D1",
    "Ligue 1 (France)": "F1",
    "Eredivisie (Netherlands)": "N1",
    "Primeira Liga (Portugal)": "P1",
    "Pro League (Belgium)": "B1",
    "Scottish Premiership": "SC0",
    "Super Lig (Turkey)": "T1",
    "Super League (Greece)": "G1",
}


def list_leagues() -> list[str]:
    """Display names of the supported leagues."""
    return list(LEAGUES)


def season_code(start_year: int) -> str:
    """Season code for a season starting in ``start_year`` (e.g. 2025 -> ``"2526"``)."""
    yy = start_year % 100
    return f"{yy:02d}{(yy + 1) % 100:02d}"


def current_season_code(today: _dt.date | None = None) -> str:
    """Most recent season code given a date (seasons are assumed to start in July)."""
    today = today or _dt.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return season_code(start_year)


def recent_seasons(n: int = 2, today: _dt.date | None = None) -> list[str]:
    """The ``n`` most recent season codes, oldest first."""
    today = today or _dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [season_code(start - i) for i in range(n)][::-1]


def fetch_league(
    league: str,
    seasons: list[str] | None = None,
    *,
    n_seasons: int = 2,
) -> pd.DataFrame:
    """Fetch and combine one or more seasons of a league in the canonical schema.

    Parameters
    ----------
    league:
        A display name from :data:`LEAGUES` or a raw division code (e.g. ``"E0"``).
    seasons:
        Explicit list of season codes. If ``None``, uses the ``n_seasons`` most
        recent seasons (more data + time decay = more stable, current ratings).

    Returns
    -------
    pandas.DataFrame
        Canonical-schema matches sorted by date. Seasons that fail to download are
        skipped with a warning.
    """
    code = LEAGUES.get(league, league)
    seasons = seasons or recent_seasons(n_seasons)

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for season in seasons:
        url = _BASE_URL.format(season=season, code=code)
        try:
            raw = read_csv_url(url, encoding="latin1")
            frames.append(standardize_dataframe(raw, dayfirst=True))
        except Exception as err:  # network / 404 / parsing
            errors.append(f"{season}: {err!r}")

    if not frames:
        raise RuntimeError(f"Could not download league {league!r}. Errors: {errors}")
    if errors:
        warnings.warn(f"Some seasons of {league!r} were skipped: {errors}", stacklevel=2)

    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
