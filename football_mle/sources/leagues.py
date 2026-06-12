"""
Free club-league data (no API key required).

Most leagues come from football-data.co.uk, whose CSVs use the ``YYZZ`` season
convention (e.g. ``"2526"``) and carry ``Date, HomeTeam, AwayTeam, FTHG, FTAG``.
The **Brasileirão** is not on that site, so it is fetched from ESPN's public,
key-less JSON API (which also serves live, in-season results).
"""
from __future__ import annotations

import datetime as _dt
import warnings

import pandas as pd

from ..data import standardize_dataframe
from ._http import read_csv_url, read_json_url

__all__ = [
    "LEAGUES",
    "list_leagues",
    "season_code",
    "current_season_code",
    "recent_seasons",
    "fetch_league",
]

_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
_ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
    "?dates={year}0101-{year}1231&limit=1000"
)

# Display name -> football-data.co.uk division code (European leagues).
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

# Display name -> ESPN soccer league code (calendar-year seasons, live results).
_ESPN_LEAGUES: dict[str, str] = {
    "Brasileirão (Brazil)": "bra.1",
}


def list_leagues() -> list[str]:
    """Display names of the supported leagues (Brasileirão first)."""
    return list(_ESPN_LEAGUES) + list(LEAGUES)


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


def _fetch_espn(code: str, n_years: int, today: _dt.date | None = None) -> pd.DataFrame:
    """Fetch the ``n_years`` most recent calendar years of an ESPN league."""
    today = today or _dt.date.today()
    years = [today.year - i for i in range(n_years)]

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for year in years:
        try:
            data = read_json_url(_ESPN_URL.format(code=code, year=year))
            rows: list[dict[str, object]] = []
            for event in data.get("events", []):
                competition = event["competitions"][0]
                if not competition["status"]["type"].get("completed"):
                    continue  # keep only finished matches (real results)
                competitors = competition["competitors"]
                home = next(c for c in competitors if c["homeAway"] == "home")
                away = next(c for c in competitors if c["homeAway"] == "away")
                rows.append(
                    {
                        "date": event["date"][:10],
                        "home": home["team"]["displayName"],
                        "away": away["team"]["displayName"],
                        "home_goals": int(home["score"]),
                        "away_goals": int(away["score"]),
                    }
                )
            if rows:
                frames.append(standardize_dataframe(pd.DataFrame(rows)))
        except Exception as err:  # network / shape change
            errors.append(f"{year}: {err!r}")

    if not frames:
        raise RuntimeError(f"Could not download ESPN league {code!r}. Errors: {errors}")
    if errors:
        warnings.warn(f"Some seasons of {code!r} were skipped: {errors}", stacklevel=2)
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_league(
    league: str,
    seasons: list[str] | None = None,
    *,
    n_seasons: int = 2,
) -> pd.DataFrame:
    """Fetch and combine recent seasons of a league in the canonical schema.

    Parameters
    ----------
    league:
        A display name from :func:`list_leagues` (or a raw football-data.co.uk
        division code). The Brasileirão is routed to the ESPN source.
    seasons:
        Explicit football-data.co.uk season codes. If ``None``, the ``n_seasons``
        most recent are used (ignored for ESPN leagues, which use calendar years).

    Returns
    -------
    pandas.DataFrame
        Canonical-schema matches sorted by date. Seasons that fail to download are
        skipped with a warning.
    """
    if league in _ESPN_LEAGUES:
        return _fetch_espn(_ESPN_LEAGUES[league], n_seasons)

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
