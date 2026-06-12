"""
Free, key-less data sources.

* :mod:`football_mle.sources.leagues` — club leagues via football-data.co.uk.
* :mod:`football_mle.sources.international` — national teams and the 2026 World
  Cup via the martj42 international-results dataset.
"""
from __future__ import annotations

from .international import (
    INTERNATIONAL_RESULTS_URL,
    fetch_international_results,
    played_matches,
    world_cup_2026_fixtures,
)
from .leagues import (
    LEAGUES,
    current_season_code,
    fetch_league,
    list_leagues,
    recent_seasons,
    season_code,
)

__all__ = [
    "LEAGUES",
    "list_leagues",
    "season_code",
    "current_season_code",
    "recent_seasons",
    "fetch_league",
    "INTERNATIONAL_RESULTS_URL",
    "fetch_international_results",
    "played_matches",
    "world_cup_2026_fixtures",
]
