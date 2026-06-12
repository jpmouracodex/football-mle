"""
Stage 1 — Data structuring and time-decay weighting.

This module defines the project's canonical data structure and implements the
exponential **time-decay** weighting of Dixon & Coles (1997), which down-weights
old matches in the likelihood.

Canonical column schema (any input ``DataFrame`` is normalized to it):

==============  =========================================================
Column          Description
==============  =========================================================
``date``        match date (``datetime``)            — optional, required if ``xi > 0``
``home``        home team name (``str``)
``away``        away team name (``str``)
``home_goals``  goals scored by the home team (``int >= 0``)
``away_goals``  goals scored by the away team (``int >= 0``)
``neutral``     ``True`` if played at a neutral venue (no home advantage) — optional
==============  =========================================================

The loader auto-detects the most common schemas, including football-data.co.uk
(``Date; HomeTeam; AwayTeam; FTHG; FTAG``), the martj42 international results
(``date; home_team; away_team; home_score; away_score; neutral``) and the
Portuguese "long" format (``Time_da_casa; Gols_feitos_pelo_time_da_casa; ...``).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "MatchData",
    "load_csv",
    "standardize_dataframe",
    "time_decay_weights",
    "xi_from_half_life",
    "half_life_from_xi",
    "prepare_data",
]

# Canonical column order after standardization (``neutral`` is appended when present).
CANONICAL_COLUMNS: tuple[str, ...] = ("date", "home", "away", "home_goals", "away_goals")

# Accepted synonyms per column (normalized before matching). Portuguese variants
# are kept so the loader still reads the reference datasets.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "date": ("date", "data", "day", "dia", "datetime", "matchdate"),
    "home": (
        "home", "hometeam", "homename", "hometeamname", "home_team", "casa",
        "equipacasa", "timedacasa", "mandante",
    ),
    "away": (
        "away", "awayteam", "awayname", "away_team", "fora", "equipafora",
        "timedeforadecasa", "visitante",
    ),
    "home_goals": (
        "homegoals", "fthg", "hg", "homescore", "home_score", "goloscasa",
        "golscasa", "golsfeitospelotimedacasa", "scoredhome",
    ),
    "away_goals": (
        "awaygoals", "ftag", "ag", "awayscore", "away_score", "golosfora",
        "golsfora", "golsfeitospelotimedeforadecasa", "scoredaway",
    ),
    "neutral": ("neutral", "neutro", "neutralvenue"),
}


def _normalize(label: object) -> str:
    """Reduce a column label to lowercase alphanumerics without accents.

    E.g. ``"Gols_feitos_pelo_time_da_casa"`` -> ``"golsfeitospelotimedacasa"``;
    ``"HomeTeam"`` -> ``"hometeam"``; ``"FTHG"`` -> ``"fthg"``.
    """
    raw = unicodedata.normalize("NFKD", str(label))
    raw = raw.encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in raw.lower() if c.isalnum())


# Reverse lookup {normalized synonym -> canonical column}, built once.
_REVERSE_LOOKUP: dict[str, str] = {
    _normalize(syn): canonical
    for canonical, synonyms in _SYNONYMS.items()
    for syn in synonyms
}


def _coerce_neutral(series: pd.Series) -> npt.NDArray[np.bool_]:
    """Convert a heterogeneous neutral-venue column to a boolean array."""
    truthy = {"true", "1", "yes", "y", "t", "sim", "verdadeiro"}
    return series.astype(str).str.strip().str.lower().isin(truthy).to_numpy()


def standardize_dataframe(
    df: pd.DataFrame,
    *,
    column_map: Mapping[str, str] | None = None,
    date_format: str | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Rename/coerce an arbitrary ``DataFrame`` to the canonical schema.

    Parameters
    ----------
    df:
        Match table in any supported convention.
    column_map:
        Explicit mapping ``{canonical -> original_name}``. Takes precedence over
        auto-detection.
    date_format:
        ``strftime`` of the date column (e.g. ``"%d/%m/%Y"``). If ``None``, parsing
        is inferred by pandas.
    dayfirst:
        Passed to :func:`pandas.to_datetime` (football-data.co.uk uses day/month).

    Returns
    -------
    pandas.DataFrame
        Table with :data:`CANONICAL_COLUMNS` (plus ``neutral`` when available;
        ``date`` may hold ``NaT`` if absent).
    """
    if column_map is not None:
        rename = {original: canonical for canonical, original in column_map.items()}
    else:
        rename = {}
        for column in df.columns:
            canonical = _REVERSE_LOOKUP.get(_normalize(column))
            if canonical is not None and canonical not in rename.values():
                rename[column] = canonical

    out = df.rename(columns=rename)

    required = ("home", "away", "home_goals", "away_goals")
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Could not identify columns {missing}. "
            f"Received columns: {list(df.columns)}. "
            "Provide column_map={'home': '...', ...} explicitly."
        )

    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    for column in ("home_goals", "away_goals"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(
            out["date"], format=date_format, dayfirst=dayfirst, errors="coerce"
        )
    else:
        out["date"] = pd.NaT

    columns = list(CANONICAL_COLUMNS)
    if "neutral" in out.columns:
        out["neutral"] = _coerce_neutral(out["neutral"])
        columns = columns + ["neutral"]

    # Drop matches without a score (fixtures not yet played).
    valid = out["home_goals"].notna() & out["away_goals"].notna()
    out = out.loc[valid].copy()
    out["home_goals"] = out["home_goals"].astype(np.int64)
    out["away_goals"] = out["away_goals"].astype(np.int64)

    if (out[["home_goals", "away_goals"]] < 0).to_numpy().any():
        raise ValueError("Negative goals found in the dataset.")

    return out.loc[:, columns].reset_index(drop=True)


def load_csv(
    path: str | Path,
    *,
    column_map: Mapping[str, str] | None = None,
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "utf-8",
    date_format: str | None = None,
    dayfirst: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """Read a CSV and return a ``DataFrame`` in the canonical schema."""
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding, **kwargs)
    return standardize_dataframe(
        df, column_map=column_map, date_format=date_format, dayfirst=dayfirst
    )


# ---------------------------------------------------------------------------
# Time decay (Dixon & Coles, 1997)
# ---------------------------------------------------------------------------
def xi_from_half_life(half_life_days: float) -> float:
    r"""Convert a **half-life** (in days) into the decay rate :math:`\xi`.

    The weight function is :math:`\phi(\Delta t) = e^{-\xi\,\Delta t}`. The
    half-life is the interval after which the weight halves, so
    :math:`\xi = \ln 2 / t_{1/2}`.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    return float(np.log(2.0) / half_life_days)


def half_life_from_xi(xi: float) -> float:
    """Inverse of :func:`xi_from_half_life` (half-life in days for a given ``xi``)."""
    if xi <= 0:
        return float("inf")
    return float(np.log(2.0) / xi)


def time_decay_weights(
    dates: Sequence[datetime] | pd.Series | npt.NDArray,
    reference_date: datetime | str | pd.Timestamp,
    xi: float,
) -> npt.NDArray[np.float64]:
    r"""Exponential decay weight of each match.

    Implements :math:`\phi(t_k) = \exp\{-\xi\,(t_{\text{ref}} - t_k)\}`, with the
    difference measured in **days**. Matches in the future relative to the
    reference get weight 1 (the difference is clipped at 0).
    """
    if xi < 0:
        raise ValueError("xi must be non-negative.")
    dates = pd.to_datetime(pd.Series(np.asarray(dates).ravel()))
    if dates.isna().any():
        raise ValueError("Missing/invalid dates; time decay requires the 'date' column.")
    reference = pd.Timestamp(reference_date)
    delta_days = (reference - dates).dt.total_seconds().to_numpy() / 86_400.0
    delta_days = np.maximum(delta_days, 0.0)
    return np.exp(-xi * delta_days)


# ---------------------------------------------------------------------------
# Structure consumed by the optimizers
# ---------------------------------------------------------------------------
@dataclass
class MatchData:
    """Vectorized, index-based representation of matches for the MLE.

    Team names are mapped to integer indices ``0..n-1`` so the likelihood is
    computed with vectorized ``numpy`` operations. ``neutral`` marks matches
    where no home advantage applies (e.g. tournament games on neutral ground).
    """

    teams: list[str]
    home_index: npt.NDArray[np.int64]
    away_index: npt.NDArray[np.int64]
    home_goals: npt.NDArray[np.int64]
    away_goals: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    neutral: npt.NDArray[np.bool_]
    _index_by_name: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index_by_name = {name: i for i, name in enumerate(self.teams)}

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_matches(self) -> int:
        return int(self.home_goals.shape[0])

    def team_index(self, name: str) -> int:
        try:
            return self._index_by_name[name]
        except KeyError as err:
            raise KeyError(
                f"Team {name!r} is not in the training set. Known teams: {self.teams}"
            ) from err


def prepare_data(
    df: pd.DataFrame,
    *,
    reference_date: datetime | str | pd.Timestamp | None = None,
    xi: float = 0.0,
    teams: Iterable[str] | None = None,
) -> MatchData:
    """Convert a canonical ``DataFrame`` into :class:`MatchData`.

    Parameters
    ----------
    df:
        Already-standardized table (see :func:`standardize_dataframe`).
    reference_date:
        Reference for the decay. If ``None`` and ``xi > 0``, uses the max date.
    xi:
        Time-decay rate. ``0`` => all matches weight 1.
    teams:
        Fixed team list (defines the index order). If ``None``, uses the sorted
        union of observed teams. Passing it explicitly aligns train and test.
    """
    missing = [c for c in ("home", "away", "home_goals", "away_goals") if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is not in canonical schema; missing {missing}.")

    if teams is None:
        teams = sorted(set(df["home"]).union(df["away"]))
    teams = list(teams)
    index = {name: i for i, name in enumerate(teams)}

    home_idx = df["home"].map(index)
    away_idx = df["away"].map(index)
    if home_idx.isna().any() or away_idx.isna().any():
        unknown = set(df["home"]).union(df["away"]) - set(teams)
        raise ValueError(f"Matches with teams outside the given list: {sorted(unknown)}")

    if xi and xi > 0:
        if reference_date is None:
            reference_date = pd.to_datetime(df["date"]).max()
        weights = time_decay_weights(df["date"], reference_date, xi)
    else:
        weights = np.ones(len(df), dtype=np.float64)

    if "neutral" in df.columns:
        neutral = np.asarray(df["neutral"], dtype=bool)
    else:
        neutral = np.zeros(len(df), dtype=bool)

    return MatchData(
        teams=teams,
        home_index=home_idx.to_numpy(dtype=np.int64),
        away_index=away_idx.to_numpy(dtype=np.int64),
        home_goals=df["home_goals"].to_numpy(dtype=np.int64),
        away_goals=df["away_goals"].to_numpy(dtype=np.int64),
        weights=weights,
        neutral=neutral,
    )
