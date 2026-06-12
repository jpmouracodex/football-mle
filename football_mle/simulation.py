r"""
Monte-Carlo tournament simulation.

Given a fitted model and the list of group-stage fixtures, this module simulates
a full tournament many times to estimate, per team, the probabilities of winning
the group, advancing, and reaching each knockout round up to the title.

The group stage is simulated **faithfully** (real fixtures, round-robin standings
with points → goal difference → goals-for tie-breaking, top-2 per group plus the
best third-placed teams — the 2026 World Cup rule). The knockout bracket is a
**strength-seeded single elimination**: a documented approximation of the
official slotting, with draws resolved by the model's relative win probability
(a proxy for extra time / penalties).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fitting import FitResult
from .likelihood import score_matrix_from_rates
from .prediction import outcome_probabilities

__all__ = ["TournamentResult", "simulate_match", "simulate_tournament", "derive_groups"]


# ---------------------------------------------------------------------------
# Fast, cached match sampler
# ---------------------------------------------------------------------------
class _MatchSampler:
    """Caches each fixture's cumulative score distribution for fast sampling."""

    def __init__(self, fit_result: FitResult, max_goals: int = 10) -> None:
        self.fit = fit_result
        self.max_goals = max_goals
        self.size = max_goals + 1
        self._cache: dict[tuple[str, str, bool], tuple[np.ndarray, tuple[float, float, float]]] = {}

    def _distribution(self, home: str, away: str, neutral: bool):
        key = (home, away, neutral)
        cached = self._cache.get(key)
        if cached is None:
            home_rate, away_rate = self.fit.rates(home, away, neutral=neutral)
            matrix = score_matrix_from_rates(
                home_rate, away_rate, self.fit.rho, self.max_goals, normalize=True
            )
            cumulative = np.cumsum(matrix.ravel())
            cumulative[-1] = 1.0
            cached = (cumulative, outcome_probabilities(matrix))
            self._cache[key] = cached
        return cached

    def sample_score(self, home: str, away: str, neutral: bool, rng: np.random.Generator) -> tuple[int, int]:
        cumulative, _ = self._distribution(home, away, neutral)
        index = int(np.searchsorted(cumulative, rng.random()))
        return divmod(index, self.size)

    def knockout_winner(self, a: str, b: str, neutral: bool, rng: np.random.Generator) -> str:
        """Single-elimination winner; draws resolved by relative win probability."""
        home_goals, away_goals = self.sample_score(a, b, neutral, rng)
        if home_goals > away_goals:
            return a
        if away_goals > home_goals:
            return b
        _, (p_home, _, p_away) = self._distribution(a, b, neutral)
        decisive = p_home + p_away
        prob_a = p_home / decisive if decisive > 0 else 0.5
        return a if rng.random() < prob_a else b


def simulate_match(
    fit_result: FitResult,
    home: str,
    away: str,
    *,
    neutral: bool = False,
    seed: int | None = None,
    max_goals: int = 10,
) -> tuple[int, int]:
    """Sample a single score ``(home_goals, away_goals)`` from the fitted model."""
    rng = np.random.default_rng(seed)
    return _MatchSampler(fit_result, max_goals).sample_score(home, away, neutral, rng)


# ---------------------------------------------------------------------------
# Group derivation and bracket seeding
# ---------------------------------------------------------------------------
def derive_groups(group_fixtures: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str, bool]]]]:
    """Reconstruct groups and per-group fixtures from a group-stage fixture list.

    Teams that face each other in the group stage form connected components; in a
    round-robin each component is exactly one group. Groups are labelled ``A, B,
    ...`` ordered by their alphabetically-first team. Returns
    ``(groups, fixtures_by_group)``.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    teams: set[str] = set()
    fixtures: list[tuple[str, str, bool]] = []
    has_neutral = "neutral" in group_fixtures.columns

    for _, row in group_fixtures.iterrows():
        home, away = row["home"], row["away"]
        neutral = bool(row["neutral"]) if has_neutral else False
        adjacency[home].add(away)
        adjacency[away].add(home)
        teams.update((home, away))
        fixtures.append((home, away, neutral))

    seen: set[str] = set()
    components: list[list[str]] = []
    for team in sorted(teams):
        if team in seen:
            continue
        stack, component = [team], set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.add(node)
            stack.extend(adjacency[node] - seen)
        components.append(sorted(component))
    components.sort(key=lambda c: c[0])

    groups: dict[str, list[str]] = {}
    fixtures_by_group: dict[str, list[tuple[str, str, bool]]] = {}
    team_to_label: dict[str, str] = {}
    for i, component in enumerate(components):
        label = chr(65 + i) if i < 26 else f"G{i + 1}"
        groups[label] = component
        fixtures_by_group[label] = []
        for team in component:
            team_to_label[team] = label
    for home, away, neutral in fixtures:
        fixtures_by_group[team_to_label[home]].append((home, away, neutral))

    return groups, fixtures_by_group


def _seed_order(n: int) -> list[int]:
    """Standard single-elimination bracket seed order (1-indexed) for ``n`` slots."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        nxt: list[int] = []
        for s in order:
            nxt.append(s)
            nxt.append(m + 1 - s)
        order = nxt
    return order


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------
@dataclass
class TournamentResult:
    """Aggregated tournament probabilities (one row per team)."""

    n_simulations: int
    probabilities: pd.DataFrame
    groups: dict[str, list[str]]

    def summary(self, top: int = 15) -> str:
        cols = ["team", "group", "p_group_winner", "p_advance", "p_quarterfinal", "p_champion"]
        view = self.probabilities[cols].head(top).copy()
        for c in cols[2:]:
            view[c] = (100 * view[c]).round(1)
        return (
            f"World Cup Monte-Carlo ({self.n_simulations} simulations) - "
            f"top {top} title contenders (%)\n"
            + view.to_string(index=False)
        )


def _simulate_group(
    sampler: _MatchSampler,
    teams: list[str],
    fixtures: list[tuple[str, str, bool]],
    rng: np.random.Generator,
) -> list[tuple[str, int, int, int]]:
    """Simulate a group; return teams ranked, each as ``(team, points, gd, gf)``."""
    points = {t: 0 for t in teams}
    goals_for = {t: 0 for t in teams}
    goals_against = {t: 0 for t in teams}

    for home, away, neutral in fixtures:
        hg, ag = sampler.sample_score(home, away, neutral, rng)
        goals_for[home] += hg
        goals_against[home] += ag
        goals_for[away] += ag
        goals_against[away] += hg
        if hg > ag:
            points[home] += 3
        elif ag > hg:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    ranked = sorted(
        teams,
        key=lambda t: (points[t], goals_for[t] - goals_against[t], goals_for[t], rng.random()),
        reverse=True,
    )
    return [(t, points[t], goals_for[t] - goals_against[t], goals_for[t]) for t in ranked]


def simulate_tournament(
    fit_result: FitResult,
    group_fixtures: pd.DataFrame,
    *,
    n_simulations: int = 5000,
    qualifiers_per_group: int = 2,
    best_third_places: int = 8,
    neutral_knockout: bool = True,
    max_goals: int = 10,
    seed: int | None = None,
) -> TournamentResult:
    """Run a Monte-Carlo simulation of the whole tournament.

    Parameters
    ----------
    fit_result:
        A fitted model whose ``teams`` include every team in ``group_fixtures``.
    group_fixtures:
        Canonical-schema fixtures of the group stage (``home, away[, neutral]``);
        scores are ignored (only the matchups define the groups).
    n_simulations:
        Number of Monte-Carlo runs.
    qualifiers_per_group, best_third_places:
        Advancement rule (default 2 per group + best 8 thirds = 32 → 2026 format).

    Returns
    -------
    TournamentResult
    """
    groups, fixtures_by_group = derive_groups(group_fixtures)
    n_groups = len(groups)
    n_qualifiers = n_groups * qualifiers_per_group + best_third_places
    if n_qualifiers & (n_qualifiers - 1) != 0:
        raise ValueError(
            f"Number of qualifiers ({n_qualifiers}) must be a power of 2 for the bracket."
        )

    known = set(fit_result.teams)
    all_teams = [t for group in groups.values() for t in group]
    unknown = [t for t in all_teams if t not in known]
    if unknown:
        raise ValueError(f"These teams are not in the fitted model: {unknown}")

    team_group = {t: label for label, group in groups.items() for t in group}
    # Strength rating used to seed the knockout bracket.
    rating = {t: fit_result.attack[fit_result.index(t)] / fit_result.defense[fit_result.index(t)] for t in all_teams}

    sampler = _MatchSampler(fit_result, max_goals)
    rng = np.random.default_rng(seed)

    count_group_winner = defaultdict(int)
    count_advance = defaultdict(int)
    count_r16 = defaultdict(int)
    count_qf = defaultdict(int)
    count_sf = defaultdict(int)
    count_final = defaultdict(int)
    count_champion = defaultdict(int)

    for _ in range(n_simulations):
        winners: list[str] = []
        runners: list[str] = []
        thirds: list[tuple[str, int, int, int]] = []

        for label, teams in groups.items():
            standings = _simulate_group(sampler, teams, fixtures_by_group[label], rng)
            count_group_winner[standings[0][0]] += 1
            winners.append(standings[0][0])
            if qualifiers_per_group >= 2:
                runners.append(standings[1][0])
            thirds.append(standings[2])  # (team, pts, gd, gf)

        best_thirds = [
            t[0]
            for t in sorted(thirds, key=lambda s: (s[1], s[2], s[3], rng.random()), reverse=True)[
                :best_third_places
            ]
        ]
        qualifiers = winners + runners + best_thirds
        for t in qualifiers:
            count_advance[t] += 1

        # Seed the knockout bracket by model strength.
        seeded = sorted(qualifiers, key=lambda t: rating[t], reverse=True)
        seed_to_team = {s + 1: team for s, team in enumerate(seeded)}
        bracket = [seed_to_team[s] for s in _seed_order(len(qualifiers))]

        current = bracket
        size = len(current)
        round_counters = {16: count_r16, 8: count_qf, 4: count_sf, 2: count_final}
        while size > 1:
            next_round = [
                sampler.knockout_winner(current[i], current[i + 1], neutral_knockout, rng)
                for i in range(0, len(current), 2)
            ]
            size //= 2
            if size in round_counters:
                for t in next_round:
                    round_counters[size][t] += 1
            current = next_round
        count_champion[current[0]] += 1

    n = float(n_simulations)
    rows = []
    for team in all_teams:
        rows.append(
            {
                "team": team,
                "group": team_group[team],
                "p_group_winner": count_group_winner[team] / n,
                "p_advance": count_advance[team] / n,
                "p_round16": count_r16[team] / n,
                "p_quarterfinal": count_qf[team] / n,
                "p_semifinal": count_sf[team] / n,
                "p_final": count_final[team] / n,
                "p_champion": count_champion[team] / n,
            }
        )
    probabilities = (
        pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    )
    return TournamentResult(n_simulations=n_simulations, probabilities=probabilities, groups=groups)
