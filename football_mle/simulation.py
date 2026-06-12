r"""
Monte-Carlo tournament simulation.

Given a fitted model and the list of group-stage fixtures, this module simulates
a full tournament many times to estimate, per team, the probabilities of winning
the group, advancing, and reaching each knockout round up to the title.

The group stage is simulated **faithfully** (real fixtures, round-robin standings
with points → goal difference → goals-for tie-breaking, top-2 per group plus the
best third-placed teams — the 2026 World Cup rule).

For the 2026 World Cup the knockout stage follows the **official FIFA bracket**:
the fixed Round-of-32 crossings of group winners/runners-up, the eight
third-place slots with their published eligibility sets (Annex C), and the exact
bracket tree through to the final. When the input is *not* the 2026 World Cup
(e.g. a synthetic tournament), a strength-seeded single-elimination bracket is
used as a fallback. Knockout draws are resolved by the model's relative win
probability (a proxy for extra time / penalties).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fitting import FitResult
from .likelihood import score_matrix_from_rates
from .prediction import outcome_probabilities

__all__ = [
    "TournamentResult",
    "simulate_match",
    "simulate_tournament",
    "derive_groups",
    "official_groups",
]


# ---------------------------------------------------------------------------
# Official 2026 FIFA World Cup knockout structure
# ---------------------------------------------------------------------------
# One unambiguous, stable-spelling team per group -> official group letter. Used
# to label the fixture-derived groups with the official letters the bracket needs.
_OFFICIAL_GROUP_ANCHORS: dict[str, str] = {
    "Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D",
    "Germany": "E", "Netherlands": "F", "Belgium": "G", "Spain": "H",
    "France": "I", "Argentina": "J", "Portugal": "K", "England": "L",
}

# Round of 32 (matches 73-88). Each slot is ("1", L)=winner of L, ("2", L)=
# runner-up of L, or ("3",)=a best-third-placed team allocated to this match.
_R32_STRUCTURE: list[tuple[int, tuple, tuple]] = [
    (73, ("2", "A"), ("2", "B")),
    (74, ("1", "E"), ("3",)),
    (75, ("1", "F"), ("2", "C")),
    (76, ("1", "C"), ("2", "F")),
    (77, ("1", "I"), ("3",)),
    (78, ("2", "E"), ("2", "I")),
    (79, ("1", "A"), ("3",)),
    (80, ("1", "L"), ("3",)),
    (81, ("1", "D"), ("3",)),
    (82, ("1", "G"), ("3",)),
    (83, ("2", "K"), ("2", "L")),
    (84, ("1", "H"), ("2", "J")),
    (85, ("1", "B"), ("3",)),
    (86, ("1", "J"), ("2", "H")),
    (87, ("1", "K"), ("3",)),
    (88, ("2", "D"), ("2", "G")),
]

# Eligibility of the eight third-place slots (FIFA regulations, Annex C headers).
_THIRD_SLOTS: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"),
    77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),
    80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),
    82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),
    87: frozenset("DEIJL"),
}

# Bracket tree from the Round of 16 onward: match -> (feeding match 1, feeding match 2).
_KNOCKOUT_FEEDS: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}
_ROUND_OF_32 = [m for m, _, _ in _R32_STRUCTURE]
_R16_MATCHES = [89, 90, 91, 92, 93, 94, 95, 96]
_QF_MATCHES = [97, 98, 99, 100]
_SF_MATCHES = [101, 102]
_FINAL_MATCH = 104


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
# Group derivation, official labelling and third-place allocation
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


def _official_labels(groups: dict[str, list[str]]) -> dict[str, str] | None:
    """Map each derived group label to its official 2026 letter, or ``None``.

    Returns ``None`` unless every group contains exactly one recognizable anchor
    team and all twelve official letters A-L are covered (i.e. this really is the
    2026 World Cup field).
    """
    if len(groups) != 12:
        return None
    label_to_official: dict[str, str] = {}
    used: set[str] = set()
    for label, team_list in groups.items():
        found = next((_OFFICIAL_GROUP_ANCHORS[t] for t in team_list if t in _OFFICIAL_GROUP_ANCHORS), None)
        if found is None or found in used:
            return None
        used.add(found)
        label_to_official[label] = found
    return label_to_official if len(used) == 12 else None


def official_groups(group_fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Groups keyed by the official 2026 letter (A-L) when the field is the World Cup.

    Falls back to the fixture-derived labels for any other input.
    """
    groups, _ = derive_groups(group_fixtures)
    official = _official_labels(groups)
    if official is None:
        return groups
    return {official[label]: teams for label, teams in groups.items()}


def _allocate_thirds(qualifying_letters: list[str]) -> dict[int, str]:
    """Assign qualifying third-placed groups to Round-of-32 slots (FIFA eligibility).

    Solves a bipartite matching between the (up to 8) qualifying third-place
    groups and the eight third slots, respecting each slot's eligibility set.
    Deterministic given a fixed slot order. Returns ``{match_number: group_letter}``.
    """
    slots = list(_THIRD_SLOTS)
    slot_to_letter: dict[int, str] = {}

    def augment(letter: str, visited: set[int]) -> bool:
        for slot in slots:
            if letter in _THIRD_SLOTS[slot] and slot not in visited:
                visited.add(slot)
                if slot not in slot_to_letter or augment(slot_to_letter[slot], visited):
                    slot_to_letter[slot] = letter
                    return True
        return False

    for letter in qualifying_letters:
        augment(letter, set())

    # Safety net (should not trigger for valid FIFA eligibility sets): place any
    # unmatched group into any still-free slot.
    matched = set(slot_to_letter.values())
    free_slots = [s for s in slots if s not in slot_to_letter]
    for letter in qualifying_letters:
        if letter not in matched and free_slots:
            slot_to_letter[free_slots.pop()] = letter
            matched.add(letter)
    return slot_to_letter


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
# Group and knockout simulation
# ---------------------------------------------------------------------------
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


def _reached_sets(won: dict[int, str]) -> dict[str, object]:
    """Translate a map of {match_number: winner} into per-round reached sets."""
    return {
        "r16": {won[m] for m in _ROUND_OF_32},
        "qf": {won[m] for m in _R16_MATCHES},
        "sf": {won[m] for m in _QF_MATCHES},
        "final": {won[m] for m in _SF_MATCHES},
        "champion": won[_FINAL_MATCH],
    }


def _official_knockout(
    sampler: _MatchSampler,
    rng: np.random.Generator,
    standings: dict[str, list[tuple[str, int, int, int]]],
    official: dict[str, str],
    thirds_ranked: list[tuple],
    neutral: bool,
) -> dict[str, object]:
    """Run the official 2026 FIFA knockout bracket for one simulation."""
    winner = {official[label]: standings[label][0][0] for label in standings}
    runner = {official[label]: standings[label][1][0] for label in standings}

    qualifying = thirds_ranked[:8]  # (label, team, pts, gd, gf)
    third_team_by_letter = {official[row[0]]: row[1] for row in qualifying}
    allocation = _allocate_thirds(list(third_team_by_letter))  # match -> letter
    third_by_match = {match: third_team_by_letter[letter] for match, letter in allocation.items()}

    def resolve(slot: tuple, match_no: int) -> str:
        if slot[0] == "1":
            return winner[slot[1]]
        if slot[0] == "2":
            return runner[slot[1]]
        return third_by_match[match_no]  # ("3",)

    won: dict[int, str] = {}
    for match_no, slot1, slot2 in _R32_STRUCTURE:
        won[match_no] = sampler.knockout_winner(
            resolve(slot1, match_no), resolve(slot2, match_no), neutral, rng
        )
    for match_no in _R16_MATCHES + _QF_MATCHES + _SF_MATCHES + [_FINAL_MATCH]:
        feed1, feed2 = _KNOCKOUT_FEEDS[match_no]
        won[match_no] = sampler.knockout_winner(won[feed1], won[feed2], neutral, rng)
    return _reached_sets(won)


def _generic_knockout(
    sampler: _MatchSampler,
    rng: np.random.Generator,
    qualifiers: list[str],
    rating: dict[str, float],
    neutral: bool,
) -> dict[str, object]:
    """Strength-seeded single-elimination bracket (fallback for non-WC inputs)."""
    seeded = sorted(qualifiers, key=lambda t: rating[t], reverse=True)
    seed_to_team = {s + 1: team for s, team in enumerate(seeded)}
    bracket = [seed_to_team[s] for s in _seed_order(len(qualifiers))]

    entering: dict[int, list[str]] = {len(bracket): list(bracket)}
    current = bracket
    size = len(current)
    while size > 1:
        nxt = [
            sampler.knockout_winner(current[i], current[i + 1], neutral, rng)
            for i in range(0, len(current), 2)
        ]
        size //= 2
        entering[size] = nxt
        current = nxt
    return {
        "r16": set(entering.get(16, [])),
        "qf": set(entering.get(8, [])),
        "sf": set(entering.get(4, [])),
        "final": set(entering.get(2, [])),
        "champion": current[0],
    }


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------
@dataclass
class TournamentResult:
    """Aggregated tournament probabilities (one row per team)."""

    n_simulations: int
    probabilities: pd.DataFrame
    groups: dict[str, list[str]]
    official_bracket: bool = False

    def summary(self, top: int = 15) -> str:
        cols = ["team", "group", "p_group_winner", "p_advance", "p_quarterfinal", "p_champion"]
        view = self.probabilities[cols].head(top).copy()
        for c in cols[2:]:
            view[c] = (100 * view[c]).round(1)
        bracket = "official FIFA bracket" if self.official_bracket else "seeded bracket"
        return (
            f"World Cup Monte-Carlo ({self.n_simulations} simulations, {bracket}) - "
            f"top {top} title contenders (%)\n"
            + view.to_string(index=False)
        )


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
    use_official_bracket: bool | None = None,
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
    use_official_bracket:
        ``None`` (default) auto-detects the 2026 World Cup field and uses the
        official FIFA knockout bracket; ``True`` forces it (error if the field is
        not the World Cup); ``False`` always uses the strength-seeded fallback.

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

    official = None if use_official_bracket is False else _official_labels(groups)
    if use_official_bracket is True and official is None:
        raise ValueError(
            "The official FIFA 2026 bracket requires the 12 World Cup groups "
            "(recognizable anchor teams)."
        )
    use_fifa = (
        official is not None
        and n_groups == 12
        and qualifiers_per_group == 2
        and best_third_places == 8
    )

    # Display groups with official letters when this is the World Cup field.
    if use_fifa:
        team_group = {t: official[label] for label, group in groups.items() for t in group}
        display_groups = {official[label]: g for label, g in groups.items()}
    else:
        team_group = {t: label for label, group in groups.items() for t in group}
        display_groups = groups
    # Strength rating used to seed the generic fallback bracket.
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
        standings = {label: _simulate_group(sampler, groups[label], fixtures_by_group[label], rng)
                     for label in groups}
        for label in groups:
            count_group_winner[standings[label][0][0]] += 1

        # Rank the third-placed teams once (shared by the advance count and the bracket).
        thirds = [(label, *standings[label][2]) for label in groups]  # (label, team, pts, gd, gf)
        thirds_ranked = sorted(thirds, key=lambda x: (x[2], x[3], x[4], rng.random()), reverse=True)

        winners = [standings[label][0][0] for label in groups]
        runners = [standings[label][1][0] for label in groups]
        best_thirds = [row[1] for row in thirds_ranked[:best_third_places]]
        for t in winners + runners + best_thirds:
            count_advance[t] += 1

        if use_fifa:
            reached = _official_knockout(sampler, rng, standings, official, thirds_ranked, neutral_knockout)
        else:
            reached = _generic_knockout(sampler, rng, winners + runners + best_thirds, rating, neutral_knockout)

        for t in reached["r16"]:
            count_r16[t] += 1
        for t in reached["qf"]:
            count_qf[t] += 1
        for t in reached["sf"]:
            count_sf[t] += 1
        for t in reached["final"]:
            count_final[t] += 1
        count_champion[reached["champion"]] += 1

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
    return TournamentResult(
        n_simulations=n_simulations,
        probabilities=probabilities,
        groups=display_groups,
        official_bracket=bool(use_fifa),
    )
