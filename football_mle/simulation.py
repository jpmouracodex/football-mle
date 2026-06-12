r"""
Monte-Carlo tournament simulation.

Given a fitted model and the list of group-stage fixtures, this module simulates
a full tournament many times to estimate, per team, the probabilities of winning
the group, advancing, and reaching each knockout round up to the title.

The group stage is simulated **faithfully** (real fixtures, round-robin standings
with points → goal difference → goals-for tie-breaking, top-2 per group plus the
best third-placed teams — the 2026 World Cup rule). Matches that have **already
been played** use their actual results; only the remaining fixtures are sampled,
so the probabilities update live as the tournament unfolds.

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
def derive_groups(group_fixtures: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str, bool, tuple[int, int] | None]]]]:
    """Reconstruct groups and per-group fixtures from a group-stage fixture list.

    Teams that face each other in the group stage form connected components; in a
    round-robin each component is exactly one group. Groups are labelled ``A, B,
    ...`` ordered by their alphabetically-first team. Each fixture is
    ``(home, away, neutral, result)``, where ``result`` is the actual
    ``(home_goals, away_goals)`` for matches already played (else ``None``).
    Returns ``(groups, fixtures_by_group)``.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    teams: set[str] = set()
    fixtures: list[tuple[str, str, bool, tuple[int, int] | None]] = []
    has_neutral = "neutral" in group_fixtures.columns
    has_results = {"home_goals", "away_goals"}.issubset(group_fixtures.columns)

    for _, row in group_fixtures.iterrows():
        home, away = row["home"], row["away"]
        neutral = bool(row["neutral"]) if has_neutral else False
        result: tuple[int, int] | None = None
        if has_results and pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            result = (int(row["home_goals"]), int(row["away_goals"]))
        adjacency[home].add(away)
        adjacency[away].add(home)
        teams.update((home, away))
        fixtures.append((home, away, neutral, result))

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
    fixtures_by_group: dict[str, list[tuple[str, str, bool, tuple[int, int] | None]]] = {}
    team_to_label: dict[str, str] = {}
    for i, component in enumerate(components):
        label = chr(65 + i) if i < 26 else f"G{i + 1}"
        groups[label] = component
        fixtures_by_group[label] = []
        for team in component:
            team_to_label[team] = label
    for home, away, neutral, result in fixtures:
        fixtures_by_group[team_to_label[home]].append((home, away, neutral, result))

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
    fixtures: list[tuple[str, str, bool, tuple[int, int] | None]],
    rng: np.random.Generator,
) -> list[tuple[str, int, int, int]]:
    """Simulate a group; return teams ranked, each as ``(team, points, gd, gf)``.

    Fixtures with a recorded ``result`` use the real score; the rest are sampled.
    """
    points = {t: 0 for t in teams}
    goals_for = {t: 0 for t in teams}
    goals_against = {t: 0 for t in teams}

    for home, away, neutral, result in fixtures:
        hg, ag = result if result is not None else sampler.sample_score(home, away, neutral, rng)
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


# ---------------------------------------------------------------------------
# Tournament result container
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


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------
def simulate_tournament(
    fit_result,
    group_fixtures,
    *,
    n_simulations: int = 5000,
    qualifiers_per_group: int = 2,
    best_third_places: int = 8,
    neutral_knockout: bool = True,
    max_goals: int = 10,
    seed=None,
) -> TournamentResult:
    """Run a Monte-Carlo simulation of the whole tournament.

    When the fixture field matches the 2026 FIFA World Cup (all 12 anchor teams
    present), the knockout bracket follows the official FIFA structure
    (fixed R32 crossings + third-place eligibility slots from Annex C).
    Otherwise a strength-seeded single-elimination bracket is used as fallback.
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

    derived_to_official = _official_labels(groups)
    use_fifa = derived_to_official is not None and n_groups == 12 and best_third_places == 8

    team_group = {t: label for label, group in groups.items() for t in group}
    rating = {
        t: fit_result.attack[fit_result.index(t)] / fit_result.defense[fit_result.index(t)]
        for t in all_teams
    }

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
        # ---- Group stage ------------------------------------------------
        group_winner_d = {}
        group_runner_d = {}
        thirds_d = []

        for label, teams in groups.items():
            standings = _simulate_group(sampler, teams, fixtures_by_group[label], rng)
            count_group_winner[standings[0][0]] += 1
            group_winner_d[label] = standings[0][0]
            group_runner_d[label] = standings[1][0]
            thirds_d.append((label, standings[2][1], standings[2][2], standings[2][3], standings[2][0]))

        best_thirds = sorted(
            thirds_d,
            key=lambda s: (s[1], s[2], s[3], rng.random()),
            reverse=True,
        )[:best_third_places]

        qualifiers = (
            list(group_winner_d.values())
            + list(group_runner_d.values())
            + [s[4] for s in best_thirds]
        )
        for t in qualifiers:
            count_advance[t] += 1

        # ---- Knockout stage ---------------------------------------------
        if use_fifa:
            off_w = {derived_to_official[k]: v for k, v in group_winner_d.items()}
            off_r = {derived_to_official[k]: v for k, v in group_runner_d.items()}
            off_3 = {derived_to_official[s[0]]: s[4] for s in best_thirds}

            slot_map = _allocate_thirds(list(off_3.keys()))

            match_winner = {}
            for match_num, slot_a, slot_b in _R32_STRUCTURE:
                def resolve(slot, mn=match_num):
                    if slot[0] == "1":
                        return off_w[slot[1]]
                    if slot[0] == "2":
                        return off_r[slot[1]]
                    return off_3[slot_map[mn]]
                a, b = resolve(slot_a), resolve(slot_b)
                match_winner[match_num] = sampler.knockout_winner(a, b, neutral_knockout, rng)

            for t in match_winner.values():
                count_r16[t] += 1

            for match_num in _R16_MATCHES + _QF_MATCHES + _SF_MATCHES + [_FINAL_MATCH]:
                f1, f2 = _KNOCKOUT_FEEDS[match_num]
                a, b = match_winner[f1], match_winner[f2]
                match_winner[match_num] = sampler.knockout_winner(a, b, neutral_knockout, rng)

            for t in [match_winner[m] for m in _R16_MATCHES]:
                count_qf[t] += 1
            for t in [match_winner[m] for m in _QF_MATCHES]:
                count_sf[t] += 1
            for t in [match_winner[m] for m in _SF_MATCHES]:
                count_final[t] += 1
            count_champion[match_winner[_FINAL_MATCH]] += 1

        else:
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

    # ---- Aggregate -------------------------------------------------------
    n = float(n_simulations)
    rows = []
    for team in all_teams:
        rows.append({
            "team": team,
            "group": team_group[team],
            "p_group_winner": count_group_winner[team] / n,
            "p_advance": count_advance[team] / n,
            "p_round16": count_r16[team] / n,
            "p_quarterfinal": count_qf[team] / n,
            "p_semifinal": count_sf[team] / n,
            "p_final": count_final[team] / n,
            "p_champion": count_champion[team] / n,
        })
    probabilities = (
        pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    )
    return TournamentResult(
        n_simulations=n_simulations,
        probabilities=probabilities,
        groups=groups,
    )
