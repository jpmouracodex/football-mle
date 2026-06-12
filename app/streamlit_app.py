"""
Streamlit front-end for football_mle.

Two modes, both backed by free, key-less data:

* **Club league** — major European leagues from football-data.co.uk.
* **World Cup 2026** — national teams from the martj42 international-results
  dataset, with a Monte-Carlo simulation of the tournament.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_mle import (  # noqa: E402
    FitResult,
    derive_groups,
    fit_dixon_coles,
    fit_maher,
    predict_match,
    prepare_data,
    score_matrix,
    simulate_tournament,
    xi_from_half_life,
)
from football_mle.sources import (  # noqa: E402
    fetch_international_results,
    fetch_league,
    list_leagues,
    played_matches,
    recent_seasons,
    world_cup_2026_fixtures,
)

st.set_page_config(page_title="football_mle — Goal prediction", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Cached data + model builders (so widget interactions don't refetch/refit)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_league_matches(league: str, n_seasons: int) -> pd.DataFrame:
    return fetch_league(league, seasons=recent_seasons(n_seasons))


@st.cache_data(ttl=3600, show_spinner=False)
def load_international() -> pd.DataFrame:
    return fetch_international_results()


def _fit(df: pd.DataFrame, model: str, half_life_days: int) -> FitResult:
    xi = xi_from_half_life(half_life_days) if half_life_days > 0 else 0.0
    data = prepare_data(df, xi=xi)
    result = fit_dixon_coles(data) if model == "Dixon-Coles" else fit_maher(data)
    result.scipy_result = None  # keep the cached object lightweight
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fit_league_model(league: str, n_seasons: int, model: str, half_life_days: int) -> FitResult:
    return _fit(load_league_matches(league, n_seasons), model, half_life_days)


@st.cache_data(ttl=3600, show_spinner=False)
def fit_international_model(window_years: int, model: str, half_life_days: int) -> FitResult:
    raw = load_international()
    since = pd.Timestamp.today() - pd.DateOffset(years=window_years)
    train = played_matches(raw, since=since)
    return _fit(train, model, half_life_days)


@st.cache_data(ttl=3600, show_spinner=False)
def world_cup_fixtures() -> pd.DataFrame:
    return world_cup_2026_fixtures(load_international())


@st.cache_data(ttl=3600, show_spinner=False)
def simulate_world_cup(window_years: int, model: str, half_life_days: int, n_sims: int) -> pd.DataFrame:
    model_fit = fit_international_model(window_years, model, half_life_days)
    fixtures = world_cup_fixtures()
    result = simulate_tournament(model_fit, fixtures, n_simulations=n_sims, seed=20260611)
    return result.probabilities


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def ratings_section(model_fit: FitResult) -> None:
    st.subheader("Team ratings (MLE)")
    table = model_fit.ratings_table()
    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.dataframe(table.round(3), use_container_width=True, height=420)
    with col_chart:
        chart = (
            alt.Chart(table)
            .mark_bar()
            .encode(
                x=alt.X("attack:Q", title="Attack strength (mean = 1)"),
                y=alt.Y("team:N", sort="-x", title=None),
                color=alt.Color("defense:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                                title="Defense (lower = better)"),
                tooltip=["team", alt.Tooltip("attack:Q", format=".3f"),
                         alt.Tooltip("defense:Q", format=".3f")],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)
    gamma = model_fit.home_advantage
    rho_txt = f" · ρ = {model_fit.rho:+.3f}" if model_fit.model == "Dixon-Coles" else ""
    st.caption(
        f"Home advantage γ = {gamma:.3f} (≈ {100 * (gamma - 1):.0f}% scoring boost at home)"
        f"{rho_txt} · log-likelihood = {model_fit.log_likelihood:.1f}"
    )


def score_heatmap(matrix, home: str, away: str, max_display: int = 6) -> alt.Chart:
    m = matrix[: max_display + 1, : max_display + 1]
    rows = [
        {"home_goals": i, "away_goals": j, "prob": float(100 * m[i, j])}
        for i in range(m.shape[0])
        for j in range(m.shape[1])
    ]
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("away_goals:O", title=f"{away} goals"),
        y=alt.Y("home_goals:O", title=f"{home} goals"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="blues"), title="P (%)"),
        tooltip=["home_goals", "away_goals", alt.Tooltip("prob:Q", format=".1f")],
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("prob:Q", format=".1f"),
        color=alt.condition(alt.datum.prob > 8, alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(height=360)


def predictor_section(model_fit: FitResult, allow_neutral: bool = False, default_neutral: bool = False) -> None:
    st.subheader("Match predictor")
    teams = sorted(model_fit.teams)
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    neutral = c3.checkbox("Neutral venue", value=default_neutral) if allow_neutral else False

    if home == away:
        st.info("Pick two different teams.")
        return

    prediction = predict_match(model_fit, home, away, max_goals=10, neutral=neutral)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"1 · {home}", f"{100 * prediction.prob_home:.1f}%")
    m2.metric("X · Draw", f"{100 * prediction.prob_draw:.1f}%")
    m3.metric(f"2 · {away}", f"{100 * prediction.prob_away:.1f}%")
    m4.metric(
        "Expected goals",
        f"{prediction.expected_home_goals:.2f} – {prediction.expected_away_goals:.2f}",
    )
    matrix = score_matrix(model_fit, home, away, max_goals=10, neutral=neutral)
    st.altair_chart(score_heatmap(matrix, home, away), use_container_width=True)
    s = prediction.most_likely_score
    st.caption(f"Most likely score: **{home} {s[0]}–{s[1]} {away}** "
               f"({100 * prediction.most_likely_score_prob:.1f}%)")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def club_league_page() -> None:
    st.sidebar.header("Club league")
    league = st.sidebar.selectbox("League", list_leagues(), index=0)
    n_seasons = st.sidebar.slider("Seasons of history", 1, 4, 2)
    model = st.sidebar.radio("Model", ["Dixon-Coles", "Maher (pure Poisson)"], index=0)
    model = "Dixon-Coles" if model.startswith("Dixon") else "Maher"
    half_life = st.sidebar.slider("Time-decay half-life (days, 0 = off)", 0, 720, 180, step=30)

    try:
        with st.spinner(f"Downloading and fitting {league}…"):
            model_fit = fit_league_model(league, n_seasons, model, half_life)
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load/fit this league: {err}")
        return

    st.success(f"{league} · {model} · {len(model_fit.teams)} teams")
    ratings_section(model_fit)
    st.divider()
    predictor_section(model_fit, allow_neutral=True)


def world_cup_page() -> None:
    st.sidebar.header("World Cup 2026")
    window = st.sidebar.slider("Training window (years of internationals)", 3, 12, 8)
    model = st.sidebar.radio("Model", ["Dixon-Coles", "Maher (pure Poisson)"], index=0)
    model = "Dixon-Coles" if model.startswith("Dixon") else "Maher"
    half_life = st.sidebar.slider("Time-decay half-life (days, 0 = off)", 0, 1460, 730, step=30)
    n_sims = st.sidebar.select_slider("Monte-Carlo simulations", [1000, 2000, 5000, 10000], value=2000)

    try:
        with st.spinner("Fitting national-team ratings…"):
            model_fit = fit_international_model(window, model, half_life)
        fixtures = world_cup_fixtures()
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load international data: {err}")
        return

    st.success(f"Fitted on internationals from the last {window} years · {len(model_fit.teams)} teams")

    tab_sim, tab_groups, tab_ratings, tab_match = st.tabs(
        ["🏆 Title odds", "👥 Groups", "📊 Ratings", "⚔️ Match"]
    )

    with tab_sim:
        with st.spinner(f"Simulating the tournament {n_sims}×…"):
            probs = simulate_world_cup(window, model, half_life, n_sims)
        st.subheader("Tournament probabilities")
        top = probs.head(16)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("p_champion:Q", title="P(win the World Cup)", axis=alt.Axis(format="%")),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", "group",
                         alt.Tooltip("p_advance:Q", format=".1%"),
                         alt.Tooltip("p_champion:Q", format=".1%")],
            )
            .properties(height=460)
        )
        st.altair_chart(chart, use_container_width=True)
        show = probs.copy()
        for c in ["p_group_winner", "p_advance", "p_round16", "p_quarterfinal", "p_semifinal", "p_final", "p_champion"]:
            show[c] = (100 * show[c]).round(1)
        st.dataframe(show, use_container_width=True, height=420)
        st.caption(
            "Group stage simulated faithfully (real fixtures, top-2 + 8 best thirds). "
            "The knockout bracket is a strength-seeded single-elimination approximation."
        )

    with tab_groups:
        st.subheader("Groups (derived from the official fixture list)")
        groups, _ = derive_groups(fixtures)
        cols = st.columns(4)
        for i, (label, teams) in enumerate(groups.items()):
            with cols[i % 4]:
                st.markdown(f"**Group {label}**")
                st.write("\n".join(f"- {t}" for t in teams))

    with tab_ratings:
        ratings_section(model_fit)

    with tab_match:
        predictor_section(model_fit, allow_neutral=True, default_neutral=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Football Predictor")
    st.caption(
        "Maher (1982) & Dixon-Coles (1997) models fitted from scratch by MLE. "
        "Free data: football-data.co.uk (clubs) and martj42/international_results (national teams)."
    )
    mode = st.sidebar.radio("Mode", ["Club league", "World Cup 2026"], index=0)
    st.sidebar.divider()
    if mode == "Club league":
        club_league_page()
    else:
        world_cup_page()


if __name__ == "__main__":
    main()
