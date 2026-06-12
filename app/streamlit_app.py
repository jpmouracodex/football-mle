"""
Streamlit front-end for football_mle.

Two modes, both backed by free, key-less data:

* **World Cup 2026** (default) — national teams from the martj42 international-
  results dataset, with a Monte-Carlo simulation of the tournament.
* **Club league** — major European leagues from football-data.co.uk.

The interface is available in English, Portuguese and Spanish (see ``i18n.py``);
only presentation strings are translated — the model logic is unchanged.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> football_mle
sys.path.insert(0, str(Path(__file__).resolve().parent))  # app dir -> i18n

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
from i18n import LANGUAGES, make_t  # noqa: E402

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
# Reusable UI components (presentation only; `t` is the translator)
# ---------------------------------------------------------------------------
def ratings_section(model_fit: FitResult, t) -> None:
    st.subheader(t("ratings_title"))
    table = model_fit.ratings_table()
    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.dataframe(table.round(3), use_container_width=True, height=420)
    with col_chart:
        chart = (
            alt.Chart(table)
            .mark_bar()
            .encode(
                x=alt.X("attack:Q", title=t("attack_axis")),
                y=alt.Y("team:N", sort="-x", title=None),
                color=alt.Color("defense:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                                title=t("defense_legend")),
                tooltip=["team", alt.Tooltip("attack:Q", format=".3f"),
                         alt.Tooltip("defense:Q", format=".3f")],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)
    gamma = model_fit.home_advantage
    rho_suffix = t("rho_suffix", rho=model_fit.rho) if model_fit.model == "Dixon-Coles" else ""
    st.caption(t("home_adv", gamma=gamma, pct=100 * (gamma - 1), rho_suffix=rho_suffix,
                 ll=model_fit.log_likelihood))


def score_heatmap(matrix, home: str, away: str, t, max_display: int = 6) -> alt.Chart:
    m = matrix[: max_display + 1, : max_display + 1]
    rows = [
        {"home_goals": i, "away_goals": j, "prob": float(100 * m[i, j])}
        for i in range(m.shape[0])
        for j in range(m.shape[1])
    ]
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("away_goals:O", title=t("goals_of", team=away)),
        y=alt.Y("home_goals:O", title=t("goals_of", team=home)),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="blues"), title=t("prob_pct")),
        tooltip=["home_goals", "away_goals", alt.Tooltip("prob:Q", format=".1f")],
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("prob:Q", format=".1f"),
        color=alt.condition(alt.datum.prob > 8, alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(height=360)


def predictor_section(model_fit: FitResult, t, allow_neutral: bool = False, default_neutral: bool = False) -> None:
    st.subheader(t("predictor_title"))
    teams = sorted(model_fit.teams)
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox(t("home_team"), teams, index=0)
    away = c2.selectbox(t("away_team"), teams, index=min(1, len(teams) - 1))
    neutral = c3.checkbox(t("neutral_venue"), value=default_neutral) if allow_neutral else False

    if home == away:
        st.info(t("pick_two"))
        return

    prediction = predict_match(model_fit, home, away, max_goals=10, neutral=neutral)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"1 · {home}", f"{100 * prediction.prob_home:.1f}%")
    m2.metric(f"X · {t('draw')}", f"{100 * prediction.prob_draw:.1f}%")
    m3.metric(f"2 · {away}", f"{100 * prediction.prob_away:.1f}%")
    m4.metric(
        t("expected_goals"),
        f"{prediction.expected_home_goals:.2f} – {prediction.expected_away_goals:.2f}",
    )
    matrix = score_matrix(model_fit, home, away, max_goals=10, neutral=neutral)
    st.altair_chart(score_heatmap(matrix, home, away, t), use_container_width=True)
    s = prediction.most_likely_score
    st.caption(t("most_likely", home=home, hs=s[0], as_=s[1], away=away,
                 prob=100 * prediction.most_likely_score_prob))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def club_league_page(t) -> None:
    st.sidebar.header(t("club_header"))
    league = st.sidebar.selectbox(t("league"), list_leagues(), index=0)
    n_seasons = st.sidebar.slider(t("seasons"), 1, 4, 2)
    model_labels = [t("model_dixon"), t("model_maher")]
    choice = st.sidebar.radio(t("model"), model_labels, index=0)
    model = "Dixon-Coles" if choice == model_labels[0] else "Maher"
    half_life = st.sidebar.slider(t("half_life"), 0, 720, 180, step=30)

    try:
        with st.spinner(t("spinner_league", league=league)):
            model_fit = fit_league_model(league, n_seasons, model, half_life)
    except Exception as err:  # noqa: BLE001
        st.error(t("error_league", err=err))
        return

    st.success(t("success_league", league=league, model=choice, n=len(model_fit.teams)))
    ratings_section(model_fit, t)
    st.divider()
    predictor_section(model_fit, t, allow_neutral=True)


def world_cup_page(t) -> None:
    st.sidebar.header(t("wc_header"))
    window = st.sidebar.slider(t("training_window"), 3, 12, 8)
    model_labels = [t("model_dixon"), t("model_maher")]
    choice = st.sidebar.radio(t("model"), model_labels, index=0)
    model = "Dixon-Coles" if choice == model_labels[0] else "Maher"
    half_life = st.sidebar.slider(t("half_life"), 0, 1460, 730, step=30)
    n_sims = st.sidebar.select_slider(t("mc_sims"), [1000, 2000, 5000, 10000], value=2000)

    try:
        with st.spinner(t("spinner_nt")):
            model_fit = fit_international_model(window, model, half_life)
        fixtures = world_cup_fixtures()
    except Exception as err:  # noqa: BLE001
        st.error(t("error_intl", err=err))
        return

    st.success(t("success_intl", window=window, n=len(model_fit.teams)))

    tab_sim, tab_groups, tab_ratings, tab_match = st.tabs(
        [t("tab_odds"), t("tab_groups"), t("tab_ratings"), t("tab_match")]
    )

    with tab_sim:
        with st.spinner(t("spinner_sim", n=n_sims)):
            probs = simulate_world_cup(window, model, half_life, n_sims)
        st.subheader(t("tournament_probs"))
        top = probs.head(16)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("p_champion:Q", title=t("pwin_axis"), axis=alt.Axis(format="%")),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", "group",
                         alt.Tooltip("p_advance:Q", format=".1%"),
                         alt.Tooltip("p_champion:Q", format=".1%")],
            )
            .properties(height=460)
        )
        st.altair_chart(chart, use_container_width=True)
        show = probs.copy()
        prob_cols = ["p_group_winner", "p_advance", "p_round16", "p_quarterfinal",
                     "p_semifinal", "p_final", "p_champion"]
        for c in prob_cols:
            show[c] = (100 * show[c]).round(1)
        show = show.rename(columns={
            "team": t("col_team"), "group": t("col_group"),
            "p_group_winner": t("col_win_group"), "p_advance": t("col_advance"),
            "p_round16": t("col_r16"), "p_quarterfinal": t("col_qf"),
            "p_semifinal": t("col_sf"), "p_final": t("col_final"), "p_champion": t("col_champion"),
        })
        st.dataframe(show, use_container_width=True, height=420)
        st.caption(t("wc_caption"))

    with tab_groups:
        st.subheader(t("groups_title"))
        groups, _ = derive_groups(fixtures)
        cols = st.columns(4)
        for i, (label, group_teams) in enumerate(groups.items()):
            with cols[i % 4]:
                st.markdown("**" + t("group_label", label=label) + "**")
                st.write("\n".join(f"- {tm}" for tm in group_teams))

    with tab_ratings:
        ratings_section(model_fit, t)

    with tab_match:
        predictor_section(model_fit, t, allow_neutral=True, default_neutral=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Football Predictor")
    lang_name = st.sidebar.selectbox("🌐 Language · Idioma", list(LANGUAGES), index=0)
    t = make_t(LANGUAGES[lang_name])
    st.caption(t("subtitle"))

    # World Cup 2026 is the default mode (listed first).
    mode_labels = [t("mode_world_cup"), t("mode_club")]
    mode = st.sidebar.radio(t("mode"), mode_labels, index=0)
    st.sidebar.divider()
    if mode == mode_labels[0]:
        world_cup_page(t)
    else:
        club_league_page(t)


if __name__ == "__main__":
    main()
