"""
Lightweight UI internationalization for the Streamlit app (EN / PT / ES).

Pure data + a tiny helper — no Streamlit or model imports — so it can be unit
checked on its own. Only presentation strings live here; the app logic is
untouched. Missing keys fall back to English.
"""
from __future__ import annotations

from typing import Callable

# Sidebar language picker: display name -> language code.
LANGUAGES: dict[str, str] = {"English": "en", "Português": "pt", "Español": "es"}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "subtitle": (
            "Maher (1982) & Dixon-Coles (1997) models, extending Vizzoni (2018), "
            "fitted from scratch by MLE. "
            "Free data: football-data.co.uk (clubs) and martj42/international_results "
            "(national teams)."
        ),
        "mode": "Mode",
        "mode_world_cup": "World Cup 2026",
        "mode_club": "Club league",
        "model": "Model",
        "model_dixon": "Dixon-Coles",
        "model_maher": "Maher (pure Poisson)",
        "half_life": "Time-decay half-life (days, 0 = off)",
        "ratings_title": "Team ratings (MLE)",
        "attack_axis": "Attack strength (mean = 1)",
        "defense_legend": "Defense (lower = better)",
        "home_adv": (
            "Home advantage γ = {gamma:.3f} (≈ {pct:.0f}% scoring boost at home)"
            "{rho_suffix} · log-likelihood = {ll:.1f}"
        ),
        "rho_suffix": " · ρ = {rho:+.3f}",
        "predictor_title": "Match predictor",
        "home_team": "Home team",
        "away_team": "Away team",
        "neutral_venue": "Neutral venue",
        "pick_two": "Pick two different teams.",
        "draw": "Draw",
        "expected_goals": "Expected goals",
        "goals_of": "{team} goals",
        "prob_pct": "P (%)",
        "most_likely": "Most likely score: **{home} {hs}–{as_} {away}** ({prob:.1f}%)",
        "club_header": "Club league",
        "league": "League",
        "seasons": "Seasons of history",
        "spinner_league": "Downloading and fitting {league}…",
        "error_league": "Could not load/fit this league: {err}",
        "success_league": "{league} · {model} · {n} teams",
        "wc_header": "World Cup 2026",
        "training_window": "Training window (years of internationals)",
        "mc_sims": "Monte-Carlo simulations",
        "spinner_nt": "Fitting national-team ratings…",
        "error_intl": "Could not load international data: {err}",
        "success_intl": "Fitted on internationals from the last {window} years · {n} teams",
        "tab_odds": "🏆 Title odds",
        "tab_groups": "👥 Groups",
        "tab_ratings": "📊 Ratings",
        "tab_match": "⚔️ Match",
        "tournament_probs": "Tournament probabilities",
        "spinner_sim": "Simulating the tournament {n}×…",
        "pwin_axis": "P(win the World Cup)",
        "wc_caption": (
            "Faithful to the official 2026 format: real group fixtures (top-2 + 8 best "
            "thirds) and FIFA's fixed knockout bracket with the Annex C third-place allocation."
        ),
        "groups_title": "Groups (derived from the official fixture list)",
        "group_label": "Group {label}",
        "col_team": "Team",
        "col_group": "Group",
        "col_win_group": "Win group",
        "col_advance": "Advance",
        "col_r16": "Round of 16",
        "col_qf": "Quarterfinal",
        "col_sf": "Semifinal",
        "col_final": "Final",
        "col_champion": "Champion",
    },
    "pt": {
        "subtitle": (
            "Modelos de Maher (1982) e Dixon-Coles (1997), estendendo Vizzoni (2018), "
            "ajustados de raiz por MLE. "
            "Dados grátis: football-data.co.uk (clubes) e martj42/international_results "
            "(seleções)."
        ),
        "mode": "Modo",
        "mode_world_cup": "Mundial 2026",
        "mode_club": "Liga de clubes",
        "model": "Modelo",
        "model_dixon": "Dixon-Coles",
        "model_maher": "Maher (Poisson puro)",
        "half_life": "Meia-vida do decaimento (dias, 0 = desligado)",
        "ratings_title": "Ratings das equipas (MLE)",
        "attack_axis": "Força de ataque (média = 1)",
        "defense_legend": "Defesa (menor = melhor)",
        "home_adv": (
            "Vantagem em casa γ = {gamma:.3f} (≈ {pct:.0f}% mais golos em casa)"
            "{rho_suffix} · log-verosimilhança = {ll:.1f}"
        ),
        "rho_suffix": " · ρ = {rho:+.3f}",
        "predictor_title": "Previsor de jogos",
        "home_team": "Equipa da casa",
        "away_team": "Equipa visitante",
        "neutral_venue": "Campo neutro",
        "pick_two": "Escolhe duas equipas diferentes.",
        "draw": "Empate",
        "expected_goals": "Golos esperados",
        "goals_of": "Golos de {team}",
        "prob_pct": "P (%)",
        "most_likely": "Placar mais provável: **{home} {hs}–{as_} {away}** ({prob:.1f}%)",
        "club_header": "Liga de clubes",
        "league": "Liga",
        "seasons": "Épocas de histórico",
        "spinner_league": "A descarregar e ajustar {league}…",
        "error_league": "Não foi possível carregar/ajustar esta liga: {err}",
        "success_league": "{league} · {model} · {n} equipas",
        "wc_header": "Mundial 2026",
        "training_window": "Janela de treino (anos de jogos de seleções)",
        "mc_sims": "Simulações Monte-Carlo",
        "spinner_nt": "A ajustar os ratings das seleções…",
        "error_intl": "Não foi possível carregar os dados de seleções: {err}",
        "success_intl": "Ajustado com jogos de seleções dos últimos {window} anos · {n} equipas",
        "tab_odds": "🏆 Probabilidades do título",
        "tab_groups": "👥 Grupos",
        "tab_ratings": "📊 Ratings",
        "tab_match": "⚔️ Jogo",
        "tournament_probs": "Probabilidades do torneio",
        "spinner_sim": "A simular o torneio {n}×…",
        "pwin_axis": "P(ganhar o Mundial)",
        "wc_caption": (
            "Fiel ao formato oficial de 2026: jogos reais da fase de grupos (2 primeiros + "
            "8 melhores terceiros) e o chaveamento oficial da FIFA com a alocação dos "
            "terceiros (Anexo C)."
        ),
        "groups_title": "Grupos (derivados do calendário oficial)",
        "group_label": "Grupo {label}",
        "col_team": "Equipa",
        "col_group": "Grupo",
        "col_win_group": "Vence grupo",
        "col_advance": "Avança",
        "col_r16": "Oitavos",
        "col_qf": "Quartos",
        "col_sf": "Meias",
        "col_final": "Final",
        "col_champion": "Campeão",
    },
    "es": {
        "subtitle": (
            "Modelos de Maher (1982) y Dixon-Coles (1997), extendiendo a Vizzoni (2018), "
            "ajustados desde cero por MLE. "
            "Datos gratuitos: football-data.co.uk (clubes) y martj42/international_results "
            "(selecciones)."
        ),
        "mode": "Modo",
        "mode_world_cup": "Mundial 2026",
        "mode_club": "Liga de clubes",
        "model": "Modelo",
        "model_dixon": "Dixon-Coles",
        "model_maher": "Maher (Poisson puro)",
        "half_life": "Vida media del decaimiento (días, 0 = desactivado)",
        "ratings_title": "Ratings de los equipos (MLE)",
        "attack_axis": "Fuerza de ataque (media = 1)",
        "defense_legend": "Defensa (menor = mejor)",
        "home_adv": (
            "Ventaja local γ = {gamma:.3f} (≈ {pct:.0f}% más goles en casa)"
            "{rho_suffix} · log-verosimilitud = {ll:.1f}"
        ),
        "rho_suffix": " · ρ = {rho:+.3f}",
        "predictor_title": "Predictor de partidos",
        "home_team": "Equipo local",
        "away_team": "Equipo visitante",
        "neutral_venue": "Campo neutral",
        "pick_two": "Elige dos equipos diferentes.",
        "draw": "Empate",
        "expected_goals": "Goles esperados",
        "goals_of": "Goles de {team}",
        "prob_pct": "P (%)",
        "most_likely": "Marcador más probable: **{home} {hs}–{as_} {away}** ({prob:.1f}%)",
        "club_header": "Liga de clubes",
        "league": "Liga",
        "seasons": "Temporadas de histórico",
        "spinner_league": "Descargando y ajustando {league}…",
        "error_league": "No se pudo cargar/ajustar esta liga: {err}",
        "success_league": "{league} · {model} · {n} equipos",
        "wc_header": "Mundial 2026",
        "training_window": "Ventana de entrenamiento (años de partidos de selecciones)",
        "mc_sims": "Simulaciones Monte-Carlo",
        "spinner_nt": "Ajustando los ratings de las selecciones…",
        "error_intl": "No se pudieron cargar los datos de selecciones: {err}",
        "success_intl": "Ajustado con partidos de selecciones de los últimos {window} años · {n} equipos",
        "tab_odds": "🏆 Probabilidades del título",
        "tab_groups": "👥 Grupos",
        "tab_ratings": "📊 Ratings",
        "tab_match": "⚔️ Partido",
        "tournament_probs": "Probabilidades del torneo",
        "spinner_sim": "Simulando el torneo {n}×…",
        "pwin_axis": "P(ganar el Mundial)",
        "wc_caption": (
            "Fiel al formato oficial de 2026: partidos reales de la fase de grupos (2 "
            "primeros + 8 mejores terceros) y el cuadro oficial de la FIFA con la "
            "asignación de terceros (Anexo C)."
        ),
        "groups_title": "Grupos (derivados del calendario oficial)",
        "group_label": "Grupo {label}",
        "col_team": "Equipo",
        "col_group": "Grupo",
        "col_win_group": "Gana grupo",
        "col_advance": "Avanza",
        "col_r16": "Octavos",
        "col_qf": "Cuartos",
        "col_sf": "Semis",
        "col_final": "Final",
        "col_champion": "Campeón",
    },
}


def make_t(lang: str) -> Callable[..., str]:
    """Return a translator ``t(key, **kwargs)`` for ``lang`` (English fallback)."""
    base = TRANSLATIONS["en"]
    local = TRANSLATIONS.get(lang, base)

    def t(key: str, **kwargs: object) -> str:
        template = local.get(key, base.get(key, key))
        return template.format(**kwargs) if kwargs else template

    return t
