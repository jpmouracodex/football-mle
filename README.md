# ⚽ football_mle — Goal prediction by Maximum Likelihood

A **from-scratch** implementation of the **Maher (1982)** and **Dixon-Coles (1997)**
football models, estimated by **Maximum Likelihood (MLE)** with numerical
optimization — not a black-box GLM — plus a **Streamlit** front-end that pulls
**free, key-less data** for major leagues and the **2026 FIFA World Cup**.

The likelihood, its analytic gradient, the time-decay weighting, the low-score
correction, the score matrix and the validation metrics are all built explicitly,
so every modelling choice is transparent and inspectable.

---

## ✨ Highlights

| Capability | Where |
|---|---|
| Maher pure-Poisson MLE | [`maher.py`](football_mle/maher.py) |
| Dixon-Coles ρ/τ low-score correction | [`dixon_coles.py`](football_mle/dixon_coles.py) |
| Exponential **time decay** (ξ / half-life) | [`data.py`](football_mle/data.py) |
| **Neutral-venue** support (home edge only for hosts) | [`likelihood.py`](football_mle/likelihood.py) |
| **Analytic gradient** (scales to hundreds of teams) | [`likelihood.py`](football_mle/likelihood.py) |
| Asymptotic **standard errors** (observed information) | [`fitting.py`](football_mle/fitting.py) |
| Score matrix + 1X2 probabilities | [`prediction.py`](football_mle/prediction.py) |
| RPS, multi-class Brier, **temporal CV**, **optimal-ξ search** | [`validation.py`](football_mle/validation.py) |
| **Monte-Carlo tournament simulator** (World Cup) | [`simulation.py`](football_mle/simulation.py) |
| Free data loaders (clubs + national teams) | [`sources/`](football_mle/sources) |
| **Streamlit** app | [`app/streamlit_app.py`](app/streamlit_app.py) |

---

## 📐 Mathematical foundation

### Maher (pure Poisson)

For home team $i$ vs away team $j$, goals are conditionally independent Poisson:

$$X_{ij}\sim\mathrm{Poisson}(\lambda_{ij}),\quad Y_{ij}\sim\mathrm{Poisson}(\mu_{ij})$$

$$\lambda_{ij}=\alpha_i\,\beta_j\,\gamma^{\,\mathbb{1}[\text{not neutral}]},\qquad \mu_{ij}=\alpha_j\,\beta_i$$

with **attack** $\alpha$, **defense** $\beta$ (higher = weaker defense) and **home
advantage** $\gamma$ — applied **only to non-neutral matches**, so the same model
serves leagues and neutral-ground tournaments.

**Identifiability:** rates depend only on $\alpha_i\beta_j$, so the scale is fixed
by $\frac1n\sum_i\alpha_i=1$ (imposed as an SLSQP equality constraint).

### Dixon-Coles correction

$$\tau_{\lambda,\mu}(x,y)=\begin{cases}1-\lambda\mu\rho&(0,0)\\1+\lambda\rho&(0,1)\\1+\mu\rho&(1,0)\\1-\rho&(1,1)\\1&\text{else}\end{cases}\qquad P(x,y)=\tau\,\mathrm{Pois}(x\mid\lambda)\,\mathrm{Pois}(y\mid\mu)$$

Since Dixon-Coles **contains** Maher ($\rho=0$), $\ell_{\text{DC}}\ge\ell_{\text{Maher}}$ under identical weights.

### Time decay & objective

$$-\ell(\theta)=-\sum_k w_k\Big[\ln\tau(x_k,y_k)+\ln\mathrm{Pois}(x_k\mid\lambda_k)+\ln\mathrm{Pois}(y_k\mid\mu_k)\Big],\quad w_k=e^{-\xi\,(t_{\text{ref}}-t_k)}$$

minimized with `scipy.optimize.minimize(method="SLSQP", jac=…)`, using the
**analytic gradient** (verified against finite differences to ~$10^{-9}$).

### Validation

- **RPS** (Constantinou & Fenton 2012), order-aware: $\frac{1}{r-1}\sum_{i=1}^{r-1}\big(\sum_{j\le i}(p_j-o_j)\big)^2\in[0,1]$.
- **Multi-class Brier**: $\sum_j(p_j-o_j)^2\in[0,2]$. Both proper; lower is better.

---

## 🗂️ Project structure

```
Regressao_Futebol/
├── football_mle/
│   ├── data.py            # Stage 1: canonical schema + time-decay weights
│   ├── likelihood.py      # core: rates, tau, NLL + analytic gradient
│   ├── fitting.py         # MLE engine (SLSQP) + standard errors
│   ├── maher.py           # Stage 2
│   ├── dixon_coles.py     # Stage 3
│   ├── prediction.py      # Stage 4: score matrix + 1X2
│   ├── validation.py      # Stage 5: RPS/Brier + temporal CV + optimal xi
│   ├── simulation.py      # Monte-Carlo tournament simulator
│   └── sources/           # free data: leagues.py (clubs), international.py (NTs/WC)
├── app/streamlit_app.py   # Streamlit front-end
├── examples/              # synthetic_data.py, full_pipeline.py
├── tests/test_model.py
└── requirements.txt
```

---

## 🚀 Installation & usage

```bash
pip install -r requirements.txt

streamlit run app/streamlit_app.py     # the interactive app
python examples/full_pipeline.py       # the 5 stages, end to end
python tests/test_model.py             # tests (or: pytest -q)
```

### Library example

```python
from football_mle import prepare_data, fit_dixon_coles, predict_match, evaluate
from football_mle.sources import fetch_league

df = fetch_league("Premier League (England)")          # free, no API key
model = fit_dixon_coles(prepare_data(df))
print(predict_match(model, "Arsenal", "Chelsea").summary())
```

---

## 🖥️ The Streamlit app

Two modes in the sidebar:

- **Club league** — pick a league (Premier League, La Liga, Serie A, Bundesliga,
  Ligue 1, Eredivisie, Primeira Liga, …), number of seasons, model and time-decay
  half-life. Shows the MLE ratings, and a match predictor with a score-probability
  heatmap and 1X2 odds.
- **World Cup 2026** — fits national-team ratings from recent internationals and
  runs a **Monte-Carlo simulation** of the tournament, reporting each team's
  probability of winning its group, advancing, and lifting the trophy. Also
  includes the groups (read from the official fixtures) and a neutral-venue match
  predictor.

### Free data sources (no API key)

| Data | Source |
|---|---|
| Club leagues | [football-data.co.uk](https://www.football-data.co.uk/) CSVs |
| National teams & **World Cup 2026 fixtures** | [martj42/international_results](https://github.com/martj42/international_results) |

The 2026 World Cup **groups are reconstructed from the official fixture list** in
the international-results dataset (connected components of the group-stage graph),
so they stay correct without any hard-coding.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick this repo.
3. Set **Main file path** to `app/streamlit_app.py` and deploy.

`requirements.txt` (repo root) and `.streamlit/config.toml` are already set up, and
the app adds the repo root to `sys.path`, so `import football_mle` works on the
cloud with no extra configuration. Data is fetched live from the free sources at
runtime — no API keys or secrets required.

---

## 🧪 Verified behaviour

- MLE recovers synthetic ground-truth (attack/defense correlation > 0.9).
- Analytic gradient matches finite differences to ~$10^{-9}$ → fits in well under a
  second even for ~200 international teams.
- Nesting holds: $\ell_{\text{DC}}\ge\ell_{\text{Maher}}$ on identical data.
- Neutral-venue rate ratio equals $\gamma$ exactly.
- Tournament probabilities are coherent (champions sum to 1, advancers to 32).

### Honest reading

- ρ's gain concentrates in the **score** distribution (and the likelihood); on the
  aggregated **1X2** market its effect is small.
- Time decay helps only when team strength **drifts** over time (real data); on
  stationary synthetic data it is roughly neutral.
- The simulator's **group stage is faithful** (real fixtures, top-2 + 8 best
  thirds); the **knockout bracket is a strength-seeded approximation** of the
  official slotting (a documented simplification, not the exact third-place table).

---

## 📚 References

- Maher, M. J. (1982). *Modelling association football scores.* **Statistica Neerlandica**, 36(3), 109–118.
- Dixon, M. J., & Coles, S. G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* **Applied Statistics**, 46(2), 265–280.
- Constantinou, A. C., & Fenton, N. E. (2012). *Solving the problem of inadequate scoring rules…* **JQAS**, 8(1).
