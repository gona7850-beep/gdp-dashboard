# Deploy the Composition Design Platform

Pick one path. All are free at usual research-scale traffic.

## Option 1 — Streamlit Community Cloud (recommended, free, 3 minutes)

The platform is already configured for Streamlit Community Cloud
(`.streamlit/config.toml`, `requirements.txt`, app entry at
`app/streamlit_app.py`).

1. Fork or own this repository on GitHub (`gona7850-beep/gdp-dashboard`
   already qualifies).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **"New app"**.
4. Fill in:
   * **Repository**: `gona7850-beep/gdp-dashboard`
   * **Branch**: `main`
   * **Main file path**: `app/streamlit_app.py`
   * **Python version**: 3.11
5. *(optional)* Under **Advanced settings → Secrets**, paste:
   ```
   ANTHROPIC_API_KEY="sk-ant-..."
   OPENALEX_MAILTO="you@example.com"
   ```
   Without these, Claude-assisted features fall back to a deterministic
   heuristic, and external paper APIs still work without auth.
6. Click **Deploy**.

The first build takes ~3 minutes (installs xgboost / optuna / pymoo /
shap / scipy). Subsequent restarts are cached. You'll get a public URL
like `https://gona7850-beep-gdp-dashboard.streamlit.app`.

What you'll see on the landing page:
- **30-second live demo** — Train a model and predict Ti-6Al-4V
  properties in two clicks.
- **HTS one-click** — Rank Nb-host compounds by the three
  thermodynamic descriptors.
- **Power-user pages 7–10** — Full workflow in the sidebar.

## Option 2 — Hugging Face Spaces (free, similar setup)

1. Create a new Space at https://huggingface.co/new-space.
2. Choose **Streamlit** as the SDK.
3. Push this repository's contents (or link it). Make sure:
   * `requirements.txt` is at the repo root.
   * `app/streamlit_app.py` is the entry.
4. Add `ANTHROPIC_API_KEY` under Space secrets if you want real
   Claude calls.

HF Spaces auto-detects Streamlit apps and serves them on a public URL.

## Option 3 — Docker on any VPS (Railway / Fly.io / Render / your own)

```bash
git clone https://github.com/gona7850-beep/gdp-dashboard.git
cd gdp-dashboard
docker compose up --build
```

This brings up:
- **port 8000** — FastAPI backend + single-page HTML UI (`web/index.html`)
- **port 8501** — Streamlit research workbench

For one-process deployments (Railway, Render), Streamlit is the better
target since it includes the most polished UI:

```bash
docker run -p 8501:8501 -e PORT=8501 your-image \
    streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

## Option 4 — Local for one researcher

```bash
git clone https://github.com/gona7850-beep/gdp-dashboard.git
cd gdp-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Streamlit (recommended)
streamlit run app/streamlit_app.py
# → http://localhost:8501

# Or FastAPI + the static web UI
uvicorn backend.main:app --reload
# → http://localhost:8000 (web UI)  →  /docs (OpenAPI)
```

---

## Verify the deploy

Once your URL is live, the landing page itself proves the platform
works — the live-demo buttons train a real model and predict Ti-6Al-4V
properties end-to-end. If you see two filled tables (CV metrics +
predictions vs literature), the deploy is healthy.

For programmatic verification:

```bash
# From your local terminal, against the deployed URL:
curl https://<your-url>/_stcore/health
# → "ok"
```

For the FastAPI track:

```bash
curl https://<your-url>/health
curl https://<your-url>/api/v1/composition/status
curl https://<your-url>/api/v1/alloyforge/status
```

---

## Configuration via environment variables

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Real Claude calls for target parsing, candidate explanation, table extraction. Without it, deterministic heuristic fallbacks run. |
| `CLAUDE_COMPOSITION_MODEL` | Override default Claude model (default `claude-sonnet-4-6`). |
| `CLAUDE_TABLE_EXTRACTOR_MODEL` | Override Claude model for PDF table extraction. |
| `OPENALEX_MAILTO`, `CROSSREF_MAILTO` | Polite email for OpenAlex / CrossRef rate limits. |
| `MP_API_KEY` | Materials Project API access. Without it, MP queries are skipped (other clients still work). |

## Costs

- **Streamlit Community Cloud**: free for public apps.
- **HF Spaces**: free CPU tier.
- **Docker on a small VPS**: ~$5/month covers it.
- **Claude API**: pay-as-you-go; only triggered when the user clicks a
  Claude button on the page. With the default Sonnet model, expect
  $0.003–0.015 per query depending on the prompt.

---

## Troubleshooting

* **"ModuleNotFoundError"** during Streamlit Cloud deploy: usually a
  missing dep. The `requirements.txt` is pinned; if you forked an older
  copy, sync from `main`.
* **First request is slow**: heavy imports (xgboost, pymoo, shap) load
  lazily. The platform front-loads them on the first training click.
* **Sandbox blocks external HTTP**: external APIs (OpenAlex / arXiv /
  CrossRef / Materials Project) return empty DataFrames on failure;
  the platform doesn't crash. Real users on Streamlit Cloud get real
  results.
