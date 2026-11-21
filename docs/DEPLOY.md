# Deploying P.O.W.E.R. for free (Vercel + Hugging Face Spaces)

| Piece | Host | Cost |
|---|---|---|
| Web console (Next.js PWA) | Vercel Hobby | free |
| API (simulator, agents, 11 models) + copilot | Hugging Face Space (Gradio SDK), CPU basic (2 vCPU, 16 GB RAM) | free |
| Database | SQLite inside the Space | free |
| Models and seed data | GitHub release asset | free |

No card is needed anywhere.

**Trade-offs of the free tier**
- A Space with no visitors for 48 hours goes to sleep. The next visit wakes it: about 2 to 4 minutes to
  download the bundle and seed, after which the live clock starts again at the current hour.
- The Space's disk is not persistent, so live history (alerts, scored forecasts) starts over after a
  restart or sleep. The 30 days of history the models need are reseeded every time.
- The Space is public: anyone can read the API, as with the console. Admin actions need `ADMIN_TOKEN`.

## 1. Runtime bundle (once per model version)

Models and seed data are git-ignored, so the Space downloads them from a release:

```powershell
uv run python scripts/make_bundle.py
```

This writes `dist/power-runtime.tar.gz` (about 110 MB: models, seed data and `configs/`; never includes `data/vapid.json` or `.env`).
On GitHub: **Releases → Draft a new release**, tag `runtime-2026.11.0`, attach the file, publish.
Copy the asset link:
`https://github.com/aryanpol73/P.O.W.E.R/releases/download/runtime-2026.11.0/power-runtime.tar.gz`

## 2. Tokens and keys (on your machine)

```powershell
uv run power-api vapid-keys
```

Also make two random tokens for `ADMIN_TOKEN` and `POWER_COPILOT_TOKEN`, for example:

```powershell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 3. Hugging Face Space (API + copilot)

1. huggingface.co → **New Space**. Name it e.g. `power-api`, SDK **Gradio** (blank template; Docker Spaces need a paid plan), hardware
   **CPU basic (free)**, visibility **Public**.
2. In the Space's **Files** tab, upload the three files from `deploy/hf-space/` (`app.py`, `requirements.txt`,
   `README.md`), replacing the template's. The Space installs the packages straight from GitHub `main`;
   `app.py` itself serves the API on port 7860 (Gradio is installed but unused).
3. **Settings → Variables and secrets**:

   | Kind | Name | Value |
   |---|---|---|
   | Variable | `POWER_BUNDLE_URL` | the release link from step 1 |
   | Variable | `CORS_ORIGINS` | your Vercel URL (step 4), e.g. `https://power-console.vercel.app` |
   | Variable | `VAPID_SUBJECT` | `mailto:` plus your email |
   | Secret | `ADMIN_TOKEN` | token from step 2 |
   | Secret | `POWER_COPILOT_TOKEN` | the other token from step 2 |
   | Secret | `GROQ_API_KEY` | your Groq key (leave out to turn the copilot off) |
   | Secret | `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY` | from step 2 |

4. The Space builds (about 10 minutes the first time) and starts. Its URL is
   `https://<your-username>-power-api.hf.space`; `/api/v1/health` returns `"status": "ok"` when ready.

## 4. Vercel (console)

1. vercel.com → **Add New → Project** → import the GitHub repo. **Root Directory: `apps/web`**.
2. Environment variables (the Space URL from step 3 for the first three):

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<your-username>-power-api.hf.space` |
   | `POWER_API_URL` | same |
   | `POWER_MCP_URL` | same (the API passes `/copilot` through to the copilot) |
   | `ADMIN_TOKEN` | same token as the Space |
   | `POWER_COPILOT_TOKEN` | same token as the Space |

3. **Deploy**, then make sure the Space's `CORS_ORIGINS` is the exact Vercel URL (no trailing slash).

`NEXT_PUBLIC_*` values are baked into the browser bundle: after changing one, redeploy on Vercel.

## Updating

- Web: push to `main`; Vercel redeploys.
- Backend: push to `main`, then Space **Settings → Factory rebuild** (the Space installs from GitHub when it builds).
- New models: build and publish a new bundle, update `POWER_BUNDLE_URL`, restart the Space.

## Local stack

`docker compose up -d --build` still runs everything locally with PostgreSQL (README).
