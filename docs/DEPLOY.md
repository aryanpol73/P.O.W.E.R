# Putting P.O.W.E.R. online for free (your PC + Cloudflare quick tunnel)

Hosted free tiers that can run the backend (Hugging Face Spaces, etc.) now need a paid plan, so the
whole stack runs on this PC and a free Cloudflare quick tunnel gives it a public `https://` address.
No account, card or domain is needed.

| Piece | Where |
|---|---|
| Console, API (simulator, agents, 11 models), copilot, TimescaleDB | Docker on this PC |
| `edge` (Caddy) | puts console and API behind one port, 8090 |
| `tunnel` (cloudflared) | publishes port 8090 as `https://<random>.trycloudflare.com` |

## Start

`.env` must contain the line `POWER_PUBLIC_API_URL=` (empty), so the console calls the API on its own
origin. Then:

```powershell
docker compose --profile public up -d --build
```

Print the public address:

```powershell
docker compose logs tunnel | Select-String trycloudflare.com | Select-Object -Last 1
```

Locally, open http://localhost:8090 (in this mode the console on 3011 expects the router in front of it).

## Good to know

- The site is up only while the PC is on and Docker Desktop is running. All containers restart on their
  own when Docker starts.
- The address changes every time the `tunnel` container restarts (reboot, Docker restart). Print it again
  with the command above. A fixed address needs a Cloudflare account plus your own domain (named tunnel).
- The site is public: anyone with the link can view it. Admin actions still need `ADMIN_TOKEN`, which
  stays on the server.
- After an outage the live clock catches up if it was down for 48 hours or less, otherwise it restarts at
  the current hour.

## Stop sharing

```powershell
docker compose stop tunnel edge
```

## Back to local-only URLs

Set `POWER_PUBLIC_API_URL=http://localhost:8011` in `.env` (or delete the line) and run
`docker compose up -d --build web`; the console is then at http://localhost:3011 again.

## Hosted for free: Vercel (console) + Render (API and copilot)

Always reachable at fixed addresses, without your PC. Free Render services have 512 MB of RAM and sleep
after 15 idle minutes (the first visit after that waits about a minute). To fit, the hosted API serves
9 of the 11 models (no LSTM, no random forest) and the live history restarts after each sleep.

1. **Runtime bundle.** `uv run python scripts/make_bundle.py` writes `dist/power-runtime.tar.gz`
   (about 150 MB: models, configs and a pre-seeded database; never `.env` or VAPID keys). On GitHub:
   **Releases → Draft a new release**, tag `runtime-2026.11.0`, attach the file, publish, and copy the
   asset link.
2. **Render.** render.com → **New → Blueprint** → pick this repo (it reads `render.yaml`). Fill in:
   `POWER_BUNDLE_URL` (step 1), `CORS_ORIGINS` (the Vercel URL from step 3; set it after step 3 if
   needed), `VAPID_*` (`uv run power-api vapid-keys`), and for `power-copilot`: `POWER_API_URL` (the
   `power-api` URL) and `GROQ_API_KEY`. `ADMIN_TOKEN` and `POWER_COPILOT_TOKEN` are generated; copy them.
3. **Vercel.** vercel.com → **Add New → Project** → this repo, **Root Directory `apps/web`**. Environment:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL`, `POWER_API_URL` | the `power-api` URL |
   | `POWER_MCP_URL` | the `power-copilot` URL |
   | `ADMIN_TOKEN`, `POWER_COPILOT_TOKEN` | the values Render generated |

   Deploy, then set Render's `CORS_ORIGINS` to the exact Vercel URL (no trailing slash).

Updating: push to `main`; Vercel and Render rebuild. New models: new bundle, new `POWER_BUNDLE_URL`.

## Hosted option (paid)

`deploy/hf-space/` holds a tested Hugging Face Space setup (Gradio SDK, needs HF PRO); see its README.
It downloads models from a release bundle made by `scripts/make_bundle.py`.
