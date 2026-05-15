# Hosting this repo on GitHub Pages

The `/docs` folder is configured as a self-contained static site:

- `docs/index.html` — landing page
- `docs/examples/buy_and_hold/` and `docs/examples/naive_momentum/` — two pre-rendered scorecards (HTML + Markdown + JSON + equity/drawdown PNGs)
- `docs/METHODS.md`, `docs/DISCLAIMERS.md`, `docs/LAUNCH.md`, `docs/PROFILE_SNIPPET.md` — accessible at their own URLs
- `docs/.nojekyll` — disables Jekyll so the existing folder structure is served verbatim (otherwise Jekyll would treat the .md research docs as posts and reorganize things)

## Manual one-click setup (must be done by the repo owner)

GitHub Pages can only be turned on through the web UI; I can't toggle it from here.

1. Push the branch to GitHub if not already pushed: `git push origin main`
2. Open the repo on GitHub: `https://github.com/<owner>/agent-backtest-lab`
3. Settings → Pages → Build and deployment
4. Source: **Deploy from a branch**
5. Branch: **main** · Folder: **/docs** · click **Save**
6. Wait 30-60 seconds. The site will be live at: `https://<owner>.github.io/agent-backtest-lab/`

The landing page links to the two example scorecards; both are fully self-contained HTML with base64-embedded PNGs and need no external resources to render.

## What the site does NOT do (deliberately)

- **No interactive demo.** The Python harness does not run in the browser. Visitors install the package locally to run it.
- **No `buy`/`sell` language.** Same compliance posture as every other emitted artifact. The not-advice disclaimer is at the top of the landing page and in the footer of every embedded scorecard.
- **No live data.** All example scorecards use the bundled synthetic GBM fixture.

## Updating the site

To refresh the example scorecards:

```bash
abl evaluate buy_and_hold --window 2020-01-06..2024-12-31 --out docs/examples/buy_and_hold --quiet
abl evaluate naive_momentum --window 2020-01-06..2024-12-31 --out docs/examples/naive_momentum --quiet
```

Commit the regenerated files. GitHub Pages will auto-rebuild on push (no extra workflow needed).

## Optional next steps (not built yet)

- **Pyodide-based interactive demo.** Run `abl` in the browser via WebAssembly Python. Substantial extra work; not necessary for launch.
- **Comparison-of-runs page.** A second example showing a deliberately-leaky strategy and a deliberately-overfit one with their red banners visible — useful as a screenshot for the README. The fixtures already exist in `tests/`; we'd just need a CLI hook to expose them.
- **Star-history badge.** Already on the README; can also be embedded on the landing page.
