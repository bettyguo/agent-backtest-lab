# Hosting this repo on GitHub Pages

The site is deployed automatically by the [`Deploy GitHub Pages` workflow](../.github/workflows/pages.yml) on every push to `main` that touches `docs/`.

## Contents of `docs/`

- `docs/index.html` — landing page (disclaimer at top, capability table, links to demos)
- `docs/examples/index.html` — listing of pre-rendered scorecards
- `docs/examples/buy_and_hold/` and `docs/examples/naive_momentum/` — pre-rendered scorecards (HTML + Markdown + JSON + equity/drawdown PNGs)
- `docs/404.html` — friendly not-found page with the not-advice disclaimer
- `docs/.nojekyll` — disables Jekyll so the folder structure is served verbatim
- `docs/METHODS.md`, `docs/DISCLAIMERS.md`, `docs/LAUNCH.md`, `docs/PROFILE_SNIPPET.md` — research-doc Markdown; landing page links to the GitHub-rendered Markdown view, not the raw Pages URL, so they display formatted

## One-time setup (must be done by the repo owner in the web UI)

You only have to do this once.

1. Push `main` to GitHub.
2. Open the repo on GitHub: `https://github.com/bettyguo/agent-backtest-lab`
3. **Settings → Pages → Build and deployment**
4. **Source: GitHub Actions** (NOT "Deploy from a branch")
5. Save. No folder selection is needed — the workflow controls what gets published.
6. The workflow auto-triggers on push; first deploy completes in ~1-2 minutes.
7. The site lives at: `https://bettyguo.github.io/agent-backtest-lab/`

If you previously selected "Deploy from a branch" with `/docs`, switch the source to "GitHub Actions" instead — the two modes are mutually exclusive and the workflow needs the Actions mode.

## Verifying a deploy

After a push, watch the workflow at:
`https://github.com/bettyguo/agent-backtest-lab/actions/workflows/pages.yml`

Both jobs (`build`, `deploy`) should turn green. The `deploy` job prints the live URL.

## What the site does NOT do (deliberately)

- **No interactive demo.** The Python harness does not run in the browser. Visitors install the package locally to run it. A Pyodide build is on the roadmap.
- **No `buy`/`sell` language.** Same compliance posture as every other emitted artifact. The not-advice disclaimer is at the top of every page and the footer of every scorecard.
- **No live data.** All example scorecards use the bundled synthetic GBM fixture.

## Updating the site

To refresh the example scorecards:

```bash
abl evaluate buy_and_hold     --window 2020-01-06..2024-12-31 --out docs/examples/buy_and_hold     --quiet
abl evaluate naive_momentum   --window 2020-01-06..2024-12-31 --out docs/examples/naive_momentum   --quiet
```

Commit and push. The workflow re-deploys automatically.

## Manual deploy

If you ever need to redeploy without changing files (e.g. after toggling Pages on for the first time), trigger the workflow manually:
`Actions → Deploy GitHub Pages → Run workflow → main → Run`.
