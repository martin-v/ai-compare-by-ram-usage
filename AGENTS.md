# Agent Guidance

## Product Goal

Help people choose a concrete open model variant, including its quantization format, for local use with a known RAM budget.

- RAM is the only resource limit. Keep quantization formats as separate result rows; do not add a quantization filter unless requested.
- Make the exact model/repository name, precision or Q format, and estimated RAM easy to scan together.
- Keep the RAM number input and slider synchronized. Only show a result as fitting when both its parameter count and precision are known and its estimate is within the selected limit.
- Prefer plain language and disclose that RAM is estimated, not measured peak usage.

## Data

- `data.js` is generated output. Do not edit it by hand.
- Run `python3 -m pip install pyarrow` and `python3 scripts/import_benchmarks.py data.js` to refresh it locally.
- Benchmark sources and source links are maintained in `scripts/import_benchmarks.py` and displayed on the page. Do not present estimates as measurements or imply scores from different benchmark tasks are directly comparable.
- RAM estimation is parameter count in billions times bytes per weight times 1.2. Real usage also depends on runtime, context length and KV cache.

## Site And Deployment

- This is a static GitHub Pages site with inline browser JavaScript and MVP.css; there is no frontend build system or package manager.
- `.github/workflows/deploy.yml` refreshes benchmark data daily and before deployment, commits changed `data.js`, then deploys the site. Keep its Actions permissions and ordered import/deploy steps intact when changing the workflow.
- Use Bash and Python for repository scripts; do not introduce Ruby.

## Validation

- Check Python changes with `python3 -m py_compile scripts/import_benchmarks.py` and run the importer when network access is available.
- Parse the inline JavaScript with `node --check` and run focused checks for RAM filtering, search and paging after UI changes.
- Run `git diff --check` before finishing.