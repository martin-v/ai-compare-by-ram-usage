# Agent Guidance

## Product Goal

Help people choose a concrete open model variant, including its quantization format, for local use with a known RAM budget.

- RAM is the only resource limit. Keep quantization formats as separate result rows; do not add a quantization filter unless requested.
- Make the exact model/repository name, precision or Q format, and estimated RAM easy to scan together.
- Keep the RAM number input and slider synchronized. Calculate KV cache from each model's supported context; if support exceeds 200,000 tokens, use half of that context. Do not use one global context request.
- Only show a result as fitting when parameter count, precision, KV-cache metadata and supported context length are known, and total estimated RAM is within the selected limit.
- Prioritize agentic work: BFCL overall, live and multi-turn scores should be prominent. Attach BFCL scores only to exact model-repository matches; never transfer a base-model score to another Q variant.
- Prefer plain language and disclose that RAM is estimated, not measured peak usage.

## Data

- `data.js` is generated output. Do not edit it by hand.
- Run `python3 -m pip install pyarrow` and `python3 scripts/import_benchmarks.py data.js` to refresh it locally.
- Benchmark sources, Hugging Face model configs, and source links are maintained in `scripts/import_benchmarks.py` and displayed on the page. Do not present estimates as measurements or imply scores from different benchmark tasks are directly comparable.
- RAM estimate is weights plus 20% runtime overhead plus the FP16 KV cache (`layers x 2 x KV heads x head dimension x 2 bytes x context tokens used`). Actual use also depends on runtime, context contents and cache dtype.

## Site And Deployment

- This is a static GitHub Pages site with inline browser JavaScript and MVP.css; there is no frontend build system or package manager.
- `.github/workflows/import-benchmarks.yml` refreshes data daily and pushes changed `data.js`; its successful completion triggers `.github/workflows/deploy.yml`, which only publishes the site. Keep write permission limited to the importer and keep Pages deploy permissions in the deploy workflow.
- Use Bash and Python for repository scripts; do not introduce Ruby.

## Validation

- Check Python changes with `python3 -m py_compile scripts/import_benchmarks.py` and run the importer when network access is available.
- Parse the inline JavaScript with `node --check` and run focused checks for RAM filtering, search and paging after UI changes.
- Run `git diff --check` before finishing.