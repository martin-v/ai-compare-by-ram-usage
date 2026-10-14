# Open Model Benchmarks

A static browser-based comparison of open models by published benchmark scores, precision, estimated RAM and generation speed. Search by model or publisher, filter by quantization, sort benchmark columns and browse the imported model list.

## Data sources

GitHub Actions refreshes `data.js` daily and before every Pages deployment:

- [Open LLM Leaderboard dataset](https://huggingface.co/datasets/open-llm-leaderboard/contents) provides MMLU-Pro, GPQA and IFEval evaluations, model parameter counts and reported precision.
- [H200 Quantization Benchmarks](https://huggingface.co/datasets/ssakethch/h200-quantization-benchmarks) provides measured MMLU/task accuracy and throughput for AWQ, GPTQ, NVFP4, FP8 and BF16 variants on NVIDIA H200 MIG using vLLM.

The importer is `scripts/import_benchmarks.py`. It writes a local JavaScript data file and needs Python 3.12+ and PyArrow. The hosted site works without a runtime API or browser-side network calls to benchmark providers.

Reported scores are source results; scores from different tasks are not directly comparable. H200 output speed is derived from measured median TPOT at one request per second. Other speed values and RAM footprints are estimates, not measured hardware results. RAM uses parameter count, reported precision and a 20% runtime-overhead allowance.

Open `index.html` directly to run the site locally.
