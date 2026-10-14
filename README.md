# Open Model Benchmarks

A static browser-based guide for choosing a concrete open model variant, including its Q/precision format, that is estimated to fit a local machine's available RAM. Set a RAM limit, optionally search for a model, then compare the matching variants and their published benchmark scores. Quantization formats are individual variants in the results, not a filter.

## Data sources

GitHub Actions refreshes `data.js` daily and before every Pages deployment:

- [Open LLM Leaderboard dataset](https://huggingface.co/datasets/open-llm-leaderboard/contents) provides MMLU-Pro, GPQA and IFEval evaluations, model parameter counts and reported precision.
- [H200 Quantization Benchmarks](https://huggingface.co/datasets/ssakethch/h200-quantization-benchmarks) provides measured MMLU/task accuracy for AWQ, GPTQ, NVFP4, FP8 and BF16 variants on NVIDIA H200 MIG using vLLM.

The importer is `scripts/import_benchmarks.py`. It writes a local JavaScript data file and needs Python 3.12+ and PyArrow. The hosted site works without a runtime API or browser-side network calls to benchmark providers.

Reported scores are source results; scores from different tasks are not directly comparable. RAM is estimated as parameter count multiplied by bytes per weight and a 20% runtime-overhead allowance. Actual requirements depend on inference runtime, context length and KV cache. Variants with unknown parameter count or precision are omitted so every result can be compared to the selected RAM limit.

Open `index.html` directly to run the site locally.
