# Open Model Benchmarks

A static browser-based guide for choosing a concrete open model variant, including its Q/precision format, that is estimated to fit a local machine's available RAM. Each variant uses its own supported context for the KV-cache estimate; contexts over 200,000 tokens are conservatively halved. Agentic BFCL tool-use results are prioritized; quantization formats remain separate variants, not a filter.

## Data sources

GitHub Actions refreshes `data.js` daily and before every Pages deployment:

- [Open LLM Leaderboard dataset](https://huggingface.co/datasets/open-llm-leaderboard/contents) provides MMLU-Pro, GPQA and IFEval evaluations, model parameter counts and reported precision.
- [H200 Quantization Benchmarks](https://huggingface.co/datasets/ssakethch/h200-quantization-benchmarks) provides measured MMLU/task accuracy for AWQ, GPTQ, NVFP4, FP8 and BF16 variants on NVIDIA H200 MIG using vLLM.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) provides agentic tool-use, live function-calling and multi-turn scores. Scores are attached only to exact Hugging Face repository matches.

The importer is `scripts/import_benchmarks.py`. It writes a local JavaScript data file and needs Python 3.12+ and PyArrow. It imports each model's Hugging Face config to estimate KV-cache size. Variants without usable architecture metadata are excluded from RAM-fit results. The hosted site works without runtime API calls.

Reported scores are source results; scores from different tasks are not directly comparable. RAM is estimated as weights (`parameters x bytes per weight`) plus 20% runtime overhead plus an FP16 KV cache for the model's supported context (`bytes per token x context tokens used`). KV-cache size is derived from the model's Hugging Face config. Actual requirements depend on inference runtime, context contents and cache dtype. Variants are omitted if parameter count, precision, KV-cache metadata or supported context length is unknown.

Open `index.html` directly to run the site locally.
