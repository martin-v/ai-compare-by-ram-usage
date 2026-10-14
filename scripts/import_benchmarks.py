import csv
import io
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

OPEN_LLM_PARQUET = (
    "https://huggingface.co/datasets/open-llm-leaderboard/contents/"
    "resolve/main/data/train-00000-of-00001.parquet"
)
H200_DATA = (
    "https://huggingface.co/datasets/ssakethch/"
    "h200-quantization-benchmarks/resolve/main/data/"
)
SOURCES = [
    {
        "name": "Open LLM Leaderboard",
        "url": "https://huggingface.co/datasets/open-llm-leaderboard/contents",
        "description": "MMLU-Pro, GPQA and IFEval evaluations; precision as reported.",
    },
    {
        "name": "H200 Quantization Benchmarks",
        "url": "https://huggingface.co/datasets/ssakethch/h200-quantization-benchmarks",
        "description": "MMLU and token latency measured on NVIDIA H200 MIG with vLLM.",
    },
]
PRECISIONS = {
    "awq": ("AWQ 4-bit", 4),
    "gptq": ("GPTQ 4-bit", 4),
    "nvfp4": ("NVFP4", 4),
    "fp8": ("FP8", 8),
    "bf16": ("BF16", 16),
    "bfloat16": ("BF16", 16),
    "float16": ("FP16", 16),
    "fp16": ("FP16", 16),
    "4bit": ("4-bit", 4),
    "8bit": ("8-bit", 8),
    "int8": ("INT8", 8),
}


def fetch(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-compare-by-ram-usage benchmark importer"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percent(value):
    result = number(value)
    return round(result * 100, 2) if result is not None else None


def quantization(precision):
    normalized = str(precision or "unknown").strip().lower()
    label, bits = PRECISIONS.get(normalized, (str(precision or "Unknown"), None))
    return label, bits


def open_llm_models():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Install pyarrow before importing the Open LLM dataset") from exc

    table = parquet.read_table(io.BytesIO(fetch(OPEN_LLM_PARQUET)))
    models = []
    for row in table.to_pylist():
        if not row.get("Available on the hub") or row.get("Flagged"):
            continue
        params = number(row.get("#Params (B)"))
        if not params or params <= 0 or params > 500:
            continue
        precision = row.get("Precision")
        quant, bits = quantization(precision)
        name = row.get("fullname") or row.get("eval_name")
        if not name:
            continue
        models.append(
            {
                "id": "open-llm:" + str(row.get("eval_name") or name),
                "name": name,
                "params": params,
                "quant": quant,
                "precision": precision,
                "bits": bits,
                "mmlu": None,
                "mmluPro": percent(row.get("MMLU-PRO Raw")),
                "gpqa": percent(row.get("GPQA Raw")),
                "ifeval": percent(row.get("IFEval Raw")),
                "gsm8k": None,
                "tps": None,
                "hardware": None,
                "source": "Open LLM Leaderboard",
                "sourceUrl": SOURCES[0]["url"],
                "modelUrl": "https://huggingface.co/" + str(name),
                "measuredAt": row.get("Submission Date") or row.get("Upload To Hub Date"),
                "weightType": row.get("Weight type"),
            }
        )
    return models


def read_csv(filename):
    content = fetch(H200_DATA + filename).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))


def h200_models():
    inventory = {
        (row["model"], row["quant"].lower()): row
        for row in read_csv("model_inventory.csv")
    }
    accuracy = defaultdict(dict)
    for row in read_csv("accuracy.csv"):
        if row.get("task") in {"mmlu", "arc_challenge", "gsm8k", "hellaswag", "winogrande"}:
            accuracy[(row["model"], row["quant"].lower())][row["task"]] = percent(
                row.get("accuracy")
            )

    speed = {}
    for row in read_csv("throughput.csv"):
        if number(row.get("request_rate")) != 1:
            continue
        key = (row["model"], row["quant"].lower())
        tpot = number(row.get("median_tpot_ms"))
        if tpot and tpot > 0:
            speed[key] = round(1000 / tpot, 2)

    models = []
    for (name, raw_quant), item in inventory.items():
        quant, bits = quantization(raw_quant)
        size = re.search(r"(\d+(?:\.\d+)?)\s*B\b", item.get("model_size", ""), re.I)
        params = number(size.group(1)) if size else None
        measured = accuracy[(name, raw_quant)]
        models.append(
            {
                "id": "h200:" + name + ":" + raw_quant,
                "name": name,
                "params": params,
                "quant": quant,
                "precision": raw_quant,
                "bits": bits,
                "mmlu": measured.get("mmlu"),
                "mmluPro": None,
                "gpqa": None,
                "ifeval": None,
                "gsm8k": measured.get("gsm8k"),
                "arcChallenge": measured.get("arc_challenge"),
                "hellaswag": measured.get("hellaswag"),
                "winogrande": measured.get("winogrande"),
                "tps": speed.get((name, raw_quant)),
                "hardware": "NVIDIA H200 MIG / vLLM",
                "source": "H200 Quantization Benchmarks",
                "sourceUrl": SOURCES[1]["url"],
                "modelUrl": "https://huggingface.co/" + name,
                "measuredAt": None,
                "weightType": "Measured quantization",
            }
        )
    return models


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data.js")
    models = open_llm_models() + h200_models()
    if len(models) < 100:
        raise RuntimeError(f"Only imported {len(models)} models; refusing to publish incomplete data")

    quantizations = sorted({model["quant"] for model in models})
    data = {
        "generatedAt": date.today().isoformat(),
        "sources": SOURCES,
        "models": models,
        "quantizations": quantizations,
    }
    output.write_text(
        "window.BENCHMARK_DATA = "
        + json.dumps(data, ensure_ascii=True, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(models)} models across {len(quantizations)} precision groups to {output}")


if __name__ == "__main__":
    main()