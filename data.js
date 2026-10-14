// params/active in billions; mmlu = publicly reported MMLU score (%), approximate.
const MODELS = [
  { name: "Llama 3.2 1B", params: 1.2, active: 1.2, mmlu: 32 },
  { name: "Gemma 2 2B", params: 2.6, active: 2.6, mmlu: 52 },
  { name: "Llama 3.2 3B", params: 3.2, active: 3.2, mmlu: 58 },
  { name: "Phi-3.5 mini", params: 3.8, active: 3.8, mmlu: 69 },
  { name: "Mistral 7B", params: 7.2, active: 7.2, mmlu: 62 },
  { name: "Qwen2.5 7B", params: 7.6, active: 7.6, mmlu: 74 },
  { name: "Llama 3.1 8B", params: 8, active: 8, mmlu: 69 },
  { name: "Gemma 2 9B", params: 9.2, active: 9.2, mmlu: 71 },
  { name: "Phi-3 medium", params: 14, active: 14, mmlu: 78 },
  { name: "Qwen2.5 14B", params: 14.7, active: 14.7, mmlu: 79 },
  { name: "Gemma 2 27B", params: 27, active: 27, mmlu: 75 },
  { name: "Qwen2.5 32B", params: 32.5, active: 32.5, mmlu: 83 },
  { name: "Mixtral 8x7B", params: 46.7, active: 12.9, mmlu: 70 },
  { name: "Llama 3.1 70B", params: 70, active: 70, mmlu: 83 },
  { name: "Llama 3.3 70B", params: 70, active: 70, mmlu: 86 },
  { name: "Qwen2.5 72B", params: 72.7, active: 72.7, mmlu: 86 },
  { name: "Llama 3.1 405B", params: 405, active: 405, mmlu: 88.6 },
  { name: "DeepSeek V3", params: 671, active: 37, mmlu: 88.5 },
];
// bytes per weight for each quantization
const QUANTS = { "FP16": 2, "Q8": 1, "Q4": 0.5 };
