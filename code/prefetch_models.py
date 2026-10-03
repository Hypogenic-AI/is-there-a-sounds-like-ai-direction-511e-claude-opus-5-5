"""Pre-download candidate subject models and independent detectors into the HF cache."""
from huggingface_hub import snapshot_download, list_repo_files
MODELS = ["google/gemma-2-2b-it", "google/gemma-2-2b", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B",
          "desklib/ai-text-detector-v1.01", "fakespot-ai/roberta-base-ai-text-detection-v1",
          "Hello-SimpleAI/chatgpt-detector-roberta", "openai-community/roberta-large-openai-detector",
          "s-nlp/roberta-base-formality-ranker", "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"]
for m in MODELS:
    files = list_repo_files(m)
    has_st = any(f.endswith(".safetensors") for f in files)
    allow = None if not has_st else ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.py"]
    p = snapshot_download(m, allow_patterns=allow)
    print("OK", m, p, flush=True)
# Gemma Scope residual SAEs used in 2503.03601 (layer 12/16/20, width 16k) - small subset
for L in [12, 16, 20]:
    snapshot_download("google/gemma-scope-2b-pt-res", allow_patterns=[f"layer_{L}/width_16k/*"])
    print("OK gemma-scope layer", L, flush=True)
