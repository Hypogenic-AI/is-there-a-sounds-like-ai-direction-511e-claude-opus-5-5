"""Download HC3, HAP-E (human-ai-parallel-corpus), MAGE fully into datasets/."""
from huggingface_hub import snapshot_download
for repo, d in [("Hello-SimpleAI/HC3", "hc3"), ("browndw/human-ai-parallel-corpus", "hape"), ("yaful/MAGE", "mage")]:
    p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=f"datasets/{d}")
    print(repo, "->", p)
