"""Stream RAID train.csv (11.8GB) from HF and keep a parallel-preserving subsample.

Keeps every row whose source_id (or own id for human rows) falls into hash bucket 0 of 20,
for attack in {none, paraphrase}. Because all models' generations for the same human
source document share source_id, this keeps human/AI parallel triples intact.
"""
import hashlib, pandas as pd, sys
from huggingface_hub import hf_hub_url
url = hf_hub_url("liamdugan/raid", "train.csv", repo_type="dataset")
KEEP_ATTACKS = {"none", "paraphrase"}
out, n_seen = [], 0
def bucket(x): return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 20
for chunk in pd.read_csv(url, chunksize=200_000):
    n_seen += len(chunk)
    c = chunk[chunk["attack"].isin(KEEP_ATTACKS)].copy()
    key = c["source_id"].where(c["model"] != "human", c["id"])
    c = c[key.map(bucket) == 0]
    out.append(c)
    print(n_seen, sum(len(o) for o in out), flush=True)
df = pd.concat(out)
df.to_parquet("datasets/raid_subset/raid_train_subset.parquet")
print(df.groupby(["model", "attack"]).size().to_string())
print(df.groupby("domain").size().to_string())
