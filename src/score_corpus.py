"""Score every corpus text with formality ranker + supervised detectors + stylometrics (covariates for E2)."""
import json
import pandas as pd
from common import *
from text_metrics import Scorers, stylometrics

df = pd.read_json(RES / "texts.jsonl", lines=True)
S = Scorers()
for n in ["formality", "desklib", "fakespot"]:
    df[n] = S.score(n, df.text.tolist(), bs=64)
st = pd.DataFrame([stylometrics(t) for t in df.text])
df = pd.concat([df.drop(columns=["text", "meta"]), st], axis=1)
df.to_parquet(RES / "texts_scored.parquet")
print(df.groupby(["set", "label"])[["formality", "desklib", "fakespot"]].mean())
