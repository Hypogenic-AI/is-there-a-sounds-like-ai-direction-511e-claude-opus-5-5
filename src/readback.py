"""Manipulation check: read every generated/reference text back with the (unsteered) instruct model and
project its mean-pooled residual at the steering layer onto d_AI (same raw-text protocol as direction fitting)."""
import sys, json
import pandas as pd
from common import *

cfg = json.load(open(RES / "steer_config.json")); LB = cfg["layer"]
d = np.load(RES / "dirs_instruct.npy")[LB, 0]
model, tok = load_model(INSTRUCT)
for f in sys.argv[1:]:
    df = pd.read_parquet(RES / f)
    A, _ = mean_pooled_acts(model, tok, df.text.tolist(), batch_size=48)
    df["proj_ai"] = A[:, LB + 1].astype(np.float32) @ d
    df[["file", "cond", "pid", "proj_ai"]].to_parquet(RES / f.replace("scores_", "readback_"))
    print(df.groupby("cond").proj_ai.mean().round(2).to_string())
