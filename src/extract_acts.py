"""Extract mean-pooled residual activations (all layers) + mean token log-prob for every text in
results/texts.jsonl, for the instruct or base model. Saves results/acts/{tag}.npy and {tag}_lp.npy"""
import sys, time, json
from common import *

tag = sys.argv[1]  # 'instruct' or 'base'
name = INSTRUCT if tag == "instruct" else BASE
set_seed(0)
texts = [json.loads(l)["text"] for l in open(RES / "texts.jsonl")]
model, tok = load_model(name)
t = time.time()
A, lp = mean_pooled_acts(model, tok, texts, batch_size=48)
np.save(ACTS / f"{tag}.npy", A); np.save(ACTS / f"{tag}_lp.npy", lp)
print(tag, A.shape, f"{time.time()-t:.0f}s")
