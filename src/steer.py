"""E3/E4 generation with activation steering.
usage: python steer.py pilot | main | base
Coefficients are alpha * mean residual norm at the steering layer (norm measured on prompt tokens,
excluding the attention-sink first token). Negative alpha on d_AI = toward 'human'."""
import sys, json, time
from common import *

mode = sys.argv[1]
set_seed(0)
NAMES = ["ai", "formality", "fluency", "length_wordlen", "domain_reddit", "posttrain", "assistant_axis", "verbosity", "ai_hape", "ai_hc3"]
HUMAN_PROMPT = ("Answer the way a real person would write it: a natural, human voice. "
                "Your answer must not sound like it was written by an AI assistant.")
MAXNEW = 160


def get_dirs(file, names, layer):
    D = np.load(RES / file)[layer]  # index i = hidden_states[i+1] = output of block i (the block we hook)
    return {n: D[i] for i, n in enumerate(names)}


def resid(v, others):
    """Orthogonalise v against span(others) (Gram-Schmidt via QR)."""
    Q, _ = np.linalg.qr(np.stack(others, 1))
    r = v - Q @ (Q.T @ v)
    return r / np.linalg.norm(r)


@torch.no_grad()
def resid_norm(model, tok, prompts, layer, chat=True):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) if chat else p for p in prompts]
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()  # output of block `layer`
    m = enc.attention_mask.clone(); m[:, 0] = 0
    n = hs.norm(dim=-1)[m.bool()].mean().item()
    tok.padding_side = "left"
    return n


def run(model, tok, prompts, conds, out_path, chat=True, sys_default=None):
    res = json.load(open(out_path)) if Path(out_path).exists() else []
    done = {r["cond"] for r in res}
    for c in conds:
        if c["cond"] in done: continue
        t = time.time()
        with Steerer(model, c.get("vec"), c.get("layer"), c.get("coef", 0.0), c.get("ablate", False), c.get("ablate_vecs")):
            gens = generate(model, tok, prompts, system=c.get("system", sys_default), max_new_tokens=MAXNEW, seed=1234, chat=chat)
        for i, g in enumerate(gens):
            res.append(dict(cond=c["cond"], method=c["method"], alpha=c.get("alpha", 0.0), layer=c.get("layer"), pid=i, text=g))
        jdump(res, out_path)
        print(c["cond"], f"{time.time()-t:.0f}s |", gens[0][:150].replace("\n", " "), flush=True)


if mode in ("pilot", "main"):
    model, tok = load_model(INSTRUCT)
    P = json.load(open(RES / ("pilot_prompts.json" if mode == "pilot" else "steer_prompts.json")))
    prompts = [p["q"] for p in P]
    rng = np.random.default_rng(7)
    conds = [dict(cond="none", method="none")]
    if mode == "pilot":
        for L in [8, 12, 16, 20]:
            D = get_dirs("dirs_instruct.npy", NAMES, L); rn = resid_norm(model, tok, prompts, L)
            for a in [-0.8, -0.4, -0.2, -0.1, 0.2, 0.4]:
                conds.append(dict(cond=f"ai_L{L}_a{a}", method="ai", alpha=a, layer=L, vec=D["ai"], coef=a * rn))
            rv = rng.standard_normal(D["ai"].shape)
            for a in [-0.8, -0.4]:
                conds.append(dict(cond=f"rand_L{L}_a{a}", method="random", alpha=a, layer=L, vec=rv, coef=a * rn))
            print("layer", L, "resid norm", rn)
        run(model, tok, prompts, conds, RES / "gen_pilot.json")
    else:
        cfg = json.load(open(RES / "steer_config.json"))  # chosen from the pilot: layer, alphas
        L, alphas, pos_alphas = cfg["layer"], cfg["alphas"], cfg["pos_alphas"]
        D = get_dirs("dirs_instruct.npy", NAMES, L); rn = resid_norm(model, tok, prompts, L)
        conf = [D[k] for k in ["formality", "fluency", "length_wordlen", "domain_reddit", "assistant_axis", "verbosity"]]
        D["ai_resid"] = resid(D["ai"], conf)
        jdump(dict(layer=L, resid_norm=rn, cos_ai_airesid=float(D["ai"] @ D["ai_resid"])), RES / "steer_meta.json")
        rvecs = [rng.standard_normal(D["ai"].shape) for _ in range(3)]  # 3 fixed random directions (null)
        for a in alphas:
            for m, v in [("ai", D["ai"]), ("ai_resid", D["ai_resid"]), ("formality", D["formality"]), ("assistant_axis", D["assistant_axis"]), ("verbosity", D["verbosity"])]:
                conds.append(dict(cond=f"{m}_a{a}", method=m, alpha=a, layer=L, vec=v, coef=a * rn))
            for k, rv in enumerate(rvecs):
                conds.append(dict(cond=f"random{k}_a{a}", method="random", alpha=a, layer=L, vec=rv, coef=a * rn))
        for a in pos_alphas:
            conds.append(dict(cond=f"ai_a{a}", method="ai", alpha=a, layer=L, vec=D["ai"], coef=a * rn))
            conds.append(dict(cond=f"random0_a{a}", method="random", alpha=a, layer=L, vec=rvecs[0], coef=a * rn))
        # per-layer directional ablation of d_AI (layer i uses the d_AI fitted at layer i)
        allD = np.load(RES / "dirs_instruct.npy")[:, 0]  # [layers 1..27/28, d]; block i output = hidden_states[i+1]
        conds.append(dict(cond="ai_ablate", method="ai_ablate", ablate_vecs=allD))
        # gentler variants (added after ai_ablate broke fluency): single L20 d_AI ablated at every block (refusal-direction style),
        # and per-layer ablation only in blocks 10-27
        conds.append(dict(cond="ai_ablate_single", method="ai_ablate_single", ablate_vecs=np.stack([D["ai"]] * len(allD))))
        mid = np.stack([allD[i] if i >= 10 else np.zeros_like(allD[i]) for i in range(len(allD))])
        conds.append(dict(cond="ai_ablate_mid", method="ai_ablate_mid", ablate_vecs=mid))
        # secondary layer (16) for d_AI and random
        L2 = cfg["layer2"]; D2 = get_dirs("dirs_instruct.npy", NAMES, L2); rn2 = resid_norm(model, tok, prompts, L2)
        for a in alphas:
            conds.append(dict(cond=f"L{L2}_ai_a{a}", method=f"L{L2}_ai", alpha=a, layer=L2, vec=D2["ai"], coef=a * rn2))
            conds.append(dict(cond=f"L{L2}_random0_a{a}", method=f"L{L2}_random", alpha=a, layer=L2, vec=rvecs[0], coef=a * rn2))
        conds.append(dict(cond="prompt_human", method="prompt", system=HUMAN_PROMPT))
        conds.append(dict(cond="prompt_human+ai", method="prompt+ai", system=HUMAN_PROMPT, alpha=alphas[1], layer=L, vec=D["ai"], coef=alphas[1] * rn))
        run(model, tok, prompts, conds, RES / "gen_main.json")
elif mode == "base":
    # E4: base model answers in completion format; steer with the base model's own d_AI
    cfg = json.load(open(RES / "steer_config.json")); L = cfg["layer"]
    model, tok = load_model(BASE)
    P = json.load(open(RES / "steer_prompts.json"))
    prompts = [f"Question: {p['q']}\nAnswer:" for p in P]
    D = get_dirs("dirs_base.npy", ["ai", "formality", "fluency", "length_wordlen", "domain_reddit", "posttrain", "ai_hape", "ai_hc3"], L)
    Di = get_dirs("dirs_instruct.npy", NAMES, L)
    rn = resid_norm(model, tok, prompts, L, chat=False)
    conds = [dict(cond="base_none", method="none")]
    for a in cfg["base_alphas"]:
        conds.append(dict(cond=f"base_ai_a{a}", method="ai", alpha=a, layer=L, vec=D["ai"], coef=a * rn))
        conds.append(dict(cond=f"base_instructdir_a{a}", method="ai_from_instruct", alpha=a, layer=L, vec=Di["ai"], coef=a * rn))
    rng = np.random.default_rng(9)
    rv = rng.standard_normal(D["ai"].shape)
    for a in cfg["base_alphas"]:
        conds.append(dict(cond=f"base_random_a{a}", method="random", alpha=a, layer=L, vec=rv, coef=a * rn))
    jdump(dict(layer=L, resid_norm=rn), RES / "steer_meta_base.json")
    run(model, tok, prompts, conds, RES / "gen_base.json", chat=False)
