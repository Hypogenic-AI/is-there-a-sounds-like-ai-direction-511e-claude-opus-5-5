"""Generation-based confound directions in Qwen2.5-7B-Instruct (persona-vector style):
 - assistant axis (cheap rebuild of Lu et al. 2601.10387): default assistant vs 40 role-play system prompts
 - verbosity: 'be very brief' vs 'be very detailed' system prompts
Activations = mean residual over *response* tokens (all layers), read in the full chat context."""
import json, random
from common import *

set_seed(1)
model, tok = load_model(INSTRUCT)
qs = [json.loads(l)["question"] for l in open(ROOT / "code/assistant-axis/data/extraction_questions.jsonl")]
random.shuffle(qs)
roles = json.load(open(ROOT / "code/assistant-axis/data/roles/role_list.json"))
skip = {"assistant", "tutor", "translator", "editor", "counselor", "researcher", "programmer", "analyst", "consultant", "teacher"}
cand = [r for r in roles if r not in skip and (ROOT / f"code/assistant-axis/data/roles/instructions/{r}.json").exists()]
random.shuffle(cand)
role_sel = cand[:40]


@torch.no_grad()
def response_acts(systems, users, responses, bs=24):
    """Mean residual over response tokens for each (system, user, response) triple."""
    out = []
    for i in range(0, len(users), bs):
        texts, starts = [], []
        for s, u, r in zip(systems[i:i+bs], users[i:i+bs], responses[i:i+bs]):
            msgs = ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            starts.append(len(tok(p, add_special_tokens=False).input_ids)); texts.append(p + r)
        tok.padding_side = "right"
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=512).to(model.device)
        o = model(**enc, output_hidden_states=True)
        m = enc.attention_mask.clone()
        for j, s0 in enumerate(starts): m[j, :s0] = 0
        mf = m.unsqueeze(-1).float()
        hs = torch.stack(o.hidden_states, 1).float()
        out.append(((hs * mf.unsqueeze(1)).sum(2) / mf.sum(1).clamp(min=1).unsqueeze(1)).half().cpu().numpy())
    tok.padding_side = "left"
    return np.concatenate(out)

recs = {}
# ---- assistant axis
Q = qs[:20]
sys_r, u_r, meta_r = [], [], []
for r in role_sel:
    ins = json.load(open(ROOT / f"code/assistant-axis/data/roles/instructions/{r}.json"))["instruction"]
    for k, q in enumerate(Q):
        sys_r.append(ins[k % len(ins)]["pos"]); u_r.append(q); meta_r.append(r)
def gen_items(systems, users, max_new=150, bs=50, seed=0):
    outs = []
    for i in range(0, len(users), bs):
        msgs = [([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}] for s, u in zip(systems[i:i+bs], users[i:i+bs])]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

resp_r = gen_items(sys_r, u_r)
sys_d = [None] * (len(Q) * 5); u_d = Q * 5
resp_d = gen_items(sys_d, u_d, seed=100)
A_role = response_acts(sys_r, u_r, resp_r); A_def = response_acts(sys_d, u_d, resp_d)
# ---- verbosity
VQ = qs[20:120]
SB = "Answer in one or two short sentences. Be extremely brief."
SL = "Give a long, detailed, thorough and comprehensive answer covering every aspect."
resp_b = gen_items([SB] * len(VQ), VQ, seed=200); resp_l = gen_items([SL] * len(VQ), VQ, seed=300, max_new=200)
A_b = response_acts([SB] * len(VQ), VQ, resp_b); A_l = response_acts([SL] * len(VQ), VQ, resp_l)
np.savez(ACTS / "confound_gen.npz", role=A_role, default=A_def, brief=A_b, long=A_l)
jdump(dict(roles=role_sel, role_samples=[dict(role=m, q=u, r=x) for m, u, x in list(zip(meta_r, u_r, resp_r))[:40:4]],
           default_samples=resp_d[:5], brief=resp_b[:5], long=resp_l[:5],
           role_meta=meta_r), RES / "confound_gen_samples.json")
print("done", A_role.shape, A_def.shape, A_b.shape, A_l.shape)
