"""Text-level measures used both as confound covariates (corpus) and as output metrics (steering):
formality (s-nlp ranker), supervised AI detectors (desklib DeBERTa-v3-large, fakespot RoBERTa),
cheap stylometrics (length, AI-ism lexicon, markdown, contractions, distinct-2)."""
import re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel, PreTrainedModel

AI_ISMS = ["delve", "tapestry", "testament", "crucial", "essential", "additionally", "furthermore", "moreover",
           "overall", "in conclusion", "it is important to note", "it's important to note", "notably", "vibrant",
           "intricate", "realm", "landscape", "navigate", "foster", "leverage", "comprehensive", "ensure",
           "various", "significant", "enhance", "multifaceted", "nuanced", "underscore", "pivotal", "seamless"]
CONTR = re.compile(r"\b\w+'(t|s|re|ve|ll|d|m)\b", re.I)


def stylometrics(t: str) -> dict:
    w = t.split(); n = max(len(w), 1); low = t.lower()
    toks = re.findall(r"\w+", low); bigr = list(zip(toks, toks[1:]))
    return dict(
        n_words=len(w),
        mean_word_len=float(np.mean([len(x) for x in toks])) if toks else 0.0,
        ai_isms_per100=100 * sum(low.count(a) for a in AI_ISMS) / n,
        markdown_per100=100 * (len(re.findall(r"^\s*([-*•]|\d+\.)\s", t, re.M)) + t.count("**") / 2 + len(re.findall(r"^#+\s", t, re.M))) / n,
        contractions_per100=100 * len(CONTR.findall(t)) / n,
        first_person_per100=100 * len(re.findall(r"\b(i|i'm|i've|my|me)\b", low)) / n,
        distinct2=len(set(bigr)) / max(len(bigr), 1),
        non_ascii_frac=sum(ord(c) > 127 for c in t) / max(len(t), 1),
    )


class DesklibAIDetectionModel(nn.Module):
    """Custom head from the desklib model card: mean-pooled DeBERTa-v3 + linear -> logit(AI).
    Weights loaded manually from the checkpoint's safetensors (model.* backbone, classifier.*)."""

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, name):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        m = cls(AutoConfig.from_pretrained(name))
        sd = load_file(hf_hub_download(name, "model.safetensors"))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "position_ids" not in k], missing
        return m

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


class Scorers:
    def __init__(self, device="cuda", which=("formality", "desklib", "fakespot")):
        self.dev, self.m = device, {}
        if "formality" in which:
            n = "s-nlp/roberta-base-formality-ranker"
            self.m["formality"] = (AutoTokenizer.from_pretrained(n), AutoModelForSequenceClassification.from_pretrained(n).to(device).eval())
        if "fakespot" in which:
            n = "fakespot-ai/roberta-base-ai-text-detection-v1"
            self.m["fakespot"] = (AutoTokenizer.from_pretrained(n), AutoModelForSequenceClassification.from_pretrained(n).to(device).eval())
        if "desklib" in which:
            n = "desklib/ai-text-detector-v1.01"
            self.m["desklib"] = (AutoTokenizer.from_pretrained(n), DesklibAIDetectionModel.from_pretrained(n).to(device).eval())

    @torch.no_grad()
    def score(self, name, texts, bs=32, max_len=512):
        tk, md = self.m[name]; out = []
        for i in range(0, len(texts), bs):
            enc = tk(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.dev)
            if name == "desklib":
                p = torch.sigmoid(md(enc.input_ids, enc.attention_mask).squeeze(-1))
            else:
                p = torch.softmax(md(**enc).logits.float(), -1)[:, 1]  # formal / AI prob
            out.append(p.float().cpu().numpy())
        return np.concatenate(out)
