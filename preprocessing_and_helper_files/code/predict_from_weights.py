"""Predict class probabilities with one saved fine-tuned fold model (the weights in build/artifacts/<tag>/fold<k>/).

  python chpc/predict_from_weights.py --weights build/artifacts/g31w/fold0 --texts posts.csv --out probs.npy
  python chpc/predict_from_weights.py --weights build/artifacts/g31w/fold0 --fold-rows 0 --limit 256 --out p.npy

--texts: a CSV with a `text` column (cleaned here with common.clean, as in training).
--fold-rows k: instead, the validation rows of pinned fold k (to check against the saved out-of-fold predictions).
The model type is read from run_config.json: Gemma label-token (LoRA), Qwen3 classification head (LoRA) or a fully
fine-tuned encoder. The output is an (n, 3) float32 array of softmax probabilities in label order 0, 1, 2.
"""
import argparse, json, os, pathlib, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import clean, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--texts")
ap.add_argument("--fold-rows", type=int)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", required=True)
a = ap.parse_args()
W = pathlib.Path(a.weights)
cfg = json.loads((W / "run_config.json").read_text())

if a.texts:
    import pandas as pd
    texts = [clean(t) for t in pd.read_csv(a.texts).text]
else:
    d = load()
    texts = [d["train"][i] for i in np.where(d["folds"] == a.fold_rows)[0]]
if a.limit:
    texts = texts[: a.limit]

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(W))
dev = "cuda" if torch.cuda.is_available() else "cpu"
if "base" in cfg and cfg.get("mode") in ("labeltok", "seqcls"):  # LoRA adapter on a 4-bit base model (train_llm.py)
    from peft import PeftModel
    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if cfg.get("quant") == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if cfg["mode"] == "labeltok":
        base = AutoModelForCausalLM.from_pretrained(cfg["base"], **kw)
        tok.padding_side = "left"
        pre = tok(cfg["prompt_prefix"])["input_ids"]
        suf = tok(cfg["prompt_suffix"], add_special_tokens=False)["input_ids"]
        body = tok(texts, add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
        ids = [pre + b + suf for b in body]
    else:
        base = AutoModelForSequenceClassification.from_pretrained(cfg["base"], num_labels=3, **kw)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base.config.pad_token_id = tok.pad_token_id
        ids = tok(texts, truncation=True, max_length=cfg["max_len"])["input_ids"]
    model = PeftModel.from_pretrained(base, str(W))
else:  # fully fine-tuned encoder (train_enc.py)
    model = AutoModelForSequenceClassification.from_pretrained(str(W), dtype=torch.float32).to(dev)
    cfg["mode"] = "encoder"
    ids = tok(texts, truncation=True, max_length=cfg["max_len"])["input_ids"]
model.eval()

P = np.zeros((len(ids), 3), np.float32)
order = np.argsort([-len(x) for x in ids], kind="stable")
with torch.inference_mode():
    for i in range(0, len(order), a.bs):
        idx = order[i:i + a.bs]
        b = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            if cfg["mode"] == "labeltok":
                lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False,
                           logits_to_keep=1).logits[:, -1, cfg["label_token_ids"]]
            else:
                lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        P[idx] = torch.softmax(lg.float(), -1).cpu().numpy()
np.save(a.out, P)
print(f"wrote {a.out} {P.shape} from {W} ({cfg['mode']})")
