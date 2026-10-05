"""One fold of a decoder-LLM classifier, LoRA or QLoRA (Instructions sections 6, 9.2, 9.3).

  CUDA_VISIBLE_DEVICES=0 python chpc/train_llm.py --model Qwen/Qwen3-32B --tag q32 --fold 0 --quant nf4 --lr 5e-5

--mode seqcls   : AutoModelForSequenceClassification, head on the last non-pad token (Qwen3).
--mode labeltok : causal LM, prompt ends in "Label:", CE over the logits of tokens "0","1","2" at the
                  last position (Gemma 4 has no sequence-classification class).
Writes build/chpc/runs/<tag>/f<k>/{oof.npy (this fold's rows, in index order), test.npy, log.json, DONE};
chpc/merge.py assembles the 47817x3 / 11955x3 artifacts.
"""
import argparse, json, os, socket, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation: Gemma 31B nf4 peaks at 21.3 GiB, fits 24 GB cards
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")  # fla's tilelang backend crashes nvcc here; use its Triton kernels (Qwen3.5-family linear attention)
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS, NCLS, OUT, ROOT, RUNS, class_weights, load, macro_f1  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--quant", default="none", choices=["none", "nf4"])
p.add_argument("--mode", default="seqcls", choices=["seqcls", "labeltok"])
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--maxlen", type=int, default=192)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--r", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--warmup", type=float, default=0.03)
p.add_argument("--eval-bs", type=int, default=32)
p.add_argument("--limit", type=int, default=0, help="smoke test: train on this many rows only")
p.add_argument("--save", action="store_true", help="save LoRA adapter (+head), tokenizer, run_config.json (Final Instr. 7.2)")
p.add_argument("--prompt", default="default", choices=["default", "q32lt"],
               help="labeltok prompt. q32lt = the Qwen3-32B AWQ fallback prompt of Final Instructions 7.4")
p.add_argument("--pseudo", default="none", choices=["none", "v2"],
               help="v2: add fold f's labelled test rows from build/out/pseudo_v2/pseudo_f{f}.npy to its training set")
p.add_argument("--head", default="default", choices=["default", "norm"],
               help="seqcls only. norm: zero-init linear head on RMS-normalised features (Qwen3's large-norm "
                    "hidden states make the default head's Adam steps overshoot; loss plateaus near 3.7)")
a = p.parse_args()

FULL = a.fold < 0  # --fold -1: one model on ALL train rows (+ >=3/5-vote pseudo rows), test predictions only
out = RUNS / a.tag / ("full" if FULL else f"f{a.fold}")
out.mkdir(parents=True, exist_ok=True)
if (out / "DONE").exists():
    print(f"{out} already DONE"); sys.exit(0)

from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig, DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model  # noqa: E402
import datasets as hfds  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
set_seed(a.seed + max(a.fold, 0))
T0 = time.time()

d = load()
y, folds = d["y"], d["folds"]
tri, vai = (np.arange(len(y)), np.zeros(0, int)) if FULL else (np.where(folds != a.fold)[0], np.where(folds == a.fold)[0])
if a.limit:
    tri = np.sort(np.random.default_rng(0).choice(tri, a.limit, replace=False))
    vai = vai[: max(64, a.limit // 4)]
Xtr, Xte = d["train"], d["test"]
# Final Instructions 3: fold f uses pseudo_f{f} only; validation = fold f's train rows; class weights from real y
PL_IDX = np.zeros(0, int)
if a.pseudo != "none":
    if FULL:  # Final Instructions 7.3: rows that at least 3 of the 5 fold-blind files agree on
        P5 = np.stack([np.load(OUT / f"pseudo_{a.pseudo}" / f"pseudo_f{f}.npy") for f in range(5)])
        cnt = np.stack([(P5 == c).sum(0) for c in range(NCLS)], 1)
        pl = np.where(cnt.max(1) >= 3, cnt.argmax(1), -1)
    else:
        pl = np.load(OUT / f"pseudo_{a.pseudo}" / f"pseudo_f{a.fold}.npy")
    assert pl.shape == (len(Xte),), pl.shape
    PL_IDX = np.where(pl >= 0)[0]
    if a.limit:
        PL_IDX = PL_IDX[: a.limit // 4]
    PL_Y = pl[PL_IDX].astype(int)
    print(f"[{a.tag} f{a.fold}] pseudo {a.pseudo}: +{len(PL_IDX)} test rows, labels {np.bincount(PL_Y, minlength=NCLS).tolist()}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
INSTR = ("Classify this social-media post (Bangla, English or Banglish). "
         "0 = explicitly toxic or hateful, 1 = subtly toxic (sarcasm, coded language), 2 = non-toxic or neutral.")
PRE_TEXT, SUF_TEXT = INSTR + "\nPost: ", "\nLabel:"
if a.prompt == "q32lt":
    PRE_TEXT = "Classify the social-media post as 0 (explicitly toxic/hateful), 1 (subtly toxic) or 2 (neutral).\nPost: "
LABEL_IDS = None

if a.mode == "labeltok":
    tok.padding_side = "left"  # the answer slot is the last position of every row
    PRE = tok(PRE_TEXT)["input_ids"]  # keeps the model's BOS
    SUF = tok(SUF_TEXT, add_special_tokens=False)["input_ids"]
    LABEL_IDS = [tok.encode(c, add_special_tokens=False) for c in "012"]
    assert all(len(t) == 1 for t in LABEL_IDS), f"label digits are not single tokens: {LABEL_IDS}"
    LABEL_IDS = [t[0] for t in LABEL_IDS]

    def encode(texts):  # truncate the post only, so "Label:" always ends the prompt
        body = tok(texts, add_special_tokens=False, truncation=True, max_length=a.maxlen)["input_ids"]
        return [PRE + b + SUF for b in body]
else:
    def encode(texts):
        return tok(texts, truncation=True, max_length=a.maxlen)["input_ids"]

ids_tr = encode([Xtr[i] for i in tri] + [Xte[j] for j in PL_IDX])
y_tr = np.concatenate([y[tri], PL_Y]) if len(PL_IDX) else y[tri]
ids_va = encode([Xtr[i] for i in vai]) if len(vai) else []
ids_te = encode(Xte)
print(f"[{a.tag} f{a.fold}] tokenized: train {len(ids_tr)} val {len(ids_va)} test {len(ids_te)} "
      f"| p99 len {np.percentile([len(x) for x in ids_tr], 99):.0f}", flush=True)

# length bucketing (version-proof; group_by_length changed in transformers 5.x): shuffle, sort inside
# chunks of 64 batches, shuffle the full batches, partial ones last, then read sequentially
rng = np.random.default_rng(a.seed + a.fold)
lens = np.array([len(x) for x in ids_tr])
order, chunk, batches = rng.permutation(len(lens)), a.bs * 64, []
for i in range(0, len(order), chunk):
    c = order[i:i + chunk]
    c = c[np.argsort(lens[c], kind="stable")]
    batches += [c[j:j + a.bs] for j in range(0, len(c), a.bs)]
full = [b for b in batches if len(b) == a.bs]
rng.shuffle(full)
sel = np.concatenate(full + [b for b in batches if len(b) < a.bs])
dtr = hfds.Dataset.from_dict({"input_ids": [ids_tr[i] for i in sel], "labels": y_tr[sel].tolist()})

kw = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
if a.quant == "nf4":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
if a.mode == "seqcls":
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=NCLS, **kw)
    model.config.pad_token_id = tok.pad_token_id  # REQUIRED: the head pools the last non-pad token
    task = "SEQ_CLS"
    if a.head == "norm":
        class NormHead(nn.Module):
            def __init__(self, d, n):
                super().__init__()
                self.lin = nn.Linear(d, n, bias=False)
                nn.init.zeros_(self.lin.weight)  # logits start at 0: loss starts at ln 3

            def forward(self, h):
                h = h.float()
                return self.lin(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6))

        model.score = NormHead(model.config.hidden_size, NCLS).to(model.device)
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    task = "CAUSAL_LM"
print(f"[{a.tag} f{a.fold}] loaded {type(model).__name__} in {time.time()-T0:.0f}s, "
      f"{torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
multimodal = any("language_model" in n for n, _ in model.named_modules())
targets = rf".*language_model.*\.({'|'.join(PROJ)})" if multimodal else PROJ  # skip vision/audio towers
model = get_peft_model(model, LoraConfig(task_type=task, r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                                         target_modules=targets))
for q in model.parameters():  # fp32 master copies for LoRA + head; compute stays bf16 under autocast
    if q.requires_grad:
        q.data = q.data.float()
model.print_trainable_parameters()
CW = torch.tensor(class_weights(y), dtype=torch.float32)


def logits3(m, batch):
    if a.mode == "seqcls":
        return m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    o = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, logits_to_keep=1)
    return o.logits[:, -1, LABEL_IDS]


class WTrainer(Trainer):
    def _get_train_sampler(self, *args, **kwargs):  # keep the bucketed order
        return torch.utils.data.SequentialSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        lg = logits3(model, inputs).float()  # loss on fp32 logits
        loss = nn.functional.cross_entropy(lg, labels, weight=CW.to(lg.device))
        return (loss, {"logits": lg}) if return_outputs else loss


class Progress(TrainerCallback):  # heartbeat the pool/watcher can read
    def on_log(self, args, state, control, logs=None, **kw):
        (out / "progress.json").write_text(json.dumps(
            {"step": state.global_step, "max_steps": state.max_steps, "logs": logs,
             "elapsed_s": round(time.time() - T0)}))


args = TrainingArguments(
    output_dir=str(out / "tmp"), seed=a.seed + a.fold,
    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    learning_rate=a.lr, num_train_epochs=a.epochs, warmup_ratio=a.warmup, lr_scheduler_type="cosine",
    weight_decay=0.01, bf16=True, optim="adamw_torch_fused", max_grad_norm=1.0,
    eval_strategy="no", save_strategy="no", logging_steps=10, report_to=[],
    dataloader_num_workers=2, dataloader_pin_memory=True, disable_tqdm=True, remove_unused_columns=False)
trn = WTrainer(model=model, args=args, train_dataset=dtr, data_collator=DataCollatorWithPadding(tok),
               callbacks=[Progress()])
# compute_loss returns a batch mean, so let Trainer divide by grad-accum steps. Without this, models whose
# forward takes **kwargs (Qwen3 seq-cls) get the SUM of the accum micro-batch losses (logged loss x accum).
trn.model_accepts_loss_kwargs = False
t1 = time.time()
trn.train()
train_min = (time.time() - t1) / 60
print(f"[{a.tag} f{a.fold}] trained in {train_min:.1f} min", flush=True)
if a.save:  # Final Instructions 7.2: adapter (+ classification head via modules_to_save), tokenizer, run config
    art = ARTIFACTS / a.tag / ("full" if FULL else f"fold{a.fold}")
    art.mkdir(parents=True, exist_ok=True)
    trn.model.save_pretrained(str(art))
    tok.save_pretrained(str(art))
    json.dump({"base": a.model, "mode": a.mode, "head": a.head, "max_len": a.maxlen, "quant": a.quant,
               "seed": a.seed + max(a.fold, 0), "fold": "full" if FULL else a.fold, "pseudo": a.pseudo,
               "prompt_prefix": PRE_TEXT if a.mode == "labeltok" else None,
               "prompt_suffix": SUF_TEXT if a.mode == "labeltok" else None,
               "label_token_ids": LABEL_IDS, "label_order": [0, 1, 2], "padding_side": tok.padding_side,
               "clean": "common.clean (Instructions 3.1)", "post_truncation_tokens": a.maxlen,
               "lora": {"r": a.r, "alpha": a.alpha, "dropout": 0.05, "targets": PROJ}},
              open(art / "run_config.json", "w"), indent=1)
    print(f"[{a.tag}] saved adapter to {art}", flush=True)


@torch.inference_mode()
def predict(ids):
    model.eval()
    order = np.argsort([-len(x) for x in ids], kind="stable")  # longest first: an OOM shows up at once
    P = np.zeros((len(ids), NCLS), np.float32)
    for i in range(0, len(order), a.eval_bs):
        idx = order[i:i + a.eval_bs]
        b = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = logits3(model, b)
        P[idx] = torch.softmax(lg.float(), -1).cpu().numpy()
    return P


pv, pt = (predict(ids_va) if len(vai) else np.zeros((0, NCLS), np.float32)), predict(ids_te)
f1, per = macro_f1(y[vai], pv) if len(vai) else (float("nan"), [])
print(f"[{a.tag} f{a.fold}] macro_f1={f1:.4f} per-class={np.round(per, 3).tolist()}", flush=True)
np.save(out / "oof.npy", pv)
np.save(out / "val_idx.npy", vai)
np.save(out / "test.npy", pt)
json.dump({"tag": a.tag, "model": a.model, "fold": a.fold, "macro_f1": f1, "f1_per_class": per,
           "hparams": {k: v for k, v in vars(a).items() if k not in ("tag", "fold")} | {"lora_targets": PROJ},
           "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
           "logged_loss_is_mean": True, "prompt": a.prompt, "saved": a.save, "pseudo": a.pseudo, "n_pseudo": int(len(PL_IDX)), "n_train": len(tri), "n_val": len(vai), "train_minutes": round(train_min, 1),
           "minutes": round((time.time() - T0) / 60, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "history": [h for h in trn.state.log_history if "loss" in h]}, open(out / "log.json", "w"), indent=1)
(out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))  # last: marks the fold complete
