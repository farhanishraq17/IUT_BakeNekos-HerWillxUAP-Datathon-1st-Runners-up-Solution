"""One fold of an encoder classifier (Instructions section 5 / 9.1): AdamW, wd 0.01, cosine, warmup 6%,
bf16, class-weighted CE, eval once per epoch, keep the FINAL epoch.

  CUDA_VISIBLE_DEVICES=0 python chpc/train_enc.py --model FacebookAI/xlm-roberta-large --tag xlmrL-s7 --fold 0 --seed 7
"""
import argparse, json, os, socket, sys, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS, NCLS, OUT, ROOT, RUNS, class_weights, load, macro_f1, neighbour_texts  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--maxlen", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--normalize", action="store_true", help="csebuetnlp normalizer (BanglaBERT models only, 9.4)")
p.add_argument("--limit", type=int, default=0, help="smoke test: train on this many rows only")
p.add_argument("--pseudo", default="none", choices=["none", "v2"],
               help="v2: add fold f's labelled test rows from build/out/pseudo_v2/pseudo_f{f}.npy to its training set")
p.add_argument("--save", action="store_true", help="save the fine-tuned model + tokenizer + run_config.json to build/artifacts/<tag>/fold<k>")
p.add_argument("--ctx", action="store_true",
               help="Final Instructions 8 (me5Lctx): text pair (post, 'previous: {prev} | next: {next}') with the "
                    "neighbours in id order over train+test (texts only), truncation only_second")
a = p.parse_args()

out = RUNS / a.tag / f"f{a.fold}"
out.mkdir(parents=True, exist_ok=True)
if (out / "DONE").exists():
    print(f"{out} already DONE"); sys.exit(0)

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,  # noqa: E402
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
import datasets as hfds  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
set_seed(a.seed + a.fold)
T0 = time.time()
d = load()
y, folds = d["y"], d["folds"]
Xtr, Xte = d["train"], d["test"]
if a.normalize:
    from normalizer import normalize
    Xtr, Xte = [normalize(t) for t in Xtr], [normalize(t) for t in Xte]
tri, vai = np.where(folds != a.fold)[0], np.where(folds == a.fold)[0]
if a.limit:
    tri = np.sort(np.random.default_rng(0).choice(tri, a.limit, replace=False))
    vai = vai[: max(64, a.limit // 4)]

# Final Instructions 3: fold f uses pseudo_f{f} only; validation = fold f's train rows; class weights from real y
PL_IDX, PL_Y = np.zeros(0, int), np.zeros(0, int)
if a.pseudo != "none":
    pl = np.load(OUT / f"pseudo_{a.pseudo}" / f"pseudo_f{a.fold}.npy")
    assert pl.shape == (len(Xte),), pl.shape
    PL_IDX = np.where(pl >= 0)[0]
    if a.limit:
        PL_IDX = PL_IDX[: a.limit // 4]
    PL_Y = pl[PL_IDX].astype(int)
    print(f"[{a.tag} f{a.fold}] pseudo {a.pseudo}: +{len(PL_IDX)} test rows, labels {np.bincount(PL_Y, minlength=NCLS).tolist()}", flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
N_CUT = 0
if a.ctx:  # second segment of every row; the same function serves train and test rows
    PREV_TR, NEXT_TR, PREV_TE, NEXT_TE = neighbour_texts(d)
    CTX_TR = [f"previous: {pv_} | next: {nx_}" for pv_, nx_ in zip(PREV_TR, NEXT_TR)]
    CTX_TE = [f"previous: {pv_} | next: {nx_}" for pv_, nx_ in zip(PREV_TE, NEXT_TE)]


def tokenize(texts, ctxs):
    global N_CUT
    if ctxs is None:
        return tok(list(texts), truncation=True, max_length=a.maxlen)["input_ids"]
    # the post is never cut by only_second; the rare post longer than maxlen-16 tokens is pre-cut so the pair fits
    texts, body = list(texts), tok(list(texts), add_special_tokens=False)["input_ids"]
    for i, b in enumerate(body):
        if len(b) > a.maxlen - 16:
            texts[i], N_CUT = tok.decode(b[:a.maxlen - 16]), N_CUT + 1
    return tok(texts, list(ctxs), truncation="only_second", max_length=a.maxlen)["input_ids"]


enc = lambda texts, ctxs=None, labels=None: hfds.Dataset.from_dict(
    {"input_ids": tokenize(texts, ctxs), **({"labels": labels.tolist()} if labels is not None else {})})
C = (lambda idx: [CTX_TR[i] for i in idx]) if a.ctx else (lambda idx: None)
CT = (lambda idx: [CTX_TE[j] for j in idx]) if a.ctx else (lambda idx: None)
dtr = enc([Xtr[i] for i in tri] + [Xte[j] for j in PL_IDX], (C(tri) + CT(PL_IDX)) if a.ctx else None,
          np.concatenate([y[tri], PL_Y]).astype(int))
dva, dte = enc([Xtr[i] for i in vai], C(vai), y[vai]), enc(Xte, CT(range(len(Xte))))
_l = [len(x) for x in dtr["input_ids"]]
print(f"[{a.tag} f{a.fold}] tokenized: train {len(dtr)} | p50 {np.percentile(_l, 50):.0f} p99 {np.percentile(_l, 99):.0f} "
      f"max {max(_l)} | posts pre-cut {N_CUT}", flush=True)
CW = torch.tensor(class_weights(y), dtype=torch.float32)


class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out_ = model(**inputs)
        lg = out_.logits.float()  # loss in fp32 always
        loss = nn.functional.cross_entropy(lg, labels, weight=CW.to(lg.device))
        return (loss, out_) if return_outputs else loss


class Progress(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        (out / "progress.json").write_text(json.dumps(
            {"step": state.global_step, "max_steps": state.max_steps, "logs": logs,
             "elapsed_s": round(time.time() - T0)}))


metrics = lambda ev: {"macro_f1": macro_f1(ev.label_ids, ev.predictions)[0]}
# dtype=float32: transformers 5 otherwise loads the checkpoint's own dtype (bug #1); bf16 autocast on top
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=NCLS, dtype=torch.float32)
args = TrainingArguments(
    output_dir=str(out / "tmp"), seed=a.seed + a.fold, per_device_train_batch_size=a.bs,
    per_device_eval_batch_size=a.bs * 4, learning_rate=a.lr, num_train_epochs=a.epochs, warmup_ratio=0.06,
    lr_scheduler_type="cosine", weight_decay=0.01, bf16=True, optim="adamw_torch_fused",
    eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to=[],
    dataloader_num_workers=2, disable_tqdm=True)
trn = WTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
               data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics, callbacks=[Progress()])
t1 = time.time()
trn.train()
train_min = (time.time() - t1) / 60
if a.save:  # the final-epoch weights that produce this fold's oof/test predictions below
    art = ARTIFACTS / a.tag / f"fold{a.fold}"
    art.mkdir(parents=True, exist_ok=True)
    trn.model.save_pretrained(str(art), safe_serialization=True)
    tok.save_pretrained(str(art))
    json.dump({"base": a.model, "fold": a.fold, "seed": a.seed + a.fold, "max_len": a.maxlen, "ctx": a.ctx,
               "label_order": [0, 1, 2], "clean": "common.clean", "hparams": vars(a)}, open(art / "run_config.json", "w"), indent=1)
    print(f"[{a.tag}] saved model to {art}", flush=True)
sm = lambda ds: torch.softmax(torch.tensor(trn.predict(ds).predictions).float(), -1).numpy()
pv, pt = sm(dva), sm(dte)
f1, per = macro_f1(y[vai], pv)
print(f"[{a.tag} f{a.fold}] macro_f1={f1:.4f} per-class={np.round(per, 3).tolist()} train {train_min:.1f} min", flush=True)
np.save(out / "oof.npy", pv.astype(np.float32))
np.save(out / "val_idx.npy", vai)
np.save(out / "test.npy", pt.astype(np.float32))
json.dump({"tag": a.tag, "model": a.model, "fold": a.fold, "macro_f1": f1, "f1_per_class": per,
           "hparams": {k: v for k, v in vars(a).items() if k not in ("tag", "fold")},
           "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
           "pseudo": a.pseudo, "n_pseudo": int(len(PL_IDX)), "ctx": a.ctx, "saved": a.save, "n_posts_precut": N_CUT,
           "tok_p99": float(np.percentile(_l, 99)), "n_train": len(tri), "n_val": len(vai), "train_minutes": round(train_min, 1),
           "minutes": round((time.time() - T0) / 60, 1),
           "history": [h for h in trn.state.log_history if "loss" in h or "eval_macro_f1" in h]},
          open(out / "log.json", "w"), indent=1)
(out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
