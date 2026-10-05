"""Shared data contract for the CHPC runs: preprocessing + pinned folds, copied from
'Instructions for Bhaiya.md' section 3.1. Every model imports this, so OOF rows line up."""
import json, os, pathlib, re, unicodedata
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
# HERWILL_DATA / HERWILL_WORK relocate the inputs and outputs (used by the final package's notebooks); the defaults
# are the repo layout
DATA = pathlib.Path(os.environ.get("HERWILL_DATA", ROOT / "Data" / "Competition Data"))
WORK = pathlib.Path(os.environ.get("HERWILL_WORK", ROOT / "build"))
OUT = WORK / "out"                      # final oof_/test_/log_<tag> artifacts
RUNS = WORK / "chpc" / "runs"           # per-fold partials: runs/<tag>/f<k>/{oof,test}.npy, log.json
CACHE = WORK / "chpc" / "data.json"
ARTIFACTS = WORK / "artifacts"          # saved fine-tuned weights: artifacts/<tag>/fold<k>/
NCLS = 3

URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
WS = re.compile(r"\s+")


def clean(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = URL.sub(" <url> ", s)
    s = USER.sub(" <user> ", s)
    return WS.sub(" ", s).strip() or "<empty>"


def build_cache():
    """Rows sorted by id; StratifiedKFold(5, shuffle, 42). Written once as json so the vLLM env
    (no pandas / sklearn) reads exactly the same texts, labels and folds."""
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    tr = pd.read_csv(DATA / "train.csv").sort_values("id").reset_index(drop=True)
    te = pd.read_csv(DATA / "test.csv").sort_values("id").reset_index(drop=True)
    y = tr.y.values
    folds = np.full(len(tr), -1)
    for i, (_, v) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(tr, y)):
        folds[v] = i
    assert np.bincount(folds).tolist() == [9564, 9564, 9563, 9563, 9563], np.bincount(folds)
    d = {"train_id": tr.id.tolist(), "test_id": te.id.tolist(), "y": y.tolist(), "folds": folds.tolist(),
         "train_raw": tr.text.astype(str).tolist(), "test_raw": te.text.astype(str).tolist(),
         "train": [clean(t) for t in tr.text], "test": [clean(t) for t in te.text]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False))
    tmp.rename(CACHE)
    return d


def load():
    d = json.loads(CACHE.read_text()) if CACHE.exists() else build_cache()
    d["y"], d["folds"] = np.array(d["y"]), np.array(d["folds"])
    return d


def neighbour_texts(d):
    """Final Instructions 8 / OMG_FINAL_sub.md 3 (copied): previous and next post in the id order of train + test
    together, whichever split they are in. Texts only, never labels. Returns PREV_TR, NEXT_TR, PREV_TE, NEXT_TE."""
    import pandas as pd
    tr = pd.DataFrame({"id": d["train_id"], "text": d["train_raw"]})
    te = pd.DataFrame({"id": d["test_id"], "text": d["test_raw"]})
    al = pd.concat([tr[["id", "text"]].assign(split=0, row=np.arange(len(tr))),
                    te[["id", "text"]].assign(split=1, row=np.arange(len(te)))]).sort_values("id").reset_index(drop=True)
    txt = [clean(t) for t in al.text]
    prev_txt = [""] + txt[:-1]
    next_txt = txt[1:] + [""]
    PREV_TR, NEXT_TR = np.empty(len(tr), object), np.empty(len(tr), object)
    PREV_TE, NEXT_TE = np.empty(len(te), object), np.empty(len(te), object)
    for k, (s, r) in enumerate(zip(al.split.values, al.row.values)):
        if s == 0: PREV_TR[r], NEXT_TR[r] = prev_txt[k], next_txt[k]
        else:      PREV_TE[r], NEXT_TE[r] = prev_txt[k], next_txt[k]
    assert (al.id.values == np.arange(len(al))).all()  # ids 0..59771, contiguous
    return PREV_TR, NEXT_TR, PREV_TE, NEXT_TE


def class_weights(y):
    return len(y) / (NCLS * np.bincount(y, minlength=NCLS))  # w_c = N / (3 * n_c)


def macro_f1(y, p):
    """Macro F1 in numpy (the vLLM env has no sklearn); matches sklearn average='macro'."""
    pred = p.argmax(1) if p.ndim == 2 else p
    f = []
    for c in range(NCLS):
        tp = np.sum((pred == c) & (y == c))
        fp, fn = np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), [float(x) for x in f]


if __name__ == "__main__":
    d = build_cache()
    print("cache written:", CACHE, "folds", np.bincount(d["folds"]).tolist())
