"""Assemble per-fold partials into the contract artifacts (Instructions 3.2) and run the 3.2 checks.

  python chpc/merge.py q32            # -> build/out/{oof,test,log}_q32.*   (partial folds allowed)
"""
import json, os, sys, time
import numpy as np
from common import NCLS, OUT, RUNS, load, macro_f1

d = load()
y, folds = d["y"], d["folds"]


def merge(tag):
    done = sorted(int(p.parent.name[1:]) for p in (RUNS / tag).glob("f*/DONE"))
    if not done:
        print(f"{tag}: no finished folds"); return None
    oof = np.zeros((len(y), NCLS), np.float32)
    tests, per_fold = [], {}
    for f in done:
        fd = RUNS / tag / f"f{f}"
        vai = np.load(fd / "val_idx.npy")
        assert np.array_equal(vai, np.where(folds == f)[0]), f"{tag} f{f}: rows are not fold {f}"
        oof[vai] = np.load(fd / "oof.npy")
        tests.append(np.load(fd / "test.npy"))
        per_fold[str(f)] = json.loads((fd / "log.json").read_text())
    test = np.mean(tests, 0).astype(np.float32)
    # the 3.2 pre-send check
    assert oof.shape == (47817, 3) and test.shape == (11955, 3)
    m = oof.sum(1) > 0
    assert np.allclose(oof[m].sum(1), 1, atol=1e-3) and np.allclose(test.sum(1), 1, atol=1e-3)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    f1, per = macro_f1(y[m], oof[m])
    first = per_fold[str(done[0])]
    log = {"tag": tag, "model": first["model"], "folds": done, "oof_macro_f1": f1, "f1_per_class": per,
           "n_oof": int(m.sum()), "fold_macro_f1": {k: v["macro_f1"] for k, v in per_fold.items()},
           "hparams": first["hparams"], "gpu": sorted({v["gpu"] for v in per_fold.values()}),
           "minutes": {k: v["minutes"] for k, v in per_fold.items()},
           "history": {k: v["history"] for k, v in per_fold.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / f".lock_{tag}"
    for _ in range(120):  # two folds finishing together must not interleave their writes
        try:
            lock.mkdir(); break
        except FileExistsError:
            time.sleep(1)
    try:
        for name, arr in (("oof", oof), ("test", test)):
            tmp = OUT / f"{name}_{tag}.tmp{os.getpid()}.npy"
            np.save(tmp, arr)
            tmp.rename(OUT / f"{name}_{tag}.npy")
        tmp = OUT / f"log_{tag}.json.tmp{os.getpid()}"
        tmp.write_text(json.dumps(log, indent=1))
        tmp.rename(OUT / f"log_{tag}.json")
    finally:
        lock.rmdir()
    print(f"{tag} OOF macro F1 = {f1:.4f} on {m.sum()} rows, folds {done}, per-class {np.round(per, 3).tolist()}")
    return f1


if __name__ == "__main__":
    for t in sys.argv[1:]:
        merge(t)
