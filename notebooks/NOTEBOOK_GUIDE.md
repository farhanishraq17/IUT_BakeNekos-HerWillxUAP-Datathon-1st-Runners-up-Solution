# Notebook guide: which notebook was used when

## The submission this package reproduces

| | |
|---|---|
| Kaggle submission | **56638488**, 2026-09-28 10:57 UTC, team IUT BakeNekos |
| Score | **private LB 0.76000** (our best), public LB 0.73957 |
| File | `submission/submission.csv` |
| Made by | `Main code Notebook/HerWILL2026_Main_Stack_v10.ipynb` (writes a byte-identical file; checked in its section 7) |

## How the notebooks depend on each other

```
Supporting 1 - Gemma 4 31B (g31)               ──▶ outputs/{oof,test,log}_g31, _g31-s2, _g31-s3, _g31-s4  ─┐
Supporting 2 - Qwen3-32B (q32)                 ──▶ outputs/{oof,test,log}_q32, _q32-s2                   ─┤
Supporting 3 - multilingual-e5-large (me5L)    ──▶ outputs/{oof,test,log}_me5L-s1, _me5L-s2, _me5L-s3     ─┼─▶ Main code Notebook ─▶ submission.csv
Supporting 4 - TF-IDF logistic regression      ──▶ outputs/{oof,test}_tfidf                               ─┘
```

Run the four supporting notebooks first (in any order: they are independent), then the main notebook. The
predictions they produced are already in each `outputs/` folder, so the main notebook runs on its own.

## Which notebook produced which file, and when (UTC)

| Notebook | Files (in its `outputs/`) | Used by the main notebook | Trained | Hardware |
|---|---|---|---|---|
| Supporting 1 · `g31_Gemma4_31B_QLoRA_finetune.ipynb` | `g31` (seed 42) · `g31-s2` · `g31-s3` · `g31-s4` | yes, averaged into member `g31` | 2026-09-27 16:48 → 09-28 02:31 | 1 GPU per fold: H200, A100 80GB, RTX A6000, RTX 6000 Ada |
| Supporting 2 · `q32_Qwen3_32B_QLoRA_finetune.ipynb` | `q32` (seed 42) · `q32-s2` | yes, member `q32` | 2026-09-27 16:29 → 20:13 | H200, RTX 6000 Ada, RTX A6000, RTX PRO 4000 |
| Supporting 3 · `me5L_e5_large_finetune.ipynb` | `me5L-s1` (seed 42) · `me5L-s2` · `me5L-s3` | yes, member `me5L` | 2026-09-27 16:44 → 18:31 | RTX PRO 4000 Blackwell |
| Supporting 4 · `tfidf_logreg.ipynb` | `tfidf` | yes, member `tfidf` | 2026-09-27 (committed 19:02) | CPU |
| Main · `HerWILL2026_Main_Stack_v10.ipynb` | `submission.csv` | – | 2026-09-28 10:30 → 10:57 (submitted) | CPU |

## Runs added after the competition (for the fine-tuned weights)

The original fine-tuning runs saved their predictions, not their weights. So that the package contains our
fine-tuned weights, seed 42 of each model was trained again on **2026-09-28, 17:23 → 19:25 UTC**
with the same notebooks, code, recipe and folds, plus `--save`:

| Re-run | Of | In `outputs/rerun_seed42/` | Weights | OOF macro F1 (original → re-run) |
|---|---|---|---|---|
| `g31w` | `g31` seed 42 | `{oof,test,log}_g31w` | `fine_tuned_models/g31_gemma4_31b_qlora_seed42/` | 0.6239 → 0.6240 |
| `q32w` | `q32` seed 42 | `{oof,test,log}_q32w` | `fine_tuned_models/q32_qwen3_32b_qlora_seed42/` | 0.5902 → 0.5915 |
| `me5Lw` | `me5L-s1` seed 42 | `{oof,test,log}_me5Lw` | `fine_tuned_models/me5L_e5_large_seed42/` | 0.5958 → 0.5936 |
| TF-IDF | `tfidf` | `outputs/rerun/` | `fine_tuned_models/tfidf_logreg/` (in the package) | 0.5219 → 0.5222 |

These re-runs are **not** used by the main notebook (the submission used the original predictions). GPU training is
not bit-deterministic, so a re-run matches its original closely but not exactly.

## Not included

Other experiments that did not feed the best private submission (pseudo-labelled Gemma / e5 runs, XLM-R, MuRIL,
BanglaBERT, IndicBERT, Qwen3-14B, context-aware e5, zero-shot LLM judges, other stack versions) are left out, as asked.
