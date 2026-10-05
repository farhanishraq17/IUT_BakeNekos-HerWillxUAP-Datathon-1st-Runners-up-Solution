# IUT BakeNekos · HerWILL X UAP Datathon 2026 · 1st Runners-up solution

3-class harmful-content classification of Bangla, English and Banglish social-media posts
(`0` explicitly toxic or hateful · `1` subtly toxic: sarcasm, coded language · `2` non-toxic or neutral), scored on
macro F1.

**Best private-leaderboard submission: 0.76000** (public 0.73957), Kaggle submission 56638488, 2026-09-28 10:57 UTC.
This package reproduces it.

**Report, slides and paper:** [Research report](report/IUT_BakeNekos_HerWILL2026_Research_Report.pdf) ·
[Presentation](report/IUT_BakeNekos_Presentation.pdf) · [IEEE-format paper](report/IUT_BakeNekos_IEEE_Paper.pdf)

![Pipeline overview](report/figures/pipeline_overview.png)

## 1. Run it (only the file paths change)

1. Install Python 3.11 and `pip install -r requirements.txt`.
2. Copy the competition files `train.csv`, `test.csv`, `sample_submission.csv` into `data/`.
3. Open **`notebooks/Main code Notebook/HerWILL2026_Main_Stack_v10.ipynb`** and *Run All* (a few minutes on a CPU).

The notebook writes `outputs/submission.csv` and checks that it is byte-identical to `submission/submission.csv`, the
file we submitted. **To run it anywhere else, only the paths in its first code cell (section 0) change**: `DATA_DIR`
(competition CSVs), `MEMBER_DIRS` (the four model-prediction folders), `OUT_DIR` and `SUBMITTED`. On Kaggle, for example,
`DATA_DIR = Path("/kaggle/input/<competition folder>")` and `MEMBER_DIRS` pointing at an uploaded copy of the
`outputs/` folders. The same holds for every supporting notebook: its section 0 is the only cell with paths.

## 2. What is in the package

```
README.md                         this file
requirements.txt                  main + TF-IDF notebooks (CPU)
requirements-training.txt         fine-tuning notebooks (GPU)
submission/submission.csv         the submitted CSV (private 0.76000)
data/                             put the competition CSVs here (not redistributed)
outputs/                          the main notebook writes submission.csv here
report/                           research report, presentation, IEEE-format paper (PDF) + figures/
notebooks/
  NOTEBOOK_GUIDE.md               which notebook produced which file, and when
  Main code Notebook/
    HerWILL2026_Main_Stack_v10.ipynb          main notebook: stacker -> calibration -> submission.csv
  Supporting 1 - Gemma 4 31B (g31)/
    g31_Gemma4_31B_QLoRA_finetune.ipynb       fine-tuning notebook (4 seeds x 5 folds)
    outputs/                                  its predictions used by the main notebook (+ rerun_seed42/)
  Supporting 2 - Qwen3-32B (q32)/
    q32_Qwen3_32B_QLoRA_finetune.ipynb        fine-tuning notebook (2 seeds x 5 folds)
    outputs/
  Supporting 3 - multilingual-e5-large (me5L)/
    me5L_e5_large_finetune.ipynb              fine-tuning notebook (3 seeds x 5 folds)
    outputs/
  Supporting 4 - TF-IDF logistic regression (tfidf)/
    tfidf_logreg.ipynb                        trains + saves the 5 fold models (CPU)
    outputs/
preprocessing_and_helper_files/
  code/
    common.py                     preprocessing clean(), pinned folds, class weights, macro F1 (shared by all models)
    train_llm.py                  one fold of Gemma 4 31B (label-token) or Qwen3-32B (classification head), 4-bit QLoRA
    train_enc.py                  one fold of an encoder (multilingual-e5-large)
    merge.py                      5 fold outputs -> oof_/test_/log_<run> (test = mean of the fold models)
    predict_from_weights.py       predict with a saved fold model
  fine_tuned_models/
    MODELS.md                     where every fine-tuned model's weights are, formats, how to load, provenance
    g31_gemma4_31b_qlora_seed42/  fold0..4: adapter + run config (weights: Kaggle dataset)
    q32_qwen3_32b_qlora_seed42/   fold0..4 (weights: Kaggle dataset)
    me5L_e5_large_seed42/         fold0..4 (weights: Kaggle dataset)
    tfidf_logreg/                 fold0..4.joblib (in full)
```

## 3. The solution

**Validation.** Rows sorted by `id`; one fixed `StratifiedKFold(5, shuffle=True, random_state=42)` split for every
model. Each out-of-fold (OOF) prediction comes from a model that never saw that row, so the OOF matrices of different
models can be stacked honestly. Test predictions are the mean of the 5 fold models.

**Preprocessing** (`common.py: clean()`): Unicode NFKC normalisation, URLs → `<url>`, @handles → `<user>`,
whitespace collapsed, empty → `<empty>`. Emojis and punctuation are kept (they carry the sarcasm signal of class 1).
The TF-IDF model lower-cases the raw post itself.

**Level 1: four text models, fine-tuned on the competition data** (one supporting notebook each):

| Member | Model | Seeds | OOF macro F1 |
|---|---|---|---|
| `g31` | Gemma 4 31B, 4-bit QLoRA (r 16), classifies through the logits of `0`/`1`/`2` after `Label:` | 4 | 0.6248 |
| `me5L` | multilingual-e5-large, full fine-tuning, 3 epochs | 3 | 0.5996 |
| `q32` | Qwen3-32B, 4-bit QLoRA + classification head | 2 | 0.5965 |
| `tfidf` | TF-IDF word 1–2 + char 2–5 grams, logistic regression | – | 0.5219 |

All neural models use class-weighted cross-entropy (N / (3·n_c)) on fp32 logits; the exact settings are in each notebook.

**Level 2: the stacker** (main notebook). A gradient-boosting classifier on the same 5 folds over 98 features: the
members' probabilities (12); the label mix of the neighbouring *training* rows in id order (windows of 1 to 2,500
rows, never including the row itself); v10's id-distance kernel label mixes (neighbours weighted by `exp(-gap/τ)`);
and label-free context (language and length). Then per-class probability multipliers tuned for macro F1, and the
test posts that appear verbatim in train with one label get that label.


**Transparency.** The id features use how the dataset was assembled (ids in collection order across train and test;
neighbouring posts share labels 61% of the time against 41% by chance). Only the provided data is used, and a row
never sees its own label. The text-only models are reported next to the stack.

## 4. Re-training the models (optional, GPU)

Each fine-tuning notebook runs as a report by default (`RUN_TRAINING = False`: results, loss curves and seed
comparisons from the saved runs). Set `RUN_TRAINING = True` and it fine-tunes every fold of every seed again with
the same script, saving each fold's weights, and refreshes its `outputs/`. Measured cost per fold: Gemma 4 31B
100–220 min, Qwen3-32B 85–135 min, e5-large about 12 min, on one 24–80 GB GPU. `pip install -r requirements-training.txt`
first, and accept the Gemma licence on Hugging Face.

## 5. Fine-tuned weights

See `preprocessing_and_helper_files/fine_tuned_models/MODELS.md`: public Kaggle datasets
([Gemma](https://www.kaggle.com/datasets/farhanishraqq/herwill-g31-gemma4-31b-qlora-seed42), [Qwen](https://www.kaggle.com/datasets/farhanishraqq/herwill-q32-qwen3-32b-qlora-seed42), [e5-large](https://www.kaggle.com/datasets/farhanishraqq/herwill-e5-large-finetuned-seed42)) with one folder per fold, plus the TF-IDF fold models
in the package. The submitted runs saved predictions only, so these weights are seed-42 re-runs of the same recipe
(2026-09-28, after the deadline); MODELS.md reports how closely they match.

## 6. Exactness

- Main notebook: byte-identical to the submitted CSV with scikit-learn 1.9.1 (checked when the package was built).
- Fine-tuning notebooks: same code, recipe and seeds; GPU training is not bit-deterministic, so re-training gives
  close but not identical predictions (e.g. e5-large seed 42: OOF 0.5958 → 0.5936).
- TF-IDF notebook: OOF 0.5222 here vs 0.5219 for the saved predictions (another machine's library build); 99.6% of
  test predictions identical.

## 7. Licences

Code dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, joblib, torch, nbformat, nbclient, ipykernel
(BSD-3-Clause); transformers, peft, accelerate, datasets, tokenizers (Apache-2.0); bitsandbytes (MIT). The base models
(`google/gemma-4-31B`, `Qwen/Qwen3-32B`, `intfloat/multilingual-e5-large`) and our fine-tuned weights built on them
fall under the base models' licences; see each model card on Hugging Face.
