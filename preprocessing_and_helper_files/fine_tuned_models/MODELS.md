# Fine-tuned models

Every level-1 model of the main notebook (stack v10) was fine-tuned / trained on the competition data only, 5 times
(one model per fold of `StratifiedKFold(5, shuffle=True, random_state=42)` over train rows sorted by `id`). Each fold
model predicts its held-out fold (the out-of-fold predictions) and the test set.

**Where the weights are.** The three neural models are public Kaggle datasets (too large for this archive); this
folder keeps each fold's configuration files and a `FILES_ON_KAGGLE.json` listing the weight files and sizes. The
TF-IDF models are here in full.

| Model | Weights (5 folds) | Format | Size | Load with |
|---|---|---|---|---|
| Gemma 4 31B QLoRA label-token (`g31`, seed 42) | [herwill-g31-gemma4-31b-qlora-seed42](https://www.kaggle.com/datasets/farhanishraqq/herwill-g31-gemma4-31b-qlora-seed42) | PEFT LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`) on `google/gemma-4-31B` in 4-bit NF4 | 2.5 GB (5 × 490 MB) | `PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, quantization_config=nf4), foldK)` |
| Qwen3-32B QLoRA + head (`q32`, seed 42) | [herwill-q32-qwen3-32b-qlora-seed42](https://www.kaggle.com/datasets/farhanishraqq/herwill-q32-qwen3-32b-qlora-seed42) | PEFT LoRA adapter incl. the classification head, on `Qwen/Qwen3-32B` in 4-bit NF4 | 2.6 GB (5 × 520 MB) | `PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(base, num_labels=3, quantization_config=nf4), foldK)` |
| multilingual-e5-large (`me5L`, seed 42) | [herwill-e5-large-finetuned-seed42](https://www.kaggle.com/datasets/farhanishraqq/herwill-e5-large-finetuned-seed42) | Full model, `model.safetensors` (fp32) + `config.json` + tokenizer | 11 GB (5 × 2.2 GB fp32) | `AutoModelForSequenceClassification.from_pretrained(foldK)` |
| TF-IDF + logistic regression (`tfidf`) | `tfidf_logreg/fold0-4.joblib` (this folder) | joblib bundle: word + char `TfidfVectorizer` and `LogisticRegression` | 24 MB | `joblib.load("foldK.joblib")` |

`code/predict_from_weights.py --weights <foldK folder> --texts posts.csv --out probs.npy` reads `run_config.json`
(base model, prompt, label-token ids, max length) and returns the 3-class probabilities for any of the three neural
models. Label order everywhere: `0` explicitly toxic/hateful, `1` subtly toxic, `2` non-toxic/neutral.

## Provenance (please read)

- The **submitted** predictions came from fine-tuning runs on 2026-09-27 that saved predictions only. Their weights
  were not kept.
- The weights here are **seed-42 re-runs** trained on 2026-09-28 (after the competition closed) with the same code,
  recipe, folds and seed. They reproduce the recipe, not the exact bits: GPU training is not deterministic.
  Out-of-fold macro F1, original → re-run: Gemma 0.6239 → 0.6240, Qwen 0.5902 → 0.5915,
  e5-large 0.5958 → 0.5936.
- The TF-IDF fold models come from the TF-IDF notebook's run in this package (scikit-learn 1.9.1): OOF 0.5222 vs
  0.5219 for the submitted predictions, 99.6% identical test predictions.



Checked: `predict_from_weights.py` on the first 256 validation posts of fold 0 reproduces the saved out-of-fold predictions (same predicted class for 100.0% of posts, max probability difference 0.0078).
`predict_from_weights.py` was also checked on earlier adapters of the same two LLMs, trained with the same code: Gemma 4 31B label-token (`g31PL-s1` fold 0) and Qwen3-32B label-token (`q32LT` fold 0) each reproduced their saved out-of-fold predictions for 254 of the first 256 fold-0 posts (max probability difference 0.031, bf16 rounding). The Qwen classification-head path (`q32w`) has not been run with this script yet.

Base-model weights are not included; they download from Hugging Face (`google/gemma-4-31B` needs its licence accepted
there). Use of the fine-tuned weights is subject to each base model's licence.
