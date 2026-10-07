# Network Intrusion Detection with Neural Networks

A senior project exploring supervised network intrusion detection with the flow records in CIC-IDS2017 `MachineLearningCSV.zip`. The main model is a feed-forward multilayer perceptron (MLP). Experiments will compare it with simpler machine-learning baselines and evaluate how performance changes when attack types or capture scenarios are held out from training.

This project is intentionally distinct from a text phishing detector: it uses tabular network-flow measurements, multiclass attack labels, and scenario-aware evaluation rather than TF-IDF features and email text.

## Project layout

```text
data/
  raw/                 # Place the original archive here (not committed)
src/ids/
  data/                # Loading, cleaning, splitting, and preprocessing
  models/              # MLP and comparison models
  training/            # Reproducible training entry points
  evaluation/          # Metrics, plots, and reports
```

## Research questions

1. How does an MLP compare with classical tabular baselines on CIC-IDS2017?
2. How does performance change when test records come from a capture day or scenario excluded from training?
3. Which attack classes are confused with benign traffic, and how does class imbalance affect results?

Use a conventional stratified train/validation/test split as a reference, then add a separate grouped or temporal holdout. For the generalization experiment, define the held-out scenario or attack family before fitting preprocessing or models. Do not use the held-out data for feature selection, scaling, tuning, or early stopping. Report both known-class classification performance and the held-out-class detection results; a closed-set classifier cannot assign a truly unseen attack label without an explicit unknown-detection rule.

## Dataset setup

1. Download CIC-IDS2017 and place `MachineLearningCSV.zip` in `data/raw/`.
2. Record the archive source/version and the files selected for each experiment.
3. Inspect column names and labels before combining CSVs. CIC-IDS2017 files can contain inconsistent whitespace, duplicate rows, non-finite values, and highly imbalanced labels.
4. Keep raw data unchanged. Store any cleaned or model-ready outputs under `data/`.

Never commit the dataset, extracted flow data, trained model weights, or machine-specific paths. See `.gitignore` for the intended exclusions.

## Experimental practice

- Fit imputation, encoding, and scaling on training data only; apply the fitted transforms to validation and test data.
- Remove identifiers and leakage-prone fields (for example, flow IDs, IP addresses, timestamps, and source-file names) from model inputs unless an experiment explicitly studies their effect.
- Keep class labels and scenario metadata separate from model features.
- Use a fixed seed and save configuration, class mapping, split definition, and metrics with each run.
- Report per-class precision/recall/F1, macro-F1, confusion matrix, and balanced accuracy alongside accuracy. State the split strategy and class distribution.
- Compare an MLP with at least one simple baseline (such as logistic regression or a tree ensemble) under identical splits and preprocessing rules.

## Development

The source tree is organized into preparation, model, training, and evaluation components. Add implementations under `src/ids/` as the experiment design is finalized.
