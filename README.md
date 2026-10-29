# DASE7506 MP2 — Reliable LLM Reasoning Selection

## Overview

This repository contains the implementation and reproduction instructions for DASE7506 Mini-Project 2: **Selecting Reliable LLM Reasoning**.

The task is to select the correct answer from four candidate solutions generated for the same mathematical question.

Our final method combines:

- **Whole-trace embeddings** to represent candidate reasoning.
- **Self-attention** to compare all four candidates.
- **Majority-vote features** based on final-answer agreement.
- **Arithmetic-error detection** to identify explicit calculation mistakes.
- **Final-answer consistency checking** to identify contradictions between a candidate's calculations and final answer.

The final model achieves:

| Metric | Result |
|---|---|
| Validation accuracy | **92.98%** |
| Test accuracy | **79.01%** |
| Checkpoint size | 1.26 MiB |
| Test inference time | Approximately 1.5 seconds |

## 1. Environment Setup

The experiments were conducted using Python 3.12 on CPU.

**Requirements:**
- Python 3.12
- PyTorch
- NumPy

The dependency versions are specified in `requirements.txt`.

From the repository root, navigate to the code directory:

```bash
cd code
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

All subsequent commands should be executed from `code/`.

## 2. Reproducing the Experiments

The experiments follow the development of the final model, starting with a majority-vote baseline and progressively introducing learned reasoning representations, candidate comparison, and explicit consistency features.

### Experiment 1 — Majority Vote Baseline

This baseline selects a candidate based on the most frequently occurring final answer among the four solutions.

No training is required.

```bash
python show.py \
  --method baseline_vote.py \
  --split validation \
  --output-dir runs/show_vote
```

**Expected validation accuracy:** 86.22%

### Experiment 2 — MLP with Reasoning Embeddings and Voting

This experiment trains an MLP using the provided whole-trace embeddings and a majority-vote feature.

Train the model:

```bash
python train2.py \
  --output-dir runs/mlp_vote
```

Evaluate the trained model:

```bash
python show.py \
  --method my_model.py \
  --checkpoint runs/mlp_vote/model.pt \
  --split validation \
  --output-dir runs/show_mlp_vote
```

**Expected validation accuracy:** 91.73%

### Experiment 3 — Self-Attention

This experiment introduces a self-attention architecture to model relationships between the four candidate solutions rather than scoring them independently.

```bash
python train3.py \
  --model attention \
  --output-dir runs/set_attention
```

**Reported validation accuracy:** 92.11% without dropout.

### Experiment 4 — Attention with Arithmetic-Error Detection

This experiment adds a binary feature indicating whether a candidate contains an explicit arithmetic error.

```bash
python train4.py \
  --model attention \
  --calc-binary \
  --no-consistency \
  --dropout 0.3 \
  --output-dir runs/set_calc_attention_drop03
```

**Reported validation accuracy:** 92.86%

### Experiment 5 — Final Model

The final model combines self-attention with three additional features:

1. Majority-vote agreement.
2. Arithmetic-error count.
3. Final-answer consistency.

Dropout is used during training to reduce overfitting.

```bash
python train4.py \
  --model attention \
  --dropout 0.3 \
  --output-dir runs/set_feat_attention_dropout03
```

**Expected validation accuracy:** 92.98%

The checkpoint is saved at:

```text
runs/set_feat_attention_dropout03/model.pt
```

## 3. Running the Final Model

The final inference implementation is provided in `my_model4.py`, which exposes:

```python
predict(bank, arrays, checkpoint)
```

To generate predictions on the test split using the final checkpoint:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py \
  --method my_model4.py \
  --checkpoint runs/set_feat_attention_dropout03/model.pt \
  --split test \
  --output runs/test-predictions.json
```

The predicted candidate indices are saved to:

```text
runs/test-predictions.json
```

**Reported test accuracy:** 79.01%

The test accuracy was measured using the official scorer. The prediction command generates the prediction file; it does not itself guarantee that the reported accuracy will be displayed.

## 4. Experimental Results

The main experimental results are summarized below.

| Method | Validation Accuracy |
|---|---:|
| Majority vote | 86.22% |
| MLP + embeddings + vote | 91.73% |
| Self-attention (no dropout) | 92.11% |
| Attention + arithmetic-error feature | 92.86% |
| **Final model** | **92.98%** |

These results illustrate the effect of progressively incorporating reasoning representations, cross-candidate interactions, and explicit consistency features.

## 5. Implementation Notes

The final model uses the following settings:

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Hidden dimension | 128 |
| Learning rate | 0.01 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Maximum epochs | 100 |
| Dropout (final model) | 0.3 |
| Optimizer | Adam |

The implementation uses only the supplied training data and fixed embeddings. No external model weights, additional datasets, or inference-time LLM calls are required.

## 6. Important Files

| File | Description |
|---|---|
| `baseline_vote.py` | Majority-vote baseline |
| `train2.py` | MLP training |
| `train3.py` | Candidate-comparison model training |
| `train4.py` | Final training pipeline and feature extraction |
| `my_model.py` | MLP inference implementation |
| `my_model4.py` | Final inference implementation |
| `show.py` | Validation evaluation |
| `predict.py` | Prediction entry point |
| `requirements.txt` | Python dependencies |

## 7. Reproducibility Notes

- All paths are relative to the `code/` directory.
- The experiments use the provided training and validation splits.
- Validation accuracy is used for model selection.
- The final checkpoint is generated using seed 42, dropout 0.3, arithmetic-error count, final-answer consistency, and majority-vote features.
- Results may show minor numerical variations across environments.
- The reported results correspond to the experiments described in the project report.

For further discussion of the research hypotheses, ablation studies, and error analysis, please refer to the accompanying project report.

## 8. Acknowledgments

This project was developed for DASE7506 — Advanced Machine Learning Methods, Fall 2026.

The course-provided dataset, embeddings, and evaluation tools were used in accordance with the project requirements.

Claude Code (Anthropic) was used as a coding and analysis assistant during development.
