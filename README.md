# IMDB Sentiment Classification with BERT-Tiny

This project fine-tunes transformer-based models for binary sentiment classification on IMDB movie reviews. The main model uses `prajjwal1/bert-tiny` from Hugging Face, with a DistilBERT baseline included for comparison.

## Features

- Hugging Face Transformers model loading and tokenization
- PyTorch fine-tuning pipeline with custom `Dataset` and `DataLoader`
- Train/validation split for model evaluation
- Validation accuracy and macro F1 evaluation
- Kaggle-style prediction file generation

## Results

| Model | Validation Accuracy | Notes |
| --- | ---: | --- |
| BERT-Tiny (`prajjwal1/bert-tiny`) | 88.11% | Main lightweight model |
| DistilBERT (`distilbert-base-uncased`) | 87.80% | Baseline experiment |

Reported Kaggle test accuracy for the BERT-Tiny run: **87.28%**.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── train_bert_tiny.ipynb
├── train_distilbert_baseline.ipynb
├── prediction.csv
├── data/
│   └── README.md
└── report/
    ├── Final_Project_Report.md
    └── Final_Project_Report.docx
```

## Data

The raw Kaggle CSV files are intentionally not committed because they are large and should be downloaded from the original dataset source.

Expected local file placement:

```text
data/train.csv
data/test.csv
```

The notebooks read the dataset from those paths.

## Setup

Python 3.10 or 3.11 is recommended for the pinned PyTorch and Transformers versions used in the original experiment.

```bash
pip install -r requirements.txt
```

Then open and run:

```text
train_bert_tiny.ipynb
```

The notebook trains the model, evaluates it on the validation split, and writes predictions to `prediction.csv`.

## Notes

- `train_bert_tiny.ipynb` is the main final model notebook.
- `train_distilbert_baseline.ipynb` is kept as a baseline experiment.
- `prediction.csv` is included as the generated submission file.
- The project report is available in `report/Final_Project_Report.md`.
