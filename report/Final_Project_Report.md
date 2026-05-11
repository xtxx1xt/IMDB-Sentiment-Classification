# IMDB Sentiment Classification

## Approach and Model Design

This project addresses binary sentiment classification on the IMDB movie review dataset. Each review is classified as either positive or negative. Since the input data is text and transformer-based models perform strongly on NLP tasks, the final model uses the lightweight pre-trained `prajjwal1/bert-tiny` model from Hugging Face.

BERT-Tiny is a compact version of the original BERT architecture, with fewer layers and hidden units. This keeps the model efficient while still using contextual word representations learned during pretraining.

The model pipeline is:

- Tokenize each review with the BERT tokenizer.
- Add special tokens such as `[CLS]` and `[SEP]`.
- Pad or truncate each sequence to a fixed maximum length.
- Pass tokenized input through the BERT-Tiny encoder.
- Use the `[CLS]` representation as the review-level embedding.
- Add a fully connected classification layer for binary prediction.

## Data Processing

The dataset was split into 80% training data and 20% validation data. Text preprocessing includes tokenization, padding to a maximum sequence length of 256, conversion to input IDs and attention masks, and label encoding where positive is mapped to `1` and negative is mapped to `0`.

## Hyperparameters

| Setting | Value |
| --- | --- |
| Model | BERT-Tiny |
| Maximum sequence length | 256 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Loss function | CrossEntropyLoss |
| Device | CPU |

## Training and Results

| Metric | Value |
| --- | ---: |
| Final training loss | 0.1288 |
| Validation accuracy | 0.8811 |
| Kaggle test accuracy | 0.8728 |

The model achieved strong performance for a compact architecture, demonstrating the effectiveness of transfer learning for sentiment classification. The validation accuracy indicates that the model generalizes well to unseen validation data, while the Kaggle test result further supports the robustness of the approach.

## Discussion

The model converged quickly within a few epochs, suggesting that fine-tuning a pre-trained transformer can be efficient even with limited computational resources. Pre-trained models perform better than many traditional methods because they capture contextual semantics. Increasing the maximum sequence length may improve performance, but it also increases computation cost. Smaller batch sizes introduce noisier gradients, but they are useful under memory constraints. The number of training epochs also affects whether the model underfits or overfits.

## Conclusion

This project successfully implements a sentiment classifier using BERT-Tiny. The model achieves competitive accuracy while remaining efficient, highlighting the value of transformer-based transfer learning for text classification.
