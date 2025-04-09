# IMDB Reviews - Sentiment Analysis

**ReviewSentimentAnalysis** is a sentiment classification project for movie reviews. It uses models like **TF-IDF + Logistic Regression**, **LightGBM**, and **BERT** to classify reviews as either positive or negative. This project includes custom preprocessing and is tested on both a movie review dataset and custom reviews.

## 🎬 Project Statement

The **Film Junky Union**, a community of movie enthusiasts, is developing a system to automatically categorize movie reviews as positive or negative. The goal is to achieve an **F1 score** of at least **0.85** on the **IMDB reviews dataset**.

## 🚀 Getting Started

### Prerequisites

Install the necessary Python libraries:

```bash
pip install -r requirements.txt
Libraries in requirements.txt:

- pandas
- scikit-learn
- nltk
- spaCy
- lightgbm
- transformers
- matplotlib/ seaborn for visualizations
- torch #PyTorch for BERT model

Make sure to download the spaCy model.
```

## 🧠 Models Used
- TF-IDF + Logistic Regression
- spaCy + TF-IDF + Logistic Regression
- spaCy + TF-IDF + LightGBM
- BERT + Logistic Regression (very small sample)

## ⚙️ BERT Usage
For BERT, I used the bert-base-uncased model to extract embeddings from a random subset of 500 reviews. These embeddings were used to train a Logistic Regression model. This method provides a compact, efficient solution to leverage BERT’s powerful contextual representations.

## 📊 Results

| Model                                  | Test Accuracy | F1 Score | ROC AUC | APS   |
|----------------------------------------|---------------|----------|---------|-------|
| **NLTK + TF-IDF + Logistic Regression**| 0.88          | 0.88     | 0.95    | 0.95  |
| **spaCy + TF-IDF + Logistic Regression**| 0.87          | 0.87     | 0.95    | 0.94  |
| **spaCy + TF-IDF + LightGBM**          | 0.86          | 0.87     | 0.94    | 0.94  |
| **BERT + Logistic Regression**         | 0.51          | 0.51     | 0.52    | 0.52  |

## 🏆 Insights
NLTK + TF-IDF + Logistic Regression outperformed other models with an F1 score of 0.88.

BERT showed strong training accuracy but overfitted on the test set due to the small training sample.

LightGBM provided fast and reliable results, though it performed similarly to the Logistic Regression models.

## 📈 Next Steps
- Experimenting with additional models or data
- Fine-tuning hyperparameters
- Improving preprocessing steps
