#  ArXiv Abstract Classification

A machine learning pipeline that automatically classifies scientific paper abstracts into research categories.

---

##  Project Overview

This project walks through a complete ML workflow:

1. **Data Collection** — Abstracts gathered via the arXiv API
2. **Preprocessing** — Tokenization, stop word removal, and lemmatization
3. **Feature Extraction** — TF-IDF vectorization
4. **Modeling** — Random Forest and Support Vector Machine (SVM) classifiers
5. **Evaluation** — Side-by-side comparison of model performance
6. **Interface** — Simple UI for classifying new abstracts

---

##  Repository Structure

| File | Description |
|------|-------------|
| `AA.ipynb` | Main Jupyter notebook with the full pipeline |
| `arxiv_abstracts.csv` | Raw abstracts collected from arXiv |
| `preprocessed_abstracts.csv` | Cleaned and preprocessed abstract data |
| `svm_classifier.joblib` | Trained SVM model |
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `tfidf_features.csv` | TF-IDF feature names |

---

##  Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook AA.ipynb
```

---

##  Results

| Model | Accuracy |
|-------|----------|
| Random Forest | — |
| **SVM (best)** | **93.33%** |

The SVM classifier successfully distinguished between all three categories. Key discriminative features include terms like *"neuron"*, *"algorithm"*, and *"network"*.

---


##  Dependencies

```
Python 3.x  •  pandas  •  scikit-learn  •  nltk  •  matplotlib  •  seaborn  •  joblib  •  ipywidgets
```

---

##  Acknowledgements

This project uses data from [arXiv.org](https://arxiv.org), an open-access archive for scholarly articles.
