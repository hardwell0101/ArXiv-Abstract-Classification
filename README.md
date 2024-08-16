# ArXiv Abstract Classification

This project develops a machine learning model to automatically classify scientific paper abstracts into different research categories, focusing on AI, machine learning, and computational neuroscience.

## Project Overview

- Data Collection: Abstracts collected from arXiv API
- Data Preprocessing: Tokenization, stop word removal, and lemmatization
- Feature Extraction: TF-IDF Vectorization
- Model Development: Random Forest and Support Vector Machine (SVM) classifiers
- Model Evaluation: Comparison of model performances
- User Interface: Simple interface for classifying new abstracts

## Files in this Repository

- `AA.ipynb`: Jupyter notebook containing the main project code
- `arxiv_abstracts.csv`: Raw data collected from arXiv
- `preprocessed_abstracts.csv`: Preprocessed abstract data
- `svm_classifier.joblib`: Trained SVM model
- `tfidf_vectorizer.joblib`: Fitted TF-IDF vectorizer
- `tfidf_features.csv`: TF-IDF feature names

## How to Use

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Open and run the `AA.ipynb` notebook in Jupyter

## Results

- SVM classifier outperformed Random Forest, achieving 93.33% accuracy
- Successfully distinguished between AI, ML, and Computational Neuroscience abstracts
- Key features identified include terms like "neuron", "algorithm", and "network"

## Future Improvements

- Expand the dataset to include more categories and abstracts
- Experiment with deep learning models like BERT for potentially higher accuracy
- Develop a web application for real-time abstract classification

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

## Acknowledgements
This project uses data from arXiv.org, an open-access archive for scholarly articles.
