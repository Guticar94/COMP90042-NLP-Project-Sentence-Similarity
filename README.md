# COMP90042_NLP_Project

The competition rules are specified in: https://github.com/drcarenhan/COMP90042_2024/tree/main

# Automated Fact-Checking System for Climate Science - Project Notebooks

This repository contains the Jupyter notebooks used for the development of an automated fact-checking system to address misinformation in climate science. The project is divided into several notebooks, each focusing on a specific technique, from data preprocessing to model evaluation.

## Notebooks Summary

### 1. Data_Preprocessing.ipynb
**Purpose:** 
- Preprocess the text data to prepare it for the information retrieval and classification tasks.

**Key Steps:**
- Tokenization: Splitting text into tokens.
- Case Folding: Converting text to lowercase.
- Stopword Removal: Removing common words using the NLTK stopwords module.
- Lemmatization: Reducing words to their root forms using WordNetLemmatizer.
- NER & POS Tagging: Enhancing contextual representation using spaCy for Named Entity Recognition (NER) and Part of Speech (POS) tagging.

### 2. Information_Retrieval_Doc2Vec_BM25.ipynb
**Purpose:** 
- Implement and evaluate the Doc2Vec + BM25 model for retrieving relevant evidence.

**Key Steps:**
- Training the Doc2Vec model to capture semantic meanings.
- Applying BM25 for keyword-based document ranking.
- Combining Doc2Vec semantic similarity with BM25 relevance scores.

### 3. Information_Retrieval_SBERT.ipynb
**Purpose:** 
- Implement and evaluate the SBERT model for semantic textual similarity and information retrieval.

**Key Steps:**
- Utilizing Sentence-BERT for sentence embeddings.
- Generating similarity scores for claim-evidence pairs using Cosine Similarity.
- Evaluating the model performance based on retrieval accuracy.

### 4. Information_Retrieval_TFIDF.ipynb
**Purpose:** 
- Implement and evaluate the TF-IDF model for information retrieval.

**Key Steps:**
- Transforming text data into TF-IDF representations.
- Calculating similarity scores using cosine similarity.
- Evaluating the effectiveness of the TF-IDF model in retrieving relevant evidence.

### 5. Multi_Class_Classification_SVM.ipynb
**Purpose:** 
- Develop and evaluate a multi-class classification model using SVM.

**Key Steps:**
- Preprocessing and concatenating data for SVM input.
- Applying PCA for dimensionality reduction.
- Training and evaluating the SVM classifier with RBF kernel.

### 6. Multi_Class_Classification_Doc2Vec_LR.ipynb
**Purpose:** 
- Develop and evaluate a multi-class classification model using Doc2Vec and Logistic Regression.

**Key Steps:**
- Generating dense vector representations using Doc2Vec.
- Training a Logistic Regression model on the vectors.
- Classifying claims into predefined labels based on evidence.

## How to Use
1. Open the notebooks in Jupyter Notebook or JupyterLab.
2. Follow the instructions in each notebook to preprocess data, train models, and evaluate their performance.

## Requirements
Each notebook contains the list of the required libraries.

## License
This project is licensed under the Unimelb License.

---