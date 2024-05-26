# COMP90042_NLP_Project

The competition rules are specified in: https://github.com/drcarenhan/COMP90042_2024/tree/main

# Evidence-Based Climate Science: Developing an Automated Fact-Checking System - Project Notebooks

This repository contains the Jupyter notebooks used for the development of an automated fact-checking system to address misinformation in climate science. The project is divided into several notebooks, each focusing on a specific technique, from data preprocessing to model evaluation.

## Notebooks Summary

### 1. SBERT.ipynb
**Purpose:** 
- Implement and evaluate the SBERT model for semantic textual similarity and information retrieval.

**Key Steps:**
- Utilizing Sentence-BERT for sentence embeddings.
- Generating similarity scores for claim-evidence pairs using Cosine Similarity.
- Evaluating the model performance based on retrieval accuracy.

### 2. SBERT_similarity.ipynb
**Purpose:** 
- Further refine and test the SBERT model for improved semantic similarity measures.

**Key Steps:**
- Fine-tuning SBERT model parameters.
- Conducting similarity scoring on enhanced datasets.
- Comparing the refined model's performance with initial SBERT implementation.

### 3. TF-IDF + POS + NER base.ipynb
**Purpose:** 
- Implement and evaluate a TF-IDF model augmented with Part of Speech (POS) and Named Entity Recognition (NER) tagging.

**Key Steps:**
- Preprocessing text data with POS and NER tagging using spaCy.
- Transforming text into TF-IDF representations.
- Calculating similarity scores using cosine similarity.
- Evaluating the effectiveness of the TF-IDF model in retrieving relevant evidence.

### 4. Doc2Vec_v2.ipynb
**Purpose:** 
- Implement and evaluate the Doc2Vec + BM25 model for retrieving relevant evidence and enhancing classification tasks.

**Key Steps:**
- Training the Doc2Vec model to capture semantic meanings.
- Generating vector representations for claims and evidence.
- Evaluating model performance based on retrieval accuracy and classification metrics.

### 5. TF-IDF + SVM classifier.ipynb
**Purpose:** 
- Develop and evaluate a multi-class classification model using TF-IDF representations and an SVM classifier.

**Key Steps:**
- Preprocessing and transforming text data into TF-IDF representations.
- Applying Principal Component Analysis (PCA) for dimensionality reduction.
- Training an SVM classifier with RBF kernel on the TF-IDF features.
- Evaluating model performance based on classification accuracy and F1 score.

## How to Use
1. Open the notebooks in Jupyter Notebook or JupyterLab.
2. Follow the instructions in each notebook to preprocess data, train models, and evaluate their performance.

## Requirements
Each notebook contains the list of the required libraries.

## License
This project is licensed under the Unimelb License.