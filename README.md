
# IMDb Sentiment Analysis

This project aims to build a sentiment analysis model for classifying IMDb movie reviews into three categories: **positive**, **negative**, and **neutral**. The model uses a **Recurrent Neural Network (RNN)** built with **Keras** and **TensorFlow**. The project leverages deep learning techniques and natural language processing (NLP) to process and classify text data effectively.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
This project is designed to predict the sentiment of a movie review by classifying it as **positive**, **negative**, or **neutral** using a deep learning model. The dataset consists of over **50,000 IMDb movie reviews**. We split the data into **80% training** and **20% testing** for model evaluation.

## Dataset
The dataset is comprised of IMDb movie reviews, labeled as positive, negative, or neutral. For this project, the dataset is split into two parts:
- **Training set**: 40,000 reviews (80%)
- **Test set**: 10,000 reviews (20%)

You can access the dataset directly from the [IMDb dataset](https://www.kaggle.com/). If you have any difficulties accessing it, please feel free to open an issue or refer to the "Usage" section below.

## Model Architecture
The sentiment analysis model uses a **Simple Recurrent Neural Network (RNN)** with the following architecture:
- **Embedding layer**: Converts words into vector representations.
- **RNN layer**: Processes the sequential text data.
- **Dense layer**: A fully connected layer to produce the output.
- **Activation function**: Softmax activation function for multi-class classification (positive, negative, neutral).

The model is trained using **Keras** with **TensorFlow** as the backend.

## Preprocessing
- **Tokenization**: Text data is tokenized into words using Keras' Tokenizer API.
- **Padding**: Padding is applied to make all sequences have the same length.
- **Data Cleaning**: Removed punctuation, stopwords, and applied text normalization techniques.

## Model Evaluation
- **Accuracy**: The model achieved an **88% accuracy** on the test set.
- **F1-Score**: The F1-score of **0.85** indicates a good balance between precision and recall in the classification task.
- **Test Cases**: The model was tested using diverse reviews that contain mixed, sarcastic, and extreme sentiments.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**:
  - **Keras** for building the RNN model.
  - **TensorFlow** as the backend for Keras.
  - **Pandas** for data handling and preprocessing.
  - **NumPy** for numerical computations.
  - **Matplotlib** and **Seaborn** for visualizing data and results.
  - **Jupyter Notebooks** for the development and experimentation environment.

## Installation
1. Clone this repository:
   
   ```bash
   git clone https://github.com/ShivamMehta02/IMDB_sentimnet_analysis
