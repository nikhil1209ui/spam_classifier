1. SMS/Email Spam Classifier using several classification models and nltk
2. Dataset Used : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
3. Framework : Streamlit
4. Web Deployment : Heroku (working app: https://spammclassifier-5b7c52be0207.herokuapp.com/)

<img width="574" alt="image" src="https://github.com/user-attachments/assets/b4af64cb-a176-4d01-ab0b-32cbe08f7b66"><img width="575" alt="image" src="https://github.com/user-attachments/assets/ec249cad-a92b-4169-84b4-52eb78edf843">

Spam Classifier
-
This project is a machine learning-based spam classifier built using Python, Jupyter Notebook, and Streamlit. The classifier identifies whether an input message (email or SMS) is spam or not based on text processing and feature extraction techniques. The model employs several machine learning algorithms to find the best-performing classifier for this task, and it is deployed as an interactive web application using Streamlit.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Table of Contents:
-
Installation

Project Overview

Dataset

Preprocessing

Model Training

Evaluation

Streamlit Web App

Usage

Acknowledgments

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Installation
-
Clone this repository:

`git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier`


Install the required packages:

`pip install -r requirements.txt`

Download the NLTK stopwords:

`import nltk
nltk.download('stopwords')
nltk.download('punkt')`

Ensure that you have Streamlit installed to run the web app:

`pip install streamlit`

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Overview
-

Data Loading: The dataset (spam.csv) is loaded and preprocessed by removing unnecessary columns and renaming the main columns to "target" and "text."
Text Preprocessing: The text data is transformed by tokenizing, removing stopwords and punctuation, and stemming.

Feature Extraction: Both Bag of Words (BOW) and TF-IDF vectorization techniques are used to convert the processed text into numerical data.
Model Training: A variety of models (Naive Bayes, SVM, XGBoost, etc.) are trained and evaluated using both BOW and TF-IDF features to find the best classifier.

Evaluation: The performance of each model is evaluated using accuracy and precision scores.

Deployment: The model and vectorizer are saved and deployed in a Streamlit app.

Dataset:
The dataset used is spam.csv, containing labeled data with "spam" or "ham" messages. The main column is text, and the target column is binary (1 for spam, 0 for non-spam).
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Preprocessing
-
The preprocessing involves the following steps:

Text Transformation: Convert text to lowercase, tokenize, remove alphanumeric characters, stopwords, and punctuation, and apply stemming.

Feature Engineering: Add basic features such as character count, word count, and sentence count.

Feature Extraction: Use Count Vectorizer (BOW) and TF-IDF vectorization techniques.

Model Training:
A variety of classifiers were tested for spam detection, including:

Naive Bayes (Gaussian, Multinomial, Bernoulli)

Support Vector Machine (SVM)

XGBoost

Logistic Regression

K-Nearest Neighbors

Decision Trees and Extra Trees

Ensemble Methods: Random Forest, Gradient Boosting, AdaBoost, and Bagging Classifiers

Each classifier was trained using both BOW and TF-IDF vectors to determine the optimal model.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Evaluation
-
Model performance is evaluated using accuracy and precision scores, with a detailed performance comparison for each model. The classifier that achieves the best balance between precision of 100% and accuracy of around 98% is used in the final application.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Streamlit Web App
-
The trained model and TF-IDF vectorizer are saved using pickle for deployment in a web app built with Streamlit. Users can input a message to predict if it’s spam or not.

Steps in the Streamlit App:
-
Text Preprocessing: The app transforms the input message similar to the training phase.

Vectorization: TF-IDF vectorization is applied to convert the message into a format suitable for the model.

Prediction: The classifier outputs whether the message is spam or not.

Usage
To run the Streamlit web app:

`streamlit run app.py`

In the app, users can input a message in the text area, click "Predict," and the app will display whether the message is classified as spam or not.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Acknowledgments
-
This project uses Scikit-Learn, XGBoost, Streamlit, and NLTK libraries. Thanks to the open-source community for providing these tools.
