PROJECT TITLE

AI-Powered SMS Spam Detection System with Explainable AI

Project Overview

Spam SMS messages are unsolicited and unwanted messages that may contain advertisements, phishing links, or fraudulent information. These messages cause inconvenience to users and can lead to serious financial and privacy risks.

This project aims to build an **AI-powered SMS Spam Detection System** that automatically classifies SMS messages as **Spam** or **Ham (Legitimate)** using Machine Learning and Natural Language Processing techniques.

Problem Statement

Telecom operators receive millions of SMS messages daily. Spam messages cause fraud, privacy risks, and poor user experience.
This project builds an AI system that automatically classifies SMS messages as Spam or Ham (legitimate) and explains why a message is spam.

Objective:

To design and implement a machine learning model that can accurately classify SMS messages as spam or non-spam and help telecom providers improve message filtering.

DATASET DETAILS:

Total messages: 5,574
Labels:
spam → unwanted / promotional / fraud messages
ham → legitimate messages
File name :
spam.csv


Solution Approach:


1. Load and explore the SMS dataset  
2. Clean and preprocess text data  
   - Lowercasing  
   - Removing punctuation  
   - Removing stopwords  
3. Convert text into numerical features using **TF-IDF Vectorization**  
4. Train a **Naive Bayes Machine Learning model**  
5. Evaluate model performance using accuracy, precision, recall, and confusion matrix  
6. Predict whether a new SMS message is spam or ham



Technologies Used:


- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorizer  
- Naive Bayes Classifier


Evaluation Metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix



Project Structure:


47_BILADUGU-RAMANJAMMA/
│
├── data/
│ └── spam.csv
│
├── src/
│ ├── train.py
│ └── predict.py
│
├── README.md
└── requirements.txt


Expected Outcome:

- A trained machine learning model capable of detecting spam SMS  
- High classification accuracy  
- Reduced spam message impact for telecom users

Future Enhancements:

- Deploy the model as a web application using Flask or FastAPI  
- Add Explainable AI (XAI) for better transparency  
- Integrate deep learning models for improved accuracy

 Conclusion:
 This project demonstrates how Machine Learning and NLP can be effectively used to solve real-world telecom problems like SMS spam detection. The solution is scalable and can be integrated into telecom systems to improve customer experience.
