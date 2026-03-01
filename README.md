# 🚀 Suggestify – ML-Based Purchase Prediction & Recommendation System

Suggestify is a supervised machine learning web application that predicts customer buying probability and provides personalized product recommendations in real time.

The system uses a Random Forest classifier trained on demographic and behavioral features such as gender, location, and search queries. The trained model is integrated into a Flask-based web interface with a modern dark-themed UI.

---

## 🎯 Features

- 🔍 Purchase probability prediction
- 🤖 Random Forest classification model
- 🧹 Data preprocessing & label encoding
- 🛡 Feature selection to prevent data leakage
- 📊 Stratified train-test split
- ⭐ Rating-based product filtering
- 🖼 Product image rendering
- 🌙 Modern responsive UI
- 🚀 Deployment-ready architecture

---

## 🧠 Machine Learning Pipeline

1. Data cleaning and preprocessing  
2. Label encoding of categorical variables  
3. Removal of non-predictive features (e.g., ratings, image URLs)  
4. Stratified train-test split  
5. Random Forest model training  
6. Probability-based prediction  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Flask  
- HTML5 & CSS3  

---


```bash
pip install -r requirements.txt
python app.py
