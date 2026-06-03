# 🎬 Movie Recommendation System (SVD)

## 📌 Overview
This is a machine learning-based movie recommendation system that suggests movies to users based on their past ratings. It uses matrix factorization (SVD) to predict user preferences.

---

## 🧠 Problem Statement
Users often struggle to find movies they will like. This project solves that by predicting movie ratings and recommending the most relevant movies.

---

## ⚙️ How it Works
1. Load MovieLens dataset (user ratings + movies)
2. Train a machine learning model (SVD - Matrix Factorization)
3. Predict ratings for unseen movies
4. Recommend top movies for each user

---

## 📊 Model Used
- Algorithm: Singular Value Decomposition (SVD)
- Type: Collaborative Filtering
- Evaluation Metric: RMSE (Root Mean Squared Error)

---

## 📁 Dataset
- MovieLens dataset
- Contains:
  - userId
  - movieId
  - ratings
  - movie titles

---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Surprise Library
- Streamlit

---

## 🚀 Features
- Personalized movie recommendations
- Machine learning-based predictions
- Interactive web app using Streamlit

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python svd_model.py
streamlit run app.py
