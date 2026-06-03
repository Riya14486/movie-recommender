import streamlit as st
import pandas as pd
import pickle

# LOAD DATA
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# LOAD MODEL
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def get_movie_name(movie_id):
    return movies[movies["movieId"] == movie_id]["title"].values[0]

def recommend(user_id, n=10):
    all_movies = ratings["movieId"].unique()

    preds = []
    for m in all_movies:
        pred = model.predict(user_id, m).est
        preds.append((m, pred))

    preds.sort(key=lambda x: x[1], reverse=True)

    top = preds[:n]

    return [(get_movie_name(mid), score) for mid, score in top]

# UI
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Movie Recommender", layout="wide")

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def get_movie_name(movie_id):
    return movies[movies["movieId"] == movie_id]["title"].values[0]

def recommend(user_id, n=10):
    all_movies = ratings["movieId"].unique()

    preds = []
    for m in all_movies:
        pred = model.predict(user_id, m).est
        preds.append((m, pred))

    preds.sort(key=lambda x: x[1], reverse=True)

    top = preds[:n]

    return [(get_movie_name(mid), score) for mid, score in top]


st.title("🎬 Movie Recommendation System (SVD)")

st.write("Get personalized movie recommendations using Machine Learning.")

user_id = st.number_input("Enter User ID", min_value=1)

if st.button("Get Recommendations"):
    results = recommend(user_id)

    st.subheader("Top Recommendations")

    for movie, score in results:
        st.markdown(f"### 🎥 {movie}")
        st.write(f"⭐ Predicted Rating: {score:.2f}")
        st.divider()