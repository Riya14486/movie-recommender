import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# USER-ITEM MATRIX
user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print(user_item.head())

from sklearn.metrics.pairwise import cosine_similarity

# similarity between users
user_similarity = cosine_similarity(user_item)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item.index,
    columns=user_item.index
)

print(user_similarity_df.head())

def recommend(user_id):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6].index

    similar_ratings = user_item.loc[similar_users]

    avg_ratings = similar_ratings.mean().sort_values(ascending=False)

    return avg_ratings.head(10)

print("\n Recommendations for User 1:\n")
print(recommend(1))