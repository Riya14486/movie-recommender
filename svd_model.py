import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

ratings = pd.read_csv("data/ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")