from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO

app = Flask(__name__)

# -------- Google Drive file IDs --------
drive_links = {
    "movies": "1IBO0cq5ScwUDQo6teSyS6QauP4Nde1wh",
    "svd_model": "1ke6W6XrJ0n8ZEkDW5Jjlg1FB7iULGgn6",
    "user_movie_matrix": "11kf9xHzq6JY8xdBxF8liTCrcCEDfhU2V"
}

# -------- Local paths --------
os.makedirs("model", exist_ok=True)
local_paths = {
    "movies": os.path.join("model", "movies.dat"),
    "svd_model": os.path.join("model", "svd_model.pkl"),
    "user_movie_matrix": os.path.join("model", "user_movie_matrix.pkl")
}

# -------- Helper to download from Google Drive --------
def download_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {destination}")
    else:
        print(f"{destination} already exists, skipping download.")

# -------- Download files if not present --------
for key in drive_links:
    download_from_gdrive(drive_links[key], local_paths[key])

# -------- Load model and data --------
with open(local_paths["svd_model"], "rb") as f:
    svd = pickle.load(f)

with open(local_paths["user_movie_matrix"], "rb") as f:
    user_movie_matrix_filled = pickle.load(f)

movies = pd.read_csv(
    local_paths["movies"],
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movieId", "title", "genres"]
)
movies["movieId"] = movies["movieId"].astype(int)

# -------- Precompute predicted ratings --------
matrix_reduced = svd.transform(user_movie_matrix_filled)
predicted_ratings_matrix = np.dot(matrix_reduced, svd.components_)
predicted_ratings = pd.DataFrame(
    predicted_ratings_matrix,
    index=user_movie_matrix_filled.index,
    columns=user_movie_matrix_filled.columns
)

# -------- Recommendation function --------
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in predicted_ratings.index:
        return []  # User not found

    user_ratings = predicted_ratings.loc[user_id]

    # Remove movies already rated
    already_rated = user_movie_matrix_filled.loc[user_id]
    already_rated = already_rated[already_rated > 0].index
    recommendations = user_ratings.drop(already_rated, errors="ignore")

    # Top N
    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)

    # Attach movie titles
    recommended_movies = movies[movies["movieId"].isin(top_movies.index)]
    return recommended_movies[["movieId", "title"]].to_dict(orient="records")

# -------- API Endpoint --------
@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    user_id = request.args.get("userId", type=int)
    n = request.args.get("num", default=5, type=int)

    if user_id is None:
        return jsonify({"error": "Missing userId parameter"}), 400

    recommendations = recommend_movies(user_id, n)
    return jsonify({"userId": user_id, "recommendations": recommendations})

# -------- Run the app for testing --------
if __name__ == "__main__":
    test_user = 1
    recs = recommend_movies(test_user, 5)
    print(f"Top 5 recommendations for user {test_user}:")
    print(recs)
    app.run(debug=True)