from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import requests

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

# -------- Download helper --------
def download_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination}...")

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise Exception(f"Failed to download file: {file_id}")

        with open(destination, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded {destination}")
    else:
        print(f"{destination} already exists")

# -------- Download files --------
for key in drive_links:
    download_from_gdrive(drive_links[key], local_paths[key])

print("Files in model folder:", os.listdir("model"))

# -------- Load model --------
try:
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

    print("All files loaded successfully!")

except Exception as e:
    print("Error loading files:", str(e))
    raise e

# -------- Recommendation function (FAST + SAFE) --------
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix_filled.index:
        return []

    # Get user vector
    user_vector = user_movie_matrix_filled.loc[user_id].values.reshape(1, -1)

    # Transform using SVD
    user_reduced = svd.transform(user_vector)

    # Reconstruct predicted ratings
    user_pred = np.dot(user_reduced, svd.components_).flatten()

    user_ratings = pd.Series(user_pred, index=user_movie_matrix_filled.columns)

    # Remove already rated movies
    already_rated = user_movie_matrix_filled.loc[user_id]
    already_rated = already_rated[already_rated > 0].index

    recommendations = user_ratings.drop(already_rated, errors="ignore")

    # Top N
    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)

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

    return jsonify({
        "userId": user_id,
        "recommendations": recommendations
    })

# -------- Health check --------
@app.route("/")
def home():
    return "Movie Recommender API is running!"

# -------- Run locally --------
if __name__ == "__main__":
    app.run(debug=True)