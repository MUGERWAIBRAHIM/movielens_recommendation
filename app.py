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
    "movies": "model/movies.dat",
    "svd_model": "model/svd_model.pkl",
    "user_movie_matrix": "model/user_movie_matrix.pkl"
}

# -------- Google Drive downloader (FIXED) --------
def download_from_gdrive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    print(f"Downloading {destination}...")

    response = session.get(URL, params={'id': file_id}, stream=True)

    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {destination}")

# -------- Validate pickle --------
def is_valid_pickle(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        start = f.read(2)
        return start != b'<!'  # HTML check

# -------- Ensure valid files --------
def ensure_file(file_key):
    path = local_paths[file_key]

    # Download if missing or corrupted
    if not is_valid_pickle(path):
        print(f"{path} invalid or missing. Re-downloading...")
        download_from_gdrive(drive_links[file_key], path)

        # Check again
        if not is_valid_pickle(path):
            raise Exception(f"{file_key} STILL corrupted after download. Fix Google Drive sharing.")

# -------- Download & validate --------
for key in drive_links:
    ensure_file(key)

print("Files ready:", os.listdir("model"))

# -------- Load data --------
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

# -------- Recommendation function --------
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix_filled.index:
        return []

    user_vector = user_movie_matrix_filled.loc[user_id].values.reshape(1, -1)

    user_reduced = svd.transform(user_vector)
    user_pred = np.dot(user_reduced, svd.components_).flatten()

    user_ratings = pd.Series(user_pred, index=user_movie_matrix_filled.columns)

    already_rated = user_movie_matrix_filled.loc[user_id]
    already_rated = already_rated[already_rated > 0].index

    recommendations = user_ratings.drop(already_rated, errors="ignore")

    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)

    recommended_movies = movies[movies["movieId"].isin(top_movies.index)]

    return recommended_movies[["movieId", "title"]].to_dict(orient="records")

# -------- API --------
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

@app.route("/")
def home():
    return "Movie Recommender API is running!"

if __name__ == "__main__":
    app.run(debug=True)