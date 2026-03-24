from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import requests
import zipfile

app = Flask(__name__)

# -------- GitHub Releases URL --------
ZIP_URL = "https://github.com/user-attachments/files/26213390/models.zip.zip"  # your release URL
MODEL_DIR = "model"
ZIP_PATH = os.path.join(MODEL_DIR, "models.zip")

# -------- Ensure model folder exists --------
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- Download & extract ZIP --------
def download_and_extract_zip(url, zip_path=ZIP_PATH, extract_to=MODEL_DIR):
    if not os.path.exists(zip_path):
        print("Downloading models.zip...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Downloaded models.zip")

    # Extract files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print("Extracted model files")

# -------- Download & extract before loading --------
download_and_extract_zip(ZIP_URL)

# -------- Local paths --------
local_paths = {
    "movies": os.path.join(MODEL_DIR, "movies.dat"),
    "svd_model": os.path.join(MODEL_DIR, "svd_model.pkl"),
    "user_movie_matrix": os.path.join(MODEL_DIR, "user_movie_matrix.pkl")
}

# -------- Load model & data --------
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
    return jsonify({"userId": user_id, "recommendations": recommendations})

@app.route("/")
def home():
    return "Movie Recommender API is running!"

if __name__ == "__main__":
    app.run(debug=True)