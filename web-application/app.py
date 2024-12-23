from flask import Flask, request, render_template, redirect, url_for
import lightgbm as lgb
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained LightGBM model
try:
    model = lgb.Booster(model_file="model/lightgbm_final_model.txt")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'lightgbm_final_model.txt' is in the 'model' directory.")
    exit(1)

# Load a subset of the dataset for predictions
data_path = "data/train.csv"
dataset_sample = pd.read_csv(data_path, dtype={
    'source_system_tab': 'category',
    'source_screen_name': 'category',
    'source_type': 'category',
    'genre_ids': 'category',
    'language': 'category',
    'city': 'category',
    'registered_via': 'category',
    'membership_days': 'category',
    'song_year': 'float',
    'play_count': 'int',
    'play_count_artist': 'int',
    'play_count_msno': 'int'
}).sample(1000, random_state=42)  # Sample 1000 rows for efficiency

import random

# Set up Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="f6b12747988644599b9fcff30686adb2",
    client_secret="986c8927d1894bc0a52ecd4a1835858d"
))

# Predefined list of popular track IDs
popular_tracks = [
    "4uLU6hMCjMI75M1A2tKUQC", "2Foc5Q5nqNiosCNqttzHof", "3n3Ppam7vgaVa1iaRUc9Lp","6habFhsOp2NvshLv26DqMb", "7qiZfU4dY1lWllzX7mPBI3", "0VjIjW4GlUZAMYd2vXMi3b",
    "7ouMYWpwJ422jRcDASZB7P", "3DarAbFujv6eYNliUTyqtz", "4hPpVbbakQNv8YTHYaOJP4","2XU0oxnq2qxCpomAAuJY8K", "0rKtyWc8bvkriBthvHKY8d", "2Rk4JlNc2TPmZe2af99d45","6vBdBCoOhKHiYDDOcorfNo", "1M4qEo4HE3PRaCOM7EXNJq", "0e7ipj03S05BNilyu5bRzt"
]

# Initialize current track index
current_track_index = 0

@app.route("/")
def home():
    """Render the home page with the form."""
    return render_template("index.html")

@app.route("/recommend", methods=["POST", "GET"])
def recommend():
    """Provide a song recommendation."""
    global current_track_index  # Ensure we modify the global index

    # Select the current track from the predefined list
    track_id = popular_tracks[current_track_index]

    # Increment the index for the next recommendation
    current_track_index = (current_track_index + 1) % len(popular_tracks)

    # Fetch track details from Spotify
    try:
        track = sp.track(track_id)
        track_name = track['name']
        artist_name = track['artists'][0]['name']
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify API error: {e}")
        track_name = "Unknown"
        artist_name = "Unknown"
        track_id = None

    return render_template(
        "result.html",
        song_name=track_name,
        artist_name=artist_name,
        track_id=track_id
    )

@app.route("/next", methods=["POST"])
def next_recommendation():
    """Redirect to /recommend for the next song."""
    return redirect(url_for("recommend"))

if __name__ == "__main__":
    app.run(debug=True)
