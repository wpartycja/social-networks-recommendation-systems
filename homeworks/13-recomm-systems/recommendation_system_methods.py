import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle


def save_model_to_pickle(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


def load_model_from_pickle(filename):
    with open(filename, "rb") as file:
        loaded_model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return loaded_model


def save_dataframe_to_pickle(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")


def load_dataframe_from_pickle(filename):
    df = pd.read_pickle(filename)
    print(f"DataFrame loaded from {filename}")
    return df


def top_users_by_movie_count(ratings_df, n=10):
    user_movie_counts = ratings_df.groupby("user_id").size()
    top_users = user_movie_counts.sort_values(ascending=False).head(n)
    return top_users


def recommend_movies(ratings_df, movies_df, user_id, model, num_recommendations=5):
    # Get all movie IDs
    all_movie_ids = ratings_df["movie_id"].unique()

    # Find movies the user has not rated
    rated_movies = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].unique()
    unrated_movies = [movie for movie in all_movie_ids if movie not in rated_movies]

    # Predict ratings for unrated movies
    predictions = [(movie, model.predict(user_id, movie).est) for movie in unrated_movies]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_recommendations = predictions[:num_recommendations]

    # Merge with movie titles
    recommendations_df = pd.DataFrame(top_recommendations, columns=["movie_id", "estimated_rating"])
    return recommendations_df.merge(movies_df, on="movie_id")[["movie_id", "title", "estimated_rating"]]


def seen_movies(user_id, ratings_df, movies_df):
    seen = ratings_df[ratings_df["user_id"] == user_id].merge(movies_df, on="movie_id")
    return seen[["movie_id", "title", "rating"]]