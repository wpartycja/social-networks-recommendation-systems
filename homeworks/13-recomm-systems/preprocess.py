import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from matplotlib import pyplot as plt

combined_data_path = "data/combined_data_1.txt"
movies_data_path = "data/movie_titles.csv"

def load_combined_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        movie_id = None
        for line in file:
            line = line.strip()
            if line.endswith(':'):
                # Movie ID line
                movie_id = int(line[:-1])
            else:
                # User ID, rating, date
                user_id, rating, date = line.split(',')
                data.append([movie_id, int(user_id), int(rating)])
    return pd.DataFrame(data, columns=["movie_id", "user_id", "rating"])

def preprocess_movie_titles(file_path, output_path):
    with open(file_path, 'r', encoding="ISO-8859-1") as infile, open(output_path, 'w', encoding="ISO-8859-1") as outfile:
        for line in infile:
            parts = line.strip().split(',', 2)  # Split only the first two commas
            if len(parts) == 3:
                movie_id, year, title = parts
                # Add quotes around the title if it contains commas
                if ',' in title:
                    title = f'"{title}"'
                outfile.write(f"{movie_id},{title}\n")


def load_movie_titles(file_path):
    return pd.read_csv(
        file_path, 
        header=None, 
        encoding = "ISO-8859-1",
        names=["movie_id", "title"], 
        quoting=csv.QUOTE_MINIMAL
    )
    

def filter_dataframe(df, item_id, functions):
    df_summary = df.groupby(item_id)["rating"].agg(functions)
    df_summary.index = df_summary.index.map(int)
    benchmark = round(df_summary["count"].quantile(0.7), 0)
    drop_list = df_summary[df_summary["count"] < benchmark].index
    return drop_list, benchmark


def perform_eda(ratings_df, movies_df):
    # Plot histogram of ratings
    plt.figure(figsize=(10, 6))
    plt.hist(ratings_df['rating'], bins=5, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot histogram of movie review counts
    movie_review_counts = ratings_df.groupby('movie_id').size()
    plt.figure(figsize=(10, 6))
    plt.hist(movie_review_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Movie Review Counts')
    plt.xlabel('Number of Reviews per Movie')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot histogram of user review counts
    user_review_counts = ratings_df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    plt.hist(user_review_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of User Review Counts')
    plt.xlabel('Number of Reviews per User')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()