"""
content_based_filtering.py
----------------------------
Content-Based Filtering (CBF) Recommender.

HOW IT WORKS:
  1. Build a feature vector for each movie from:
       - Genre one-hot encoding (which genres does it have?)
       - Year bucket (decade: 90s, 00s, 10s, 20s)
       - Quality score (high/medium/low)
  2. For a given user, build a TASTE PROFILE by averaging the
     feature vectors of movies they rated highly (≥ 3.5 stars)
  3. Compute cosine similarity between the user's taste profile
     and every movie they haven't seen yet
  4. Recommend the most similar movies

STRENGTHS:
  - Works even for new users with just a FEW ratings (solves cold-start)
  - Explainable: "recommended because you liked Action movies from the 2010s"
  - No need for other users' data

WEAKNESSES:
  - Can't discover movies outside the user's known taste
    ("filter bubble" — if you only ever rated Sci-Fi, you'll only
    get Sci-Fi recommendations)
  - Needs rich item metadata (genres, descriptions, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class ContentBasedFilter:

    def __init__(self, min_rating_threshold=3.5):
        """
        Parameters
        ----------
        min_rating_threshold : float
            Only use movies rated AT OR ABOVE this threshold to build
            the user's taste profile. We want the profile to represent
            what the user LOVES, not what they rated poorly.
        """
        self.threshold = min_rating_threshold
        self.mlb       = MultiLabelBinarizer()
        self.scaler    = MinMaxScaler()
        self.movie_features = None   # (n_movies × n_features) array
        self.movie_ids      = None   # ordered list of movie_ids

    # ----------------------------------------------------------
    def fit(self, movies_df, ratings_df):
        """
        Build the movie feature matrix.

        Parameters
        ----------
        movies_df  : DataFrame with [movie_id, genres, year, quality]
        ratings_df : DataFrame with [user_id, movie_id, rating]
        """
        movies = movies_df.copy().set_index('movie_id')
        self.movie_ids = list(movies.index)

        # --- Genre features (multi-hot encoding) ---
        genre_lists = movies['genres'].str.split('|').tolist()
        genre_matrix = self.mlb.fit_transform(genre_lists)  # (n_movies, n_genres)

        # --- Year bucket features ---
        def year_bucket(y):
            if y < 2000: return 0   # 1990s
            if y < 2010: return 1   # 2000s
            if y < 2020: return 2   # 2010s
            return 3                # 2020s

        year_buckets = np.array([year_bucket(y) for y in movies['year']])
        year_onehot = np.eye(4)[year_buckets]  # (n_movies, 4)

        # --- Quality score (normalised) ---
        quality = movies['quality'].values.reshape(-1, 1)

        # --- Concatenate all features ---
        self.movie_features = np.hstack([genre_matrix, year_onehot, quality])

        self.movies_df  = movies_df
        self.ratings_df = ratings_df

        print(f"  Content features: {self.movie_features.shape[1]} dims "
              f"({genre_matrix.shape[1]} genres + 4 year + 1 quality)")
        print(f"  Genres: {list(self.mlb.classes_)}")
        return self

    # ----------------------------------------------------------
    def _build_user_profile(self, user_id):
        """
        Build a taste profile vector for a user by averaging
        feature vectors of their highly-rated movies.
        Returns None if the user has no qualifying ratings.
        """
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        liked = user_ratings[user_ratings['rating'] >= self.threshold]

        if liked.empty:
            # Fall back: use ALL ratings if none above threshold
            liked = user_ratings

        if liked.empty:
            return None

        # Get indices of liked movies in our feature matrix
        liked_movie_ids = liked['movie_id'].tolist()
        movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        indices = [movie_id_to_idx[mid] for mid in liked_movie_ids
                   if mid in movie_id_to_idx]

        if not indices:
            return None

        # Weight by rating (higher-rated movies influence the profile more)
        weights = liked[liked['movie_id'].isin(self.movie_ids)]['rating'].values
        weighted_features = self.movie_features[indices] * weights.reshape(-1, 1)
        return weighted_features.mean(axis=0, keepdims=True)

    # ----------------------------------------------------------
    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return top-n content-based recommendations for a user.

        Parameters
        ----------
        user_id      : int
        n            : number of recommendations
        exclude_seen : exclude already-rated movies

        Returns
        -------
        DataFrame with [movie_id, similarity_score, explanation]
        """
        profile = self._build_user_profile(user_id)
        if profile is None:
            return pd.DataFrame(columns=['movie_id', 'similarity_score'])

        # Cosine similarity between user profile and all movies
        sims = cosine_similarity(profile, self.movie_features)[0]

        results = pd.DataFrame({
            'movie_id':        self.movie_ids,
            'similarity_score': sims,
        })

        # Exclude seen movies
        if exclude_seen:
            seen = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
            )
            results = results[~results['movie_id'].isin(seen)]

        results = results.sort_values('similarity_score', ascending=False).head(n)

        # Add simple explanation: top matching genre
        movie_lookup = self.movies_df.set_index('movie_id')
        profile_genre_scores = dict(zip(
            self.mlb.classes_,
            profile[0, :len(self.mlb.classes_)]
        ))
        top_genre = max(profile_genre_scores, key=profile_genre_scores.get)
        results['explanation'] = f"Because you like {top_genre} movies"

        return results.reset_index(drop=True)

    # ----------------------------------------------------------
    def get_similar_movies(self, movie_id, n=10):
        """
        Find movies most similar to a given movie (item-to-item similarity).
        Useful for "Users who watched X also liked..." sections.
        """
        movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        if movie_id not in movie_id_to_idx:
            return pd.DataFrame()

        idx = movie_id_to_idx[movie_id]
        movie_vec = self.movie_features[idx:idx+1]
        sims = cosine_similarity(movie_vec, self.movie_features)[0]

        results = pd.DataFrame({
            'movie_id':        self.movie_ids,
            'similarity_score': sims,
        })
        # Exclude the movie itself
        results = results[results['movie_id'] != movie_id]
        return results.sort_values('similarity_score', ascending=False).head(n).reset_index(drop=True)

    # ----------------------------------------------------------
    def save(self, path='models/cbf_model.pkl'):
        joblib.dump(self, path)
        print(f"  Saved CBF model → {path}")

    @staticmethod
    def load(path='models/cbf_model.pkl'):
        return joblib.load(path)
