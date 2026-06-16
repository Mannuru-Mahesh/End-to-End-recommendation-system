"""
collaborative_filtering.py
----------------------------
Collaborative Filtering (CF) Recommender using SVD
(Singular Value Decomposition) Matrix Factorization.

HOW IT WORKS:
  1. Build a User × Movie rating matrix (sparse — most entries missing)
  2. Apply TruncatedSVD to decompose it into:
       U (users × k)  ·  Σ (k × k)  ·  Vt (k × movies)
     where k = number of latent factors (hidden patterns like
     "likes action", "prefers 90s films", etc.)
  3. Reconstruct the full matrix: predicted_ratings = U · Σ · Vt
  4. For a given user, recommend movies with the highest
     predicted rating that the user HASN'T seen yet.

STRENGTHS:
  - Discovers hidden taste patterns without needing genre labels
  - "Users like you also liked..." recommendations
  - Gets better as more ratings come in

WEAKNESSES:
  - Cold-start problem: can't recommend for brand new users
    with few/no ratings (no column in the matrix)
  - Popularity bias: tends to recommend well-rated popular movies
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib


class CollaborativeFilter:

    def __init__(self, n_factors=50, random_state=42):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors (dimensions) to keep after SVD.
            More factors = more expressive but slower + risk of overfitting.
        """
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_matrix  = None   # U · Σ  (users  × k)
        self.item_matrix  = None   # Vt     (k × movies)
        self.user_index   = None   # user_id → row index
        self.movie_index  = None   # movie_id → col index
        self.movie_ids    = None   # col index → movie_id

    # ----------------------------------------------------------
    def fit(self, ratings_df):
        """
        Build the user-item matrix and decompose with SVD.

        Parameters
        ----------
        ratings_df : DataFrame with columns [user_id, movie_id, rating]
        """
        # Mean-center ratings per user (removes user bias:
        # some users always rate 5★, others always rate 2★)
        user_means = ratings_df.groupby('user_id')['rating'].mean()
        ratings_df = ratings_df.copy()
        ratings_df['rating_centered'] = (
            ratings_df['rating'] -
            ratings_df['user_id'].map(user_means)
        )

        # Build index mappings
        unique_users  = sorted(ratings_df['user_id'].unique())
        unique_movies = sorted(ratings_df['movie_id'].unique())
        self.user_index  = {uid: i for i, uid in enumerate(unique_users)}
        self.movie_index = {mid: j for j, mid in enumerate(unique_movies)}
        self.movie_ids   = unique_movies

        # Build sparse user × movie matrix
        rows = ratings_df['user_id'].map(self.user_index)
        cols = ratings_df['movie_id'].map(self.movie_index)
        vals = ratings_df['rating_centered']

        n_users  = len(unique_users)
        n_movies = len(unique_movies)
        R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

        # SVD decomposition: R ≈ U · Σ · Vt
        # user_matrix = U · Σ  (each row is a user's latent "taste vector")
        self.user_matrix = self.svd.fit_transform(R)
        self.item_matrix = self.svd.components_  # shape: (k, n_movies)

        # Store user means for de-centering predictions
        self.user_means = user_means
        self.ratings_df = ratings_df

        print(f"  SVD: {n_users} users × {n_movies} movies → {self.n_factors} latent factors")
        print(f"  Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.3f}")
        return self

    # ----------------------------------------------------------
    def predict_rating(self, user_id, movie_id):
        """
        Predict the rating user_id would give movie_id.
        Returns None if user not in training data.
        """
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return None

        u_idx = self.user_index[user_id]
        m_idx = self.movie_index[movie_id]

        # Dot product of user latent vector and movie latent vector
        pred_centered = self.user_matrix[u_idx] @ self.item_matrix[:, m_idx]

        # Add back user's mean rating (de-center)
        user_mean = self.user_means.get(user_id, 3.0)
        return float(np.clip(pred_centered + user_mean, 1.0, 5.0))

    # ----------------------------------------------------------
    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return top-n movie recommendations for a user.

        Parameters
        ----------
        user_id      : int
        n            : number of recommendations
        exclude_seen : if True, don't recommend movies already rated

        Returns
        -------
        DataFrame with columns [movie_id, predicted_rating]
        sorted descending by predicted_rating.
        """
        if user_id not in self.user_index:
            return pd.DataFrame(columns=['movie_id', 'predicted_rating'])

        u_idx = self.user_index[user_id]
        user_mean = self.user_means.get(user_id, 3.0)

        # Predict ratings for ALL movies at once (vectorized dot product)
        pred_centered = self.user_matrix[u_idx] @ self.item_matrix
        predictions   = np.clip(pred_centered + user_mean, 1.0, 5.0)

        # Build results DataFrame
        results = pd.DataFrame({
            'movie_id':        self.movie_ids,
            'predicted_rating': predictions
        })

        # Remove already-seen movies
        if exclude_seen:
            seen = set(
                self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
            )
            results = results[~results['movie_id'].isin(seen)]

        return results.sort_values('predicted_rating', ascending=False).head(n).reset_index(drop=True)

    # ----------------------------------------------------------
    def save(self, path='models/cf_model.pkl'):
        joblib.dump(self, path)
        print(f"  Saved CF model → {path}")

    @staticmethod
    def load(path='models/cf_model.pkl'):
        return joblib.load(path)
