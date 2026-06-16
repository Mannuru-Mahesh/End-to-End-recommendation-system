"""
hybrid_recommender.py
----------------------
Hybrid Recommendation Engine.

HOW IT WORKS:
  Combines Collaborative Filtering (CF) and Content-Based Filtering (CBF)
  using a WEIGHTED SWITCHING HYBRID strategy:

  ┌─────────────────────────────────────────────────────────────┐
  │  NEW USER (< 5 ratings)  → 100% Content-Based              │
  │  ACTIVE USER (≥ 5 ratings, ≤ 20)  → 30% CF + 70% CBF      │
  │  POWER USER  (> 20 ratings)  → 70% CF + 30% CBF            │
  └─────────────────────────────────────────────────────────────┘

  For each candidate movie:
    hybrid_score = α * cf_score + (1-α) * cbf_score

  where α (alpha) is determined by how many ratings the user has.

WHY HYBRID?
  - CF alone fails for cold-start users → CBF fills the gap
  - CBF alone creates a "filter bubble" → CF adds serendipity
  - Together they handle the full spectrum of user activity levels

FALLBACK:
  - If neither model can generate recs (brand new user, no history)
    → return globally popular/highly-rated movies
"""

import numpy as np
import pandas as pd
import joblib


class HybridRecommender:

    def __init__(self, cf_model, cbf_model, ratings_df, movies_df):
        """
        Parameters
        ----------
        cf_model   : fitted CollaborativeFilter instance
        cbf_model  : fitted ContentBasedFilter instance
        ratings_df : the full ratings DataFrame
        movies_df  : the full movies DataFrame
        """
        self.cf    = cf_model
        self.cbf   = cbf_model
        self.ratings = ratings_df
        self.movies  = movies_df

        # Pre-compute popularity fallback:
        # global ranking = weighted combination of avg rating × log(count)
        movie_stats = (
            ratings_df.groupby('movie_id')['rating']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'avg_rating', 'count': 'n_ratings'})
        )
        movie_stats['popularity_score'] = (
            movie_stats['avg_rating'] *
            np.log1p(movie_stats['n_ratings'])
        )
        self.popularity_ranking = (
            movie_stats.reset_index()
            .sort_values('popularity_score', ascending=False)
        )

    # ----------------------------------------------------------
    def _get_alpha(self, user_id):
        """
        Determine the CF weight (alpha) based on user activity level.
        More ratings → trust CF more (it has more signal to work with).
        """
        n_ratings = len(self.ratings[self.ratings['user_id'] == user_id])
        if n_ratings < 5:
            return 0.0   # pure CBF — not enough data for CF
        elif n_ratings <= 20:
            return 0.3   # mostly CBF + some CF
        else:
            return 0.7   # mostly CF + some CBF

    # ----------------------------------------------------------
    def recommend(self, user_id, n=10):
        """
        Generate hybrid recommendations for a user.

        Returns a DataFrame with columns:
          [movie_id, title, genres, year, hybrid_score, cf_score,
           cbf_score, alpha, source]
        """
        alpha = self._get_alpha(user_id)
        n_ratings = len(self.ratings[self.ratings['user_id'] == user_id])

        # Seen movies (exclude from recommendations)
        seen = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'])
        all_movie_ids = set(self.movies['movie_id'])
        candidate_ids = all_movie_ids - seen

        # ── Collaborative Filtering scores ──────────────────────
        cf_scores = {}
        if alpha > 0 and user_id in self.cf.user_index:
            cf_recs = self.cf.recommend(user_id, n=len(candidate_ids),
                                         exclude_seen=True)
            cf_scores = dict(zip(cf_recs['movie_id'], cf_recs['predicted_rating']))
            # Normalize CF scores to 0-1
            if cf_scores:
                min_cf, max_cf = min(cf_scores.values()), max(cf_scores.values())
                rng = max_cf - min_cf if max_cf > min_cf else 1
                cf_scores = {k: (v - min_cf) / rng for k, v in cf_scores.items()}

        # ── Content-Based Filtering scores ──────────────────────
        cbf_scores = {}
        if (1 - alpha) > 0:
            cbf_recs = self.cbf.recommend(user_id, n=len(candidate_ids),
                                           exclude_seen=True)
            if not cbf_recs.empty:
                cbf_scores = dict(zip(cbf_recs['movie_id'],
                                       cbf_recs['similarity_score']))
                # Already 0-1 (cosine similarity)

        # ── Combine scores ───────────────────────────────────────
        rows = []
        for mid in candidate_ids:
            cf_s  = cf_scores.get(mid, 0.0)
            cbf_s = cbf_scores.get(mid, 0.0)
            hybrid_s = alpha * cf_s + (1 - alpha) * cbf_s
            rows.append({
                'movie_id':    mid,
                'cf_score':    round(cf_s, 4),
                'cbf_score':   round(cbf_s, 4),
                'hybrid_score': round(hybrid_s, 4),
                'alpha':       alpha,
            })

        if not rows:
            return self._popularity_fallback(n, seen)

        results = pd.DataFrame(rows).sort_values('hybrid_score', ascending=False).head(n)

        # Enrich with movie metadata
        results = results.merge(
            self.movies[['movie_id', 'title', 'genres', 'year']],
            on='movie_id', how='left'
        )

        # Tag the dominant source
        results['source'] = results['alpha'].apply(
            lambda a: 'Content-Based' if a == 0
                      else ('Hybrid (CF-heavy)' if a > 0.5 else 'Hybrid (CBF-heavy)')
        )

        return results[['movie_id', 'title', 'genres', 'year',
                         'hybrid_score', 'cf_score', 'cbf_score',
                         'alpha', 'source']].reset_index(drop=True)

    # ----------------------------------------------------------
    def _popularity_fallback(self, n, seen):
        """Return globally popular movies not yet seen by the user."""
        fallback = (
            self.popularity_ranking[
                ~self.popularity_ranking['movie_id'].isin(seen)
            ]
            .head(n)
            .merge(self.movies[['movie_id', 'title', 'genres', 'year']],
                   on='movie_id', how='left')
        )
        fallback['hybrid_score'] = fallback['popularity_score'] / fallback['popularity_score'].max()
        fallback['cf_score']  = 0.0
        fallback['cbf_score'] = 0.0
        fallback['alpha']     = 0.0
        fallback['source']    = 'Popularity Fallback'
        return fallback[['movie_id', 'title', 'genres', 'year',
                          'hybrid_score', 'cf_score', 'cbf_score',
                          'alpha', 'source']].reset_index(drop=True)

    # ----------------------------------------------------------
    def explain(self, user_id, movie_id):
        """
        Generate a human-readable explanation for why a movie was recommended.
        """
        n_ratings = len(self.ratings[self.ratings['user_id'] == user_id])
        alpha = self._get_alpha(user_id)
        movie = self.movies[self.movies['movie_id'] == movie_id].iloc[0]

        # Get user's top genres from their liked movies
        liked = self.ratings[
            (self.ratings['user_id'] == user_id) &
            (self.ratings['rating'] >= 3.5)
        ]
        liked_movies = liked.merge(self.movies, on='movie_id')
        genre_counts = {}
        for genres_str in liked_movies['genres']:
            for g in genres_str.split('|'):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        top_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "unknown"

        lines = [
            f"📽️  Why we recommended '{movie['title']}':",
            f"   Genre: {movie['genres']} | Year: {movie['year']}",
            f"   Your activity level: {n_ratings} ratings → "
            f"{'CF-heavy' if alpha > 0.5 else 'CBF-heavy' if alpha > 0 else 'pure CBF'} hybrid (α={alpha})",
        ]

        if alpha > 0:
            lines.append(f"   • CF signal: users with similar taste to you liked this movie")
        if (1 - alpha) > 0:
            lines.append(f"   • CBF signal: matches your taste for '{top_genre}' movies")

        return '\n'.join(lines)

    # ----------------------------------------------------------
    def save(self, path='models/hybrid_model.pkl'):
        joblib.dump(self, path)
        print(f"  Saved Hybrid model → {path}")

    @staticmethod
    def load(path='models/hybrid_model.pkl'):
        return joblib.load(path)
