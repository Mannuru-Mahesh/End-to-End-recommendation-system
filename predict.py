"""
predict.py
-----------
Loads the trained recommendation models and generates live
recommendations for three user types:
  1. Power user  (many ratings → CF-heavy hybrid)
  2. New user    (few ratings → CBF-heavy)
  3. Cold-start  (only 2 ratings → pure content-based)

Also demonstrates:
  - Item-to-item similarity ("More like this")
  - Explanation generation

Usage:
    python src/predict.py
"""

import sys, os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from collaborative_filtering import CollaborativeFilter
from content_based_filtering import ContentBasedFilter
from hybrid_recommender      import HybridRecommender

# =================================================================
# LOAD ARTEFACTS
# =================================================================
print("Loading models...")
cf     = CollaborativeFilter.load('models/cf_model.pkl')
cbf    = ContentBasedFilter.load('models/cbf_model.pkl')
hybrid = HybridRecommender.load('models/hybrid_model.pkl')

movies  = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
users   = pd.read_csv('data/users.csv')

def user_info(uid):
    u = users[users['user_id'] == uid].iloc[0]
    n = len(ratings[ratings['user_id'] == uid])
    return f"User {uid} | Age: {u['age_group']} | Fav genre: {u['fav_genre']} | {n} ratings"

def show_recs(recs_df, movies_df, title="Recommendations"):
    # If title column not already present, merge metadata
    if 'title' not in recs_df.columns:
        recs_df = recs_df.merge(movies_df[['movie_id','title','genres','year']],
                                on='movie_id', how='left')
    score_col = next((c for c in ['hybrid_score','predicted_rating','similarity_score']
                      if c in recs_df.columns), None)
    print(f"\n  {'#':<3} {'Title':<35} {'Genres':<32} {'Year'}")
    print(f"  {'─'*82}")
    for rank, (_, row) in enumerate(recs_df.head(10).iterrows(), 1):
        t = str(row.get('title',''))[:33]
        g = str(row.get('genres',''))[:30]
        y = str(int(row.get('year', 0))) if row.get('year', 0) else ''
        s = f"{row.get(score_col, 0):.3f}" if score_col else ''
        print(f"  {rank:<3} {t:<35} {g:<32} {y}  [{s}]")


# =================================================================
# FIND REPRESENTATIVE USERS
# =================================================================
user_activity = ratings.groupby('user_id').size().reset_index(name='n')

# Power user: > 60 ratings
power_users = user_activity[user_activity['n'] > 60]['user_id'].values
power_uid   = int(power_users[0]) if len(power_users) > 0 else 1

# New user: 5–20 ratings
new_users   = user_activity[(user_activity['n'] >= 5) & (user_activity['n'] <= 20)]['user_id'].values
new_uid     = int(new_users[0]) if len(new_users) > 0 else 2

# Cold-start: ≤ 3 ratings
cold_users  = user_activity[user_activity['n'] <= 3]['user_id'].values
cold_uid    = int(cold_users[0]) if len(cold_users) > 0 else 3


print("\n" + "=" * 70)
print("RECOMMENDATION SYSTEM — LIVE DEMO")
print("=" * 70)

# =================================================================
# SCENARIO 1: POWER USER — hybrid CF-heavy
# =================================================================
print(f"\n{'─'*70}")
print(f"SCENARIO 1 — POWER USER (CF-heavy hybrid)")
print(f"  {user_info(power_uid)}")

# Show what they've recently watched
recent = (ratings[ratings['user_id'] == power_uid]
          .sort_values('timestamp', ascending=False)
          .head(5)
          .merge(movies[['movie_id','title','genres']], on='movie_id'))
print(f"\n  Recently watched:")
for _, r in recent.iterrows():
    print(f"    ★{r['rating']}  {r['title'][:40]}  ({r['genres']})")

print(f"\n  → Hybrid recommendations (α={hybrid._get_alpha(power_uid)}):")
recs = hybrid.recommend(power_uid, n=10)
show_recs(recs, movies)

print(f"\n  → Explanation for top pick:")
if not recs.empty:
    print("  " + hybrid.explain(power_uid, recs.iloc[0]['movie_id']).replace('\n', '\n  '))

# =================================================================
# SCENARIO 2: NEW USER — CBF-heavy hybrid
# =================================================================
print(f"\n{'─'*70}")
print(f"SCENARIO 2 — NEW USER (CBF-heavy hybrid)")
print(f"  {user_info(new_uid)}")

recent2 = (ratings[ratings['user_id'] == new_uid]
           .merge(movies[['movie_id','title','genres']], on='movie_id'))
print(f"\n  All ratings so far:")
for _, r in recent2.iterrows():
    print(f"    ★{r['rating']}  {r['title'][:40]}  ({r['genres']})")

print(f"\n  → Hybrid recommendations (α={hybrid._get_alpha(new_uid)}):")
recs2 = hybrid.recommend(new_uid, n=10)
show_recs(recs2, movies)

# =================================================================
# SCENARIO 3: COLD-START USER — pure content-based
# =================================================================
print(f"\n{'─'*70}")
print(f"SCENARIO 3 — COLD-START USER (pure content-based)")
print(f"  {user_info(cold_uid)}")

recent3 = (ratings[ratings['user_id'] == cold_uid]
           .merge(movies[['movie_id','title','genres']], on='movie_id'))
print(f"\n  Only ratings:")
for _, r in recent3.iterrows():
    print(f"    ★{r['rating']}  {r['title'][:40]}  ({r['genres']})")

print(f"\n  → Hybrid recommendations (α={hybrid._get_alpha(cold_uid)}) — falls back to CBF:")
recs3 = hybrid.recommend(cold_uid, n=10)
show_recs(recs3, movies)

# =================================================================
# SCENARIO 4: "MORE LIKE THIS" — item-to-item similarity
# =================================================================
print(f"\n{'─'*70}")
print(f"SCENARIO 4 — ITEM-TO-ITEM SIMILARITY  ('More Like This')")

sample_movie = movies.sample(1, random_state=7).iloc[0]
print(f"\n  Seed movie: '{sample_movie['title']}'")
print(f"  Genres: {sample_movie['genres']} | Year: {sample_movie['year']}")
print(f"\n  → Most similar movies:")
similar = cbf.get_similar_movies(sample_movie['movie_id'], n=8)
similar = similar.merge(movies[['movie_id','title','genres','year']], on='movie_id')
for i, row in similar.iterrows():
    print(f"    {i+1}. {row['title'][:40]:<42}  ({row['genres']})  sim={row['similarity_score']:.3f}")

print(f"\n{'─'*70}")
print("✅ Demo complete.")
print("   CF excels for power users | CBF saves cold-start | Hybrid balances both.")
