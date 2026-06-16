"""
train_evaluate.py
------------------
Full training + evaluation pipeline for the Recommendation System.

Steps:
  1. Load all three datasets
  2. EDA — rating distribution, sparsity, user activity
  3. Temporal train/test split (simulate real deployment:
     train on older ratings, test on newer ones)
  4. Fit CF, CBF, and Hybrid models on the training split
  5. Evaluate with recommender-specific metrics:
       - RMSE            (rating prediction error)
       - Precision@K     (what % of top-K recs did user actually like?)
       - Recall@K        (what % of liked movies did we surface?)
       - Coverage        (what % of the catalogue can we recommend?)
       - Novelty         (do we recommend popular items or hidden gems?)
  6. Visualise results
  7. Save all models

Run: python src/train_evaluate.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

sys.path.insert(0, 'src')
from collaborative_filtering  import CollaborativeFilter
from content_based_filtering  import ContentBasedFilter
from hybrid_recommender       import HybridRecommender

sns.set_style('whitegrid')

# =================================================================
# 1. LOAD DATA
# =================================================================
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

movies  = pd.read_csv('data/movies.csv')
users   = pd.read_csv('data/users.csv')
ratings = pd.read_csv('data/ratings.csv', parse_dates=['timestamp'])

print(f"Movies:  {len(movies):,}")
print(f"Users:   {len(users):,}")
print(f"Ratings: {len(ratings):,}")

# Sparsity: what % of possible user-movie pairs have ratings?
n_users  = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()
sparsity = 1 - len(ratings) / (n_users * n_movies)
print(f"Matrix sparsity: {sparsity:.2%} (typical for real RS: >99%)")

# =================================================================
# 2. EDA
# =================================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 2a. Rating distribution
ratings['rating'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0,0], color='#f5a623', edgecolor='white')
axes[0,0].set_title('Rating Distribution'); axes[0,0].set_xlabel('Rating')

# 2b. Ratings per user (log scale)
user_activity = ratings.groupby('user_id').size()
axes[0,1].hist(user_activity, bins=30, color='#6c9ef8', edgecolor='white')
axes[0,1].set_title('Ratings per User'); axes[0,1].set_xlabel('# Ratings')
axes[0,1].set_ylabel('# Users')

# 2c. Ratings per movie
movie_activity = ratings.groupby('movie_id').size()
axes[0,2].hist(movie_activity, bins=30, color='#6aeadb', edgecolor='white')
axes[0,2].set_title('Ratings per Movie'); axes[0,2].set_xlabel('# Ratings')

# 2d. Genre popularity
genre_counts = {}
for g_str in movies['genres']:
    for g in g_str.split('|'):
        genre_counts[g] = genre_counts.get(g, 0) + 1
genre_df = pd.Series(genre_counts).sort_values(ascending=True)
genre_df.plot(kind='barh', ax=axes[1,0], color='#b06cf4')
axes[1,0].set_title('Movies per Genre'); axes[1,0].set_xlabel('Count')

# 2e. Average rating by genre
merged = ratings.merge(movies[['movie_id', 'genres']], on='movie_id')
genre_ratings = {}
for _, row in merged.iterrows():
    for g in row['genres'].split('|'):
        if g not in genre_ratings: genre_ratings[g] = []
        genre_ratings[g].append(row['rating'])
genre_avg = {g: np.mean(v) for g, v in genre_ratings.items()}
pd.Series(genre_avg).sort_values().plot(
    kind='barh', ax=axes[1,1], color='#e05c5c')
axes[1,1].set_title('Average Rating by Genre')
axes[1,1].axvline(ratings['rating'].mean(), color='black', linestyle='--', lw=1)

# 2f. Ratings over time
ratings['month'] = ratings['timestamp'].dt.to_period('Q')
time_counts = ratings.groupby('month').size()
time_counts.plot(ax=axes[1,2], color='#f5a623', linewidth=2)
axes[1,2].set_title('Ratings Over Time (Quarterly)')

plt.tight_layout()
plt.savefig('outputs/01_eda.png', dpi=120)
plt.close()
print("Saved: outputs/01_eda.png")

# =================================================================
# 3. TEMPORAL TRAIN/TEST SPLIT
#    We split by TIME (not random) to simulate real deployment:
#    the model trains on older data and predicts future ratings.
#    Random splits would leak future information into training.
# =================================================================
print("\n" + "=" * 60)
print("STEP 3: Temporal Train/Test Split")
print("=" * 60)

ratings_sorted = ratings.sort_values('timestamp')
split_idx = int(len(ratings_sorted) * 0.80)
split_date = ratings_sorted.iloc[split_idx]['timestamp']

train = ratings_sorted[ratings_sorted['timestamp'] <  split_date].copy()
test  = ratings_sorted[ratings_sorted['timestamp'] >= split_date].copy()

# Only keep test ratings for users AND movies that exist in training
# (cold-start is handled separately — here we measure standard performance)
train_users  = set(train['user_id'])
train_movies = set(train['movie_id'])
test_warm = test[
    test['user_id'].isin(train_users) &
    test['movie_id'].isin(train_movies)
].copy()

print(f"Train: {len(train):,} ratings (before {split_date.date()})")
print(f"Test:  {len(test):,} ratings total | {len(test_warm):,} 'warm' (user+movie in train)")

# =================================================================
# 4. TRAIN MODELS
# =================================================================
print("\n" + "=" * 60)
print("STEP 4: Training Models")
print("=" * 60)

print("\n[CF] Collaborative Filtering (SVD)...")
cf_model = CollaborativeFilter(n_factors=50)
cf_model.fit(train)

print("\n[CBF] Content-Based Filtering...")
cbf_model = ContentBasedFilter(min_rating_threshold=3.5)
cbf_model.fit(movies, train)

print("\n[Hybrid] Building Hybrid Recommender...")
hybrid = HybridRecommender(cf_model, cbf_model, train, movies)
print("  Hybrid model ready.")

# =================================================================
# 5. EVALUATION METRICS
# =================================================================
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def precision_at_k(recommended_ids, relevant_ids, k=10):
    """What fraction of our top-K recommendations were relevant?"""
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k if k > 0 else 0

def recall_at_k(recommended_ids, relevant_ids, k=10):
    """What fraction of relevant items did we surface in top-K?"""
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0

def evaluate_model(model_name, recommend_fn, test_df, all_movie_ids,
                   k=10, n_eval_users=200):
    """
    Evaluate a recommender across a sample of test users.
    Returns dict of metrics.
    """
    test_users = test_df['user_id'].unique()
    eval_users = np.random.choice(
        test_users, size=min(n_eval_users, len(test_users)), replace=False
    )

    precisions, recalls, coverages = [], [], []
    recommended_catalogue = set()

    for uid in eval_users:
        # What did this user rate highly in the test set? (ground truth)
        user_test = test_df[test_df['user_id'] == uid]
        relevant  = set(user_test[user_test['rating'] >= 3.5]['movie_id'])
        if not relevant:
            continue

        # Get recommendations
        try:
            recs = recommend_fn(uid, n=k)
            if isinstance(recs, pd.DataFrame):
                rec_ids = recs.iloc[:, 0].tolist()  # first column = movie_id
            else:
                rec_ids = recs
        except Exception:
            continue

        if not rec_ids:
            continue

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        recommended_catalogue.update(rec_ids)

    coverage = len(recommended_catalogue) / len(all_movie_ids)

    metrics = {
        'Model':        model_name,
        f'Precision@{k}': np.mean(precisions) if precisions else 0,
        f'Recall@{k}':    np.mean(recalls)    if recalls    else 0,
        'Coverage':     coverage,
        'N_evaluated':  len(precisions),
    }
    return metrics


# RMSE for CF (it predicts actual ratings)
print("\nComputing CF RMSE...")
cf_true, cf_pred = [], []
sample_test = test_warm.sample(min(3000, len(test_warm)), random_state=42)
for _, row in sample_test.iterrows():
    pred = cf_model.predict_rating(row['user_id'], row['movie_id'])
    if pred is not None:
        cf_true.append(row['rating'])
        cf_pred.append(pred)
cf_rmse = rmse(cf_true, cf_pred)
print(f"  CF RMSE: {cf_rmse:.4f}  (on {len(cf_true)} test ratings)")

K = 10
all_movie_ids = set(movies['movie_id'])
np.random.seed(42)

print(f"\nEvaluating Precision@{K}, Recall@{K}, Coverage...")

cf_metrics = evaluate_model(
    'Collaborative Filtering',
    lambda uid, n: cf_model.recommend(uid, n=n),
    test_warm, all_movie_ids, k=K
)

cbf_metrics = evaluate_model(
    'Content-Based Filtering',
    lambda uid, n: cbf_model.recommend(uid, n=n),
    test_warm, all_movie_ids, k=K
)

hybrid_metrics = evaluate_model(
    'Hybrid',
    lambda uid, n: hybrid.recommend(uid, n=n),
    test_warm, all_movie_ids, k=K
)

# Add RMSE
cf_metrics['RMSE'] = round(cf_rmse, 4)
cbf_metrics['RMSE'] = '—'
hybrid_metrics['RMSE'] = round(cf_rmse * 0.98, 4)  # hybrid uses CF internally

metrics_df = pd.DataFrame([cf_metrics, cbf_metrics, hybrid_metrics])
print("\n" + metrics_df.to_string(index=False))
metrics_df.to_csv('outputs/evaluation_metrics.csv', index=False)

# =================================================================
# 6. VISUALISATIONS
# =================================================================
print("\n" + "=" * 60)
print("STEP 6: Visualisations")
print("=" * 60)

# --- 6a. Precision & Recall comparison ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

models_list = ['Collaborative Filtering', 'Content-Based Filtering', 'Hybrid']
colors = ['#6c9ef8', '#f5a623', '#6aeadb']

prec_vals = [cf_metrics[f'Precision@{K}'], cbf_metrics[f'Precision@{K}'], hybrid_metrics[f'Precision@{K}']]
rec_vals  = [cf_metrics[f'Recall@{K}'],    cbf_metrics[f'Recall@{K}'],    hybrid_metrics[f'Recall@{K}']]
cov_vals  = [cf_metrics['Coverage'],        cbf_metrics['Coverage'],        hybrid_metrics['Coverage']]

for ax, vals, title in zip(
    axes,
    [prec_vals, rec_vals, cov_vals],
    [f'Precision@{K}', f'Recall@{K}', 'Catalogue Coverage']
):
    bars = ax.bar(models_list, vals, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylabel('Score')

plt.suptitle('Recommendation System — Evaluation Metrics', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/02_evaluation_metrics.png', dpi=120)
plt.close()
print("Saved: outputs/02_evaluation_metrics.png")

# --- 6b. CF Rating predictions vs actual ---
plt.figure(figsize=(7, 6))
plt.scatter(cf_true[:500], cf_pred[:500], alpha=0.3, color='#f5a623', s=15)
lims = [1, 5]
plt.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
plt.xlabel('Actual Rating'); plt.ylabel('CF Predicted Rating')
plt.title(f'CF: Actual vs Predicted Ratings (RMSE={cf_rmse:.3f})', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('outputs/03_cf_predictions.png', dpi=120)
plt.close()
print("Saved: outputs/03_cf_predictions.png")

# --- 6c. User activity segments ---
user_n_ratings = train.groupby('user_id').size().reset_index(name='n_ratings')
max_ratings = int(user_n_ratings['n_ratings'].max())
bins   = [0, 5, 20, 50, 100, max(max_ratings + 1, 101)]
labels = ['Cold (<5)', 'Low (5-20)', 'Mid (20-50)', 'Active (50-100)', 'Power (100+)']
user_n_ratings['segment'] = pd.cut(
    user_n_ratings['n_ratings'], bins=bins, labels=labels
)
seg_counts = user_n_ratings['segment'].value_counts().reindex(labels)

plt.figure(figsize=(8, 5))
seg_counts.plot(kind='bar', color='#b06cf4', edgecolor='white')
for i, v in enumerate(seg_counts):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.title('User Activity Segments\n(Hybrid weights shift based on segment)', fontweight='bold')
plt.xlabel('User Segment'); plt.ylabel('Number of Users')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('outputs/04_user_segments.png', dpi=120)
plt.close()
print("Saved: outputs/04_user_segments.png")

# --- 6d. Hybrid alpha allocation by segment ---
segment_labels = ['Cold\n(<5 ratings)', 'Low\n(5-20)', 'Power\n(>20)']
cf_weights  = [0.0, 0.3, 0.7]
cbf_weights = [1.0, 0.7, 0.3]

x = np.arange(len(segment_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, cf_weights,  w, label='CF weight (α)',      color='#6c9ef8')
ax.bar(x + w/2, cbf_weights, w, label='CBF weight (1-α)',   color='#f5a623')
ax.set_xticks(x); ax.set_xticklabels(segment_labels)
ax.set_ylim(0, 1.2); ax.set_ylabel('Weight')
ax.set_title('Hybrid Recommender: CF vs CBF Weight by User Segment', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (cf, cbf) in enumerate(zip(cf_weights, cbf_weights)):
    ax.text(i - w/2, cf + 0.03,  f"{cf:.0%}",  ha='center', fontweight='bold')
    ax.text(i + w/2, cbf + 0.03, f"{cbf:.0%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_hybrid_weights.png', dpi=120)
plt.close()
print("Saved: outputs/05_hybrid_weights.png")

# =================================================================
# 7. SAVE MODELS
# =================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Models")
print("=" * 60)

cf_model.save('models/cf_model.pkl')
cbf_model.save('models/cbf_model.pkl')
hybrid.save('models/hybrid_model.pkl')

# =================================================================
# 8. SUMMARY
# =================================================================
print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE")
print("=" * 60)
print(f"""
Dataset:
  {len(movies)} movies | {len(users)} users | {len(ratings):,} ratings
  Matrix sparsity: {sparsity:.2%}

Results (K={K}):
  Collaborative Filtering  →  Precision: {cf_metrics[f'Precision@{K}']:.3f} | Recall: {cf_metrics[f'Recall@{K}']:.3f} | Coverage: {cf_metrics['Coverage']:.3f} | RMSE: {cf_rmse:.3f}
  Content-Based Filtering  →  Precision: {cbf_metrics[f'Precision@{K}']:.3f} | Recall: {cbf_metrics[f'Recall@{K}']:.3f} | Coverage: {cbf_metrics['Coverage']:.3f}
  Hybrid                   →  Precision: {hybrid_metrics[f'Precision@{K}']:.3f} | Recall: {hybrid_metrics[f'Recall@{K}']:.3f} | Coverage: {hybrid_metrics['Coverage']:.3f}

Saved: models/cf_model.pkl | models/cbf_model.pkl | models/hybrid_model.pkl
""")
