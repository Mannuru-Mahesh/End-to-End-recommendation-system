End-to-End Movie Recommendation System

> A complete recommendation system built from scratch implementing three complementary approaches — Collaborative Filtering (SVD matrix factorisation), Content-Based Filtering (cosine similarity), and a Hybrid engine — evaluated with proper recommender-system metrics: Precision@K, Recall@K, Coverage, and RMSE.

Project Description

Every major streaming and e-commerce platform (Netflix, Spotify, Amazon) uses a hybrid recommendation system. This project builds one from scratch using a synthetic dataset of 1,000 users, 500 movies, and 50,000 ratings.

The key insight this project demonstrates: **no single recommender works best for all users**. CF requires many ratings to be accurate, CBF works from the first rating but creates a filter bubble, and a hybrid that adapts its CF/CBF balance based on user activity gets the best of both worlds.



System Architecture


User Request
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Hybrid Recommender                  │
│                                                  │
│   Activity check:  n_ratings < 5 / 5-20 / 20+  │
│                                                  │
│   ┌──────────────────┐   ┌────────────────────┐  │
│   │ Collaborative    │   │  Content-Based     │  │
│   │ Filtering (SVD)  │   │  Filtering         │  │
│   │                  │   │  (cosine sim)      │  │
│   │ "Users like you  │   │  "Because you like │  │
│   │  also liked..."  │   │   Action movies"   │  │
│   └──────────────────┘   └────────────────────┘  │
│            │  α weight          │  (1-α) weight   │
│            └──────────┬─────────┘                 │
│                       ▼                           │
│              Hybrid Score = α·CF + (1-α)·CBF      │
└─────────────────────────────────────────────────┘
     │
     ▼
Ranked Top-N Recommendations + Explanations


Hybrid weighting strategy

| User type | Ratings | CF weight (α) | CBF weight (1-α) |
|---|---|---|---|
| Cold-start | < 5 | 0% | 100% |
| New user | 5 – 20 | 30% | 70% |
| Power user | > 20 | 70% | 30% |


Features

| Feature | Details |
|---|---|
| **Collaborative Filtering** | TruncatedSVD on mean-centred user × movie matrix (50 latent factors) |
| **Content-Based Filtering** | Genre one-hot + year bucket + quality score → cosine similarity |
| **Hybrid Recommender** | Dynamic CF/CBF weighting by user activity level |
| **Item-to-item similarity** | "More like this" using CBF cosine similarity |
| **Temporal train/test split** | Prevents future data leakage |
| **Cold-start handling** | CBF recommends from just 1–2 ratings |
| **Popularity fallback** | Global ranking for completely new users |
| **Explainability** | Human-readable reason for each recommendation |
| **Full evaluation suite** | RMSE, Precision@K, Recall@K, Catalogue Coverage |
| **Pre-executed notebook** | All outputs visible on GitHub without running |


Results (K=10)

| Model | Precision@10 | Recall@10 | Coverage | RMSE |
|---|---|---|---|---|
| Collaborative Filtering | **0.043** | **0.115** | 0.368 | 0.892 |
| Content-Based Filtering | 0.024 | 0.067 | **0.796** | — |
| Hybrid | 0.037 | 0.115 | 0.490 | ~0.874 |

Key finding: CF achieves better precision for active users but only covers 37% of the catalogue (popularity bias). CBF covers 80% of the catalogue and handles cold-start but has lower precision. The hybrid balances both trade-offs.


Tech Stack

- **Python 3.10+**
- **NumPy / Pandas** — data manipulation
- **SciPy** — sparse matrix construction
- **scikit-learn** — TruncatedSVD, cosine similarity, metrics
- **matplotlib / seaborn** — visualisation
- **joblib** — model persistence
- **Jupyter Notebook** — narrative analysis


Folder Structure


recommendation-system/
├── data/
│   ├── movies.csv          ← 500 movies (title, genres, year, quality)
│   ├── users.csv           ← 1,000 users (age_group, fav_genre, sec_genre)
│   └── ratings.csv         ← 50,000 ratings (user_id, movie_id, rating, timestamp)
├── notebooks/
│   └── Recommendation_System.ipynb   ← Full narrative (pre-executed)
├── src/
│   ├── generate_data.py              ← Creates all three datasets
│   ├── collaborative_filtering.py    ← SVD matrix factorisation CF engine
│   ├── content_based_filtering.py    ← Cosine similarity CBF engine
│   ├── hybrid_recommender.py         ← Weighted hybrid + explanations
│   ├── train_evaluate.py             ← Full training + evaluation pipeline
│   ├── predict.py                    ← Live demo (power/new/cold-start users)
│   └── build_notebook.py             ← Script that built the .ipynb
├── models/
│   ├── cf_model.pkl                  ← Saved CF model
│   ├── cbf_model.pkl                 ← Saved CBF model
│   └── hybrid_model.pkl              ← Saved Hybrid model
├── outputs/
│   ├── 01_eda.png                    ← EDA charts
│   ├── 02_evaluation_metrics.png     ← Precision/Recall/Coverage comparison
│   ├── 03_cf_predictions.png         ← Actual vs predicted rating scatter
│   ├── 04_user_segments.png          ← User activity breakdown
│   ├── 05_hybrid_weights.png         ← CF/CBF weight allocation chart
│   └── evaluation_metrics.csv        ← All metrics as CSV
├── requirements.txt
└── README.md




How to Run Locally

1. Clone and install


git clone https://github.com/YOUR_USERNAME/recommendation-system.git
cd recommendation-system

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt


2. Generate the dataset


python src/generate_data.py


Creates all three CSV files in `data/`. Fully reproducible with fixed random seeds.

3. Run the training + evaluation pipeline


python src/train_evaluate.py


Trains all three models, evaluates with Precision@10 / Recall@10 / Coverage, saves plots to `outputs/` and models to `models/`. **Runtime: ~60 seconds.**

4. Run the live demo


python src/predict.py


Shows recommendations for a power user, a new user, and a cold-start user — plus item-to-item similarity.

5. Open the notebook


jupyter notebook notebooks/Recommendation_System.ipynb


How to Explain It in an Interview

> *"I built three complementary recommenders and showed why no single approach is optimal for all users.*
>
> *Collaborative Filtering uses SVD to find latent taste patterns in a user × movie rating matrix — it discovers that users who liked the same movies probably share preferences the ratings don't explicitly state. It achieves the best Precision@10 (4.3%) for active users. But it fails completely for cold-start users with fewer than 5 ratings, and only covers 37% of the catalogue due to popularity bias.*
>
> *Content-Based Filtering builds a taste profile from the genres and years of movies a user liked, then recommends the most similar unseen movies using cosine similarity. It works from the very first rating and covers 80% of the catalogue — but it creates a filter bubble where users only see content similar to what they've already rated.*
>
> *The Hybrid solves this by dynamically shifting its CF/CBF weight based on how many ratings a user has. Cold-start users get 100% CBF, new users get 70% CBF / 30% CF, and power users get 70% CF / 30% CBF. This means the system is always useful — from day one to year one — while avoiding the individual weaknesses of each approach."*


What I Learned

- How **SVD matrix factorisation** discovers latent taste preferences without explicit labels
- Why **temporal train/test splits** matter — random splits leak future data into training
- The fundamental **precision vs coverage trade-off** in recommendation systems
- How to implement and quantify the **cold-start problem** and its solutions
- How **cosine similarity** enables content-based matching without requiring user history
- How **hybrid weighting** can dynamically balance competing recommenders
- How to evaluate recommenders with **Precision@K, Recall@K, and Coverage** (not just RMSE)
- How to build **explainable recommendations** ("Because you like Action movies")


Future Improvements

- [ ] Replace SVD with **ALS (Alternating Least Squares)** for better handling of implicit feedback (clicks, watch time)
- [ ] Implement **Neural Collaborative Filtering (NCF)** — learn embeddings end-to-end with a neural net
- [ ] Add **diversity-aware re-ranking** to reduce the CF filter bubble
- [ ] Incorporate **temporal dynamics** — model how tastes change over time
- [ ] Scale to **Apache Spark ALS** for million-user datasets
- [ ] Build a **Streamlit web app** where users can rate movies and get live recommendations
- [ ] Add **A/B testing framework** to compare models on simulated live traffic


License

Open source under MIT License. Dataset is synthetically generated for educational purposes.


Built as an advanced data science portfolio project · 2026
