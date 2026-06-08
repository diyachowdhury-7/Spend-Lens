# Spend Lens — Spending Pattern Dashboard

An unsupervised machine learning web app that clusters personal transaction data to reveal hidden spending patterns.
Built with K-Means clustering, served via Flask, with an editorial-style dashboard UI.

You upload a CSV or Excel file of transactions, the app automatically groups them into behavioral clusters using K-Means,
finds the optimal number of clusters using silhouette scoring, and visualizes the patterns across a dashboard with scatter plots,
monthly trends, and category breakdowns.

Unlike a basic expense tracker that shows what you spent on, this reveals how you behave — high-spend end-of-month bursts,
daily impulse patterns, fixed recurring costs — without you labeling anything.

---

## ML pipeline

- **Feature engineering** — extracts `amount`, `day_of_month`, and `day_of_week` from raw transactions
- **Normalization** — applies z-score standardization (StandardScaler) so no feature dominates distance calculations; chosen over min-max specifically for outlier robustness
- **K selection** — runs K-Means for K=2 through K=7, scores each with silhouette score, picks the K with best cluster separation
- **Clustering** — final K-Means run using K-Means++ initialization and 10 random restarts to avoid local minima
- **Interpretation** — cluster averages are computed and labeled based on amount + timing heuristics

---

## Project structure
spend-lens/
├── app.py
├── index.html
└── README.md

---

## Input format

Accepts `.csv` or `.xlsx`. Minimum required columns:

| date | amount |
|---|---|
| 2024-01-01 | 450 |
| 2024-01-02 | 1200 |

Optional columns `description` and `category` enhance cluster cards if present.

---

## What I learned

- Unsupervised learning fundamentals — no labels, no loss function, structure-finding
- Why normalization method matters (z-score vs min-max, outlier sensitivity)
- K-Means++ initialization and why random init causes local minima
- Elbow method vs silhouette score for choosing K
- Interpreting unlabeled clusters by reading feature averages
