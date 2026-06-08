from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = file.filename.lower()
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file.read()))
        else:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

        # Validate required columns
        required = ['date', 'amount']
        for col in required:
            if col not in df.columns:
                return jsonify({'error': f'Missing required column: {col}'}), 400

        df['date'] = pd.to_datetime(df['date'])
        df['day_of_month'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        features = df[['amount', 'day_of_month', 'day_of_week']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Find best K using silhouette
        best_k = 2
        best_score = -1
        silhouette_data = []
        inertia_data = []

        for k in range(2, min(8, len(df) // 5 + 1)):
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = km.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            silhouette_data.append({'k': k, 'score': round(score, 4)})
            inertia_data.append({'k': k, 'inertia': round(km.inertia_, 2)})
            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering with best K
        kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
        df['cluster'] = kmeans.fit_predict(features_scaled)

        # Cluster summaries
        cluster_summary = []
        for c in range(best_k):
            cdf = df[df['cluster'] == c]
            summary = {
                'cluster': int(c),
                'count': int(len(cdf)),
                'avg_amount': round(float(cdf['amount'].mean()), 2),
                'max_amount': round(float(cdf['amount'].max()), 2),
                'min_amount': round(float(cdf['amount'].min()), 2),
                'avg_day_of_month': round(float(cdf['day_of_month'].mean()), 1),
                'avg_day_of_week': round(float(cdf['day_of_week'].mean()), 1),
                'total_spend': round(float(cdf['amount'].sum()), 2),
            }
            if 'category' in df.columns:
                summary['top_category'] = cdf['category'].value_counts().idxmax()
            cluster_summary.append(summary)

        # Scatter plot data
        scatter = []
        for _, row in df.iterrows():
            point = {
                'amount': float(row['amount']),
                'day_of_month': int(row['day_of_month']),
                'cluster': int(row['cluster']),
            }
            if 'description' in df.columns:
                point['description'] = str(row['description'])
            if 'category' in df.columns:
                point['category'] = str(row['category'])
            scatter.append(point)

        # Monthly spend trend
        monthly = df.groupby('month')['amount'].sum().reset_index()
        monthly_data = [{'month': int(r['month']), 'total': round(float(r['amount']), 2)} for _, r in monthly.iterrows()]

        # Category breakdown if exists
        category_data = []
        if 'category' in df.columns:
            cat = df.groupby('category')['amount'].sum().reset_index()
            category_data = [{'category': r['category'], 'total': round(float(r['amount']), 2)} for _, r in cat.iterrows()]

        return jsonify({
            'best_k': best_k,
            'best_silhouette': round(best_score, 4),
            'total_transactions': len(df),
            'total_spend': round(float(df['amount'].sum()), 2),
            'cluster_summary': cluster_summary,
            'scatter': scatter,
            'silhouette_data': silhouette_data,
            'inertia_data': inertia_data,
            'monthly_data': monthly_data,
            'category_data': category_data,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
