import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df=pd.read_csv('transactions.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_month'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['date'].dt.month
features = df[['amount', 'day_of_month', 'day_of_week']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(features_scaled[:5])
silhouette_scores = []
k_range = range(2, 11)  # silhouette needs at least 2 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(score)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

print(df.groupby('cluster')[['amount', 'day_of_month', 'day_of_week']].mean())
plt.scatter(df['amount'], df['day_of_month'], 
            c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Amount')
plt.ylabel('Day of Month')
plt.title('Spending Clusters')
plt.colorbar(label='Cluster')
plt.show()
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()