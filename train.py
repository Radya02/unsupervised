import pandas as pd, numpy as np, joblib, json, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned-Data.csv').drop(columns=['Country', 'Gender_Female', 'Gender_Male', 'Gender_Transgender'])
scaler = StandardScaler()
X = scaler.fit_transform(df)
# обучение
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X[:5000]).inertia_ for k in range(2,8)]
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
pca = PCA(n_components=2, random_state=42).fit(X)
X2 = pca.transform(X)

joblib.dump(km, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
json.dump(list(df.columns), open('features.json', 'w'))

colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12']
fig, axes = plt.subplots(1, 2, figsize=(13,5))
fig.suptitle('COVID-19 K-Means Clustering (K=4)', fontsize=14, fontweight='bold')
axes[0].plot(range(2,8), inertias, 'bo-', lw=2)
axes[0].axvline(4, color='red', ls='--', label='K=4')
axes[0].set(title='Elbow Method', xlabel='K', ylabel='Inertia'); axes[0].legend(); axes[0].grid(alpha=.3)
idx = np.random.choice(len(X2), 3000, replace=False)
for i,c in enumerate(colors):
    m = km.labels_[idx]==i
    axes[1].scatter(X2[idx][m,0], X2[idx][m,1], c=c, s=8, alpha=.5, label=f'Cluster {i}')
axes[1].set(title='PCA Visualization', xlabel='PC1', ylabel='PC2'); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig('figure.png', dpi=130, bbox_inches='tight')
print("✅ Done: model.pkl, scaler.pkl, figure.png")
