import pickle
from sklearn.cluster import KMeans

MODEL_FILE = 'model.pkl'

def train_kmeans(X):
    """
    Fit a 3-cluster KMeans on X and pickle the model.
    """
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(km, f)
    return km
