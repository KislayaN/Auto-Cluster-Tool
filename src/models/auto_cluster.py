# Algorithm profiler: analyzes the data and tells which algorithm to use: 
# - Run density sweep -> if clean eps window exists -> flag dbscan
# - check cluster tendancy (hopkins statistic) -> is data even clusterable
# - Default to GMM if not clear density structure

import sys
import os
import numpy as np

from sklearn.cluster import DBSCAN

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator
from data.load import Load_Data
from data.processed_data import Preprocess
from src.features.feature_engineering import Feature_Engineer
from sklearn.neighbors import NearestNeighbors

from sklearn.datasets import make_blobs

loader = Load_Data()
dataset = loader.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')
raw_data = loader.dataset
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
processor = Preprocess(dataset=raw_data, path=loader.path)
processed_dataframe = processor.preprocess()
feature_engine = Feature_Engineer(data=processed_dataframe, dataframe_cols=processor.feature_columns)
X_dataframe = feature_engine.perform()
feature_columns = feature_engine.feature_engineered_df_cols

def has_clean_eps_window(X, min_samples=4, target_cluster=(2,4), window_size = 3):
    stable_count = 0
    for eps in np.arange(0.3, 3.0, 0.1):
        labels = DBSCAN(min_samples=min_samples, eps=eps).fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        noise_ratio = (labels == -1).sum() / len(X)
        print(f"eps={eps:.1f} | clusters={n_clusters} | noise={round(noise_ratio*100, 1)}")
        
        if target_cluster[0] <= n_clusters <= target_cluster[1] and noise_ratio < 0.15:
            stable_count += 1
        else: 
            stable_count = 0
            
    if stable_count > window_size: 
            return True
    else: 
            return False

def hopkins_stats(X, sample_ratio = 0.1):
    X = np.array(X)
    n = X.shape[0]
    d = X.shape[1]
    m = int(n * sample_ratio) # sample size
    
    # random sample from actual data
    random_indices = np.random.choice(n, m, replace=False)
    X_sample = X[random_indices]
    
    # generate uniformly random points in the same space
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    X_random = np.random.uniform(mins, maxs, (m, d))
    
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    
    # Distance from real sample to nearest real neighbor
    real_distances = nn.kneighbors(X_sample)[0][:, 1]
    
    # Distance from random points to nearest real neighbor  
    random_distances = nn.kneighbors(X_random)[0][:, 0]
    
    u = np.sum(random_distances)
    w = np.sum(real_distances)
    
    return u / (u + w)
