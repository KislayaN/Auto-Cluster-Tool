import sys
import os
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.kmeans import KMeansClustering_
from src.models.gmm import gmm_pipeline
from src.models.agglomerative import AgglomerativeClustering_
from src.models.dbscan import DBSCAN_pipeline

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
        
class Auto_Cluster: 
    def __init__(self, dataset):
         self.data = dataset
         self.hopkins_stats = hopkins_stats(self.data)
         self.density_sweep_result = has_clean_eps_window(self.data)
         self.scores = {}
         
    def best_model(self):
        valid_scores = {k: v for k, v in self.scores.items() if v is not None}
        
        for model in valid_scores:
            silhouette = valid_scores[model]['Silhouette_Score']
            dav_b = valid_scores[model]['Davies_Bouldin_Score']
            cal_h = valid_scores[model]['Calinski_Harabasz_Score']
            
            valid_scores[model]['combined_scores'] = silhouette - (dav_b / 10) + (cal_h / 1000)
            
        best = max(valid_scores, key=lambda k: valid_scores[k]['combined_scores'])
        
        print("-------------------- Model Comparison --------------------")
        for model, v in valid_scores.items():
            marker = " <-- BEST" if model == best else ""
            print(f"Model: {model} | Silhouette: {v['Silhouette_Score']:.4f} | DB: {v['Davies_Bouldin_Score']:.4f} | CH: {v['Calinski_Harabasz_Score']:.2f}{marker}")
        
        print(f"\nThe best model for the given dataset is {best}")
            
        return best 
         
    def model_selector(self):
        
        pipelines = {}
        
        if self.hopkins_stats < 0.6:
            print(f"Hopkins_stats = {self.hopkins_stats}")
            print("This given dataset has not any clusters, try another datasets for ideal results")
        else:
            print(f"Hopkins_stats = {self.hopkins_stats}")
            if self.density_sweep_result:
                print("Clean Window Found")
                print("\nDBSCAN is running.... ")
                dbscan = DBSCAN_pipeline(dataset=self.data)
                dbscan.fit_predict()
                self.scores['DBSCAN'] = dbscan.evaluate()
                if self.scores['DBSCAN'] is None:
                    print("DBSCAN: 0 valid clusters found — excluded from comparison")
                pipelines['DBSCAN'] = dbscan
            
            print("\nGMM is running.... ")
            gmm_model = gmm_pipeline(data=self.data)
            gmm_model.fit_predict()
            best_k = gmm_model._find_optimal_components()
            print(f"{best_k} components found by Gaussian Mixture")
            result_gmm = gmm_model.evaluate()
            self.scores['GMM'] = result_gmm
            if self.scores['GMM'] is None:
                print("Gaussian Mixture: 0 valid clusters found — excluded from comparison")
            pipelines['GMM'] = gmm_model
            gmm_model.plot()

            print("\nKMeans is running.... ")
            kmeans_model = KMeansClustering_(k=best_k, dataset=self.data)
            kmeans_model.fit()
            result_km = kmeans_model.evaluate()
            self.scores['K-Means'] = result_km
            if self.scores['K-Means'] is None:
                print("K-Means: 0 valid clusters found — excluded from comparison")
            pipelines['K-Means'] = kmeans_model
            kmeans_model.plot()
            
            print("\nAgglomerative is running.... ")
            agglomerative_model = AgglomerativeClustering_(n_clusters=best_k, dataset=self.data)
            agglomerative_model.fit_predict()
            result_agg = agglomerative_model.evaluate()
            self.scores['Agglomerative'] = result_agg
            if self.scores['Agglomerative'] is None:
                print("Agglomerative: 0 valid clusters found — excluded from comparison")
            pipelines['Agglomerative'] = agglomerative_model
            agglomerative_model.plot_dendogram()
            
            best_name = self.best_model()
            print("Automatic Model Selection Performed")
            
        return pipelines[best_name]