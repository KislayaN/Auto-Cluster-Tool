from sklearn.cluster import AgglomerativeClustering

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator

class AgglomerativeClustering_:
    def __init__(self, n_clusters, dataset=None):
        self.n_clusters = n_clusters
        self.metric = 'euclidean'
        self.linkage = 'ward'
        self.dataset = dataset
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage, metric=self.metric)
    
    def fit_predict(self):
        self.labels = self.model.fit_predict(self.dataset)
        return self.labels
    
    def evaluate(self):
        evaluator = Evaluator(dataset=self.dataset, labels=self.labels)
        evaluation = evaluator.evaluate()
        return evaluation