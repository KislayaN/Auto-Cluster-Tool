import sys
import os
import pandas as pd
from sklearn.decomposition import PCA


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualization.plots import feature_dist_scatter_plot, feature_correlation_plot, _plot_2d, _plot_3d, _scree_plot

class EDA:
    def __init__(self, X_dataframe=None, feature_names=None):
        self.X_dataframe = X_dataframe
        self.feature_names = feature_names
        
    def run(self):
        for col in self.feature_names:
            print(f"\nDataset Median for {col} is {self.X_dataframe[col].median()}")
            print(f"Dataset description for {col} is \n{self.X_dataframe[col].describe()}")

        # Feature Distribution Plot
        feature_dist_scatter_plot(dataframe=self.X_dataframe, feature_names=self.feature_names, bins=20)

        # Correlation plot
        feature_correlation_plot(self.X_dataframe.corr(), annot=True)


class PCA_Plot:
    def __init__(self, processed_dataframe, n_components, processor=None):
        self.processed_dataframe = processed_dataframe
        self.n_components = n_components
        self.processor = processor
        
    def fit(self):
        pca_viz = PCA(n_components=self.n_components)
        data_pca = pca_viz.fit_transform(self.processed_dataframe)
        
        n_selected = pca_viz.n_components_
        pc_names = [f'PC{i+1}' for i in range(n_selected)]
        
        loadings = pd.DataFrame(
            pca_viz.components_.T,
            columns=pc_names,
            index=self.processor.feature_columns
        )
        
        top_drivers = {pc: loadings[pc].abs().idxmax() for pc in pc_names}
        for pc, driver in top_drivers.items():
            print(f"Top Driver for {pc}: {driver}")
            
        if n_selected == 2:

            _plot_2d(data_pca, pca_viz, top_drivers, self.processor)
        elif n_selected >= 3: 

            _plot_3d(data_pca[:, :3], n_selected, top_drivers, self.processor)

        _scree_plot(pca_viz.explained_variance_ratio_, n_selected)