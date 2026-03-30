import os 
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load import Load_Data
from data.processed_data import Preprocess
from notebooks.eda import EDA
from src.features.feature_engineering import Feature_Engineer
from src.models.auto_cluster import Auto_Cluster
from src.visualization.plots import plot_comparision, _plot_2d

def main(path, show_eda=True):
    
    # Load the data
    data_loader = Load_Data()
    data_loader.load_data(path=path)
    dataframe = data_loader.dataset
    
    # Processing 
    processor = Preprocess(dataset=dataframe, path=data_loader.path)
    processor.preprocess()
    X_dataframe = processor.X_dataframe
    feature_names = processor.feature_columns
    
    # EDA
    if show_eda:
        eda = EDA(
            X_dataframe=X_dataframe,
            feature_names=feature_names
        )
        eda.run()
    
    # Gathering Processed data
    processed_dataframe = processor.processed_df
    processed_dataframe_cols = processor.feature_columns
    
    # Feature Engineering
    feature_engineering = Feature_Engineer(
        data=processed_dataframe,
        dataframe_cols=processed_dataframe_cols
    )
    feature_engineering.perform()
    
    # Auto Cluster
    dataset = feature_engineering.data
    cluster_setup = Auto_Cluster(dataset=dataset)
    best_pipeline = cluster_setup.model_selector()
    
    # Scores Comparison
    plot_comparision(cluster_setup.scores)
    
    # 2D PCA plot
    _plot_2d(data=dataset, labels=best_pipeline.labels)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Cluster Tool")
    parser.add_argument('--path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--eda', type=bool, required=True, help='Show Exploratory Data Analysis (EDA) plots')
    
    args = parser.parse_args()
    main(path=args.path, show_eda=args.eda)