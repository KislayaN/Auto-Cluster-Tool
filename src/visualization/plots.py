import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def feature_dist_scatter_plot(dataframe, feature_names, bins):
    
    # Number of rows to be kept 
    number_of_features = len(feature_names)
    
    cols = math.ceil(math.sqrt(number_of_features))
    rows = math.ceil(number_of_features / cols)
     
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        axes[i].hist(dataframe[feature], bins=bins, edgecolor='black')
        axes[i].set_title(f"Distribution of {feature}")
        
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()
    
def feature_correlation_plot(correlation_matrix, annot):
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm')
    plt.show()
    
def _plot_3d(data, n_components, top_drivers, processor):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Access the target data from the processor's dataset
    target_data = processor.dataset[processor.target]
    
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                         c=pd.Categorical(target_data).codes, 
                         cmap='viridis', alpha=0.6)
    
    ax.set_title(f"3D Cluster View (Top 3 of {n_components} PCs)")
    ax.set_xlabel(f"PC1: {top_drivers['PC1']}")
    ax.set_ylabel(f"PC2: {top_drivers['PC2']}")
    ax.set_zlabel(f"PC3: {top_drivers['PC3']}")
    plt.show()

def _plot_2d(data, pca, top_drivers, processor):
    plt.figure(figsize=(8, 6))
    # Access the target column correctly
    target_data = processor.dataset[processor.target]
    
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=target_data, palette='viridis')
    plt.title(f"2D Cluster View (Total PCs: {pca.n_components_})")
    plt.xlabel(f"PC1 ({top_drivers['PC1']}): {pca.explained_variance_ratio_[0]:.1%}")
    plt.ylabel(f"PC2 ({top_drivers['PC2']}): {pca.explained_variance_ratio_[1]:.1%}")
    plt.show()

def _scree_plot(explained_variance_ratio, n_selected):
    plt.figure(figsize=(8, 4))
    cumulative_var = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, n_selected + 1), cumulative_var, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':')
    plt.title("Cumulative Explained Variance (95% Threshold)")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.grid(True, alpha=0.3)
    plt.show()