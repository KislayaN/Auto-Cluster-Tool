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

def _plot_2d(data, labels):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    for label in set(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {label}', alpha=0.8, edgecolors='k')
    plt.title("Best Model Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
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

def plot_comparision(scores: dict):
    valid = {k: v for k, v in scores.items() if v is not None}
    
    models = list(valid.keys())
    
    silhouette = [valid[m]['Silhouette_Score'] for m in models]
    dav_b = [valid[m]['Davies_Bouldin_Score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, axes = plt.subplots(figsize=(10,5))
    
    bars1 = axes.bar(x=x-width/2, height=silhouette, width=width, label='Silhouette (higher=better)', color='steelblue')
    bars2 = axes.bar(x=x+width/2, height=dav_b, width=width, label='Davies-Bouldin (lower=better)', color='coral')
    
    for bar in bars1:
        axes.text(bar.get_x() + bar.get_width() /2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', fontsize=9)
        
    for bar in bars2:
        axes.text(bar.get_x() + bar.get_width() /2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', fontsize=9)
        
    axes.set_xticks(x)
    axes.set_xticklabels(models)
    axes.set_title("Model comparison")
    axes.legend()
    plt.tight_layout()
    plt.show()