#!/usr/bin/env python3
# Metabolic Pathway Network Analysis
# This script performs clustering analysis on metabolic pathway network data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set the random seed for reproducibility
np.random.seed(42)

# Function to load the data
def load_data(file_path):
    """
    Load the metabolic pathway network data from a file
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the pathway network data
    """
    # Define column names based on the provided description
    columns = [
        'Pathway', 'ConnectedComponents', 'Diameter', 'Radius', 'Centralization',
        'ShortestPath', 'CharacteristicPathLength', 'AvgNumNeighbours', 'Density',
        'Heterogeneity', 'IsolatedNodes', 'NumberOfSelfLoops', 'MultiEdgeNodePair',
        'NeighborhoodConnectivity', 'NumberOfDirectedEdges', 'Stress', 'SelfLoops',
        'PartnerOfMultiEdgedNodePairs', 'Degree', 'TopologicalCoefficient',
        'BetweennessCentrality', 'Radiality', 'Eccentricity', 'NumberOfUndirectedEdges',
        'ClosenessCentrality', 'AverageShortestPathLength', 'ClusteringCoefficient',
        'nodeCount', 'edgeCount'
    ]
    
    # Try different delimiters as the file format wasn't specified
    try:
        # First, try comma-separated
        df = pd.read_csv(file_path, names=columns, na_values=['?'])
    except:
        try:
            # Try tab-separated
            df = pd.read_csv(file_path, sep='\t', names=columns, na_values=['?'])
        except:
            try:
                # Try space-separated
                df = pd.read_csv(file_path, sep='\\s+', names=columns, na_values=['?'])
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Please check the file format and try again.")
                return None
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

# Function to preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    
    Returns:
    --------
    tuple
        (Preprocessed data, Feature names, Scaler object)
    """
    print("\n--- Data Preprocessing ---")
    
    # First, separate the pathway text column if it exists
    if 'Pathway' in df.columns:
        pathway_names = df['Pathway']
        df_features = df.drop('Pathway', axis=1)
    else:
        pathway_names = pd.Series([f"Pathway_{i}" for i in range(len(df))])
        df_features = df
    
    # Convert all columns to numeric, forcing errors to NaN
    print("Converting columns to numeric types...")
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    
    # Display missing values
    missing_values = df_features.isnull().sum()
    print(f"Missing values per column:\n{missing_values[missing_values > 0]}")
    
    # Replace infinite values with NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values using SimpleImputer with mean strategy
    print("Imputing missing values using mean imputation...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_features),
        columns=df_features.columns
    )
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_imputed),
        columns=df_imputed.columns
    )
    
    print(f"Preprocessing complete. Shape after preprocessing: {df_scaled.shape}")
    return df_scaled, pathway_names, scaler

# Function for exploratory data analysis
def perform_eda(df, df_scaled):
    """
    Perform exploratory data analysis on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    df_scaled : pd.DataFrame
        Preprocessed and scaled data
    """
    print("\n--- Exploratory Data Analysis ---")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe().T)
    
    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    print("Correlation matrix saved as 'correlation_matrix.png'")
    
    # PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title('PCA Visualization of Metabolic Pathway Data', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('pca_visualization.png')
    plt.close()
    print("PCA visualization saved as 'pca_visualization.png'")
    
    return pca, pca_result

# Function to find optimal number of clusters for K-means
def find_optimal_clusters(df_scaled):
    """
    Find the optimal number of clusters for K-means using the elbow method
    and silhouette scores
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        Preprocessed and scaled data
    
    Returns:
    --------
    int
        Optimal number of clusters
    """
    print("\n--- Finding Optimal Number of Clusters ---")
    
    # Sample data if too large (for faster computation)
    if len(df_scaled) > 10000:
        print(f"Dataset is large ({len(df_scaled)} samples). Sampling 10,000 points for cluster analysis...")
        df_sample = df_scaled.sample(10000, random_state=42)
    else:
        df_sample = df_scaled
    
    # Elbow method
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    k_range = range(2, 21)  # Try from 2 to 20 clusters
    
    for k in k_range:
        print(f"Testing k={k}...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_sample)
        
        # Calculate metrics
        inertia.append(kmeans.inertia_)
        labels = kmeans.labels_
        
        try:
            silhouette = silhouette_score(df_sample, labels)
            calinski = calinski_harabasz_score(df_sample, labels)
            davies_bouldin = davies_bouldin_score(df_sample, labels)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            silhouette = np.nan
            calinski = np.nan
            davies_bouldin = np.nan
        
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        davies_bouldin_scores.append(davies_bouldin)
        print(f"Silhouette: {silhouette:.3f}")
    
    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Elbow plot
    axs[0, 0].plot(k_range, inertia, 'bo-')
    axs[0, 0].set_xlabel('Number of clusters')
    axs[0, 0].set_ylabel('Inertia')
    axs[0, 0].set_title('Elbow Method')
    axs[0, 0].grid(alpha=0.3)
    
    # Silhouette score
    axs[0, 1].plot(k_range, silhouette_scores, 'ro-')
    axs[0, 1].set_xlabel('Number of clusters')
    axs[0, 1].set_ylabel('Silhouette Score')
    axs[0, 1].set_title('Silhouette Analysis')
    axs[0, 1].grid(alpha=0.3)
    
    # Calinski-Harabasz Index
    axs[1, 0].plot(k_range, calinski_scores, 'go-')
    axs[1, 0].set_xlabel('Number of clusters')
    axs[1, 0].set_ylabel('Calinski-Harabasz Index')
    axs[1, 0].set_title('Calinski-Harabasz Analysis')
    axs[1, 0].grid(alpha=0.3)
    
    # Davies-Bouldin Index
    axs[1, 1].plot(k_range, davies_bouldin_scores, 'mo-')
    axs[1, 1].set_xlabel('Number of clusters')
    axs[1, 1].set_ylabel('Davies-Bouldin Index')
    axs[1, 1].set_title('Davies-Bouldin Analysis')
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters_analysis.png')
    plt.close()
    print("Optimal clusters analysis saved as 'optimal_clusters_analysis.png'")
    
    # Find the best K using silhouette score (higher is better)
    best_k_silhouette = k_range[np.nanargmax(silhouette_scores)]
    
    # Find the best K using Calinski-Harabasz Index (higher is better)
    best_k_calinski = k_range[np.nanargmax(calinski_scores)]
    
    # Find the best K using Davies-Bouldin Index (lower is better)
    best_k_davies = k_range[np.nanargmin(davies_bouldin_scores)]
    
    print(f"\nBest K using Silhouette Score: {best_k_silhouette}")
    print(f"Best K using Calinski-Harabasz Index: {best_k_calinski}")
    print(f"Best K using Davies-Bouldin Index: {best_k_davies}")
    
    # Use the silhouette score as the final decision
    optimal_k = best_k_silhouette
    print(f"\nSelected optimal number of clusters: {optimal_k}")
    
    return optimal_k

# Function to run K-means clustering
def run_kmeans(df_scaled, optimal_k, pca_result, pca):
    """
    Run K-means clustering with the optimal number of clusters
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        Preprocessed and scaled data
    optimal_k : int
        Optimal number of clusters
    pca_result : np.ndarray
        PCA results for visualization
    pca : PCA
        Fitted PCA object for transforming centroids
    
    Returns:
    --------
    KMeans
        Fitted K-means model
    """
    print(f"\n--- Running K-means Clustering with {optimal_k} clusters ---")
    
    # Create and fit K-means model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    
    # Calculate metrics
    silhouette = silhouette_score(df_scaled, labels)
    calinski = calinski_harabasz_score(df_scaled, labels)
    davies_bouldin = davies_bouldin_score(df_scaled, labels)
    
    print(f"K-means clustering metrics:")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
    
    # Visualize clusters on PCA plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, 
                         cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Cluster')
    
    # Add centroids - now using the pca parameter
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, marker='X', 
               c='red', edgecolor='black', label='Centroids')
    
    plt.title(f'K-means Clustering Results (k={optimal_k})', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('kmeans_clusters.png')
    plt.close()
    print("K-means clusters visualization saved as 'kmeans_clusters.png'")
    
    return kmeans, labels
# Function to analyze cluster characteristics
def analyze_clusters(df, pathway_names, labels, optimal_k):
    """
    Analyze the characteristics of each cluster
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    pathway_names : pd.Series
        Names of the pathways
    labels : np.ndarray
        Cluster labels
    optimal_k : int
        Number of clusters
    """
    print("\n--- Analyzing Cluster Characteristics ---")
    
    # Add cluster labels to the original data
    df_with_clusters = df.copy()
    if 'Pathway' in df_with_clusters.columns:
        df_with_clusters = df_with_clusters.drop('Pathway', axis=1)
    df_with_clusters['Pathway'] = pathway_names
    df_with_clusters['Cluster'] = labels
    
    # Calculate summary statistics for each cluster
    for cluster in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        pathways_in_cluster = cluster_data['Pathway'].tolist()
        
        print(f"\nCluster {cluster} Statistics:")
        print(f"Number of pathways: {len(cluster_data)}")
        print(f"Sample pathways: {pathways_in_cluster[:5]}")
        print("\nFeature statistics:")
        
        # Display statistics for key features
        cluster_stats = cluster_data.drop(['Pathway', 'Cluster'], axis=1).describe().T[['mean', 'std', 'min', 'max']]
        cluster_stats = cluster_stats.sort_values('mean', ascending=False)
        print(cluster_stats.head(10))  # Show top 10 features by mean value
    
    # Save cluster assignments to file
    cluster_assignments = pd.DataFrame({
        'Pathway': pathway_names,
        'Cluster': labels
    })
    cluster_assignments.to_csv('cluster_assignments.csv', index=False)
    print("\nCluster assignments saved to 'cluster_assignments.csv'")

# Function to compare different clustering models
def compare_models(df_scaled, optimal_k, pca_result):
    """
    Compare different clustering models
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        Preprocessed and scaled data
    optimal_k : int
        Optimal number of clusters
    pca_result : np.ndarray
        PCA results for visualization
    
    Returns:
    --------
    dict
        Dictionary containing the models and their performance metrics
    """
    print("\n--- Comparing Different Clustering Models ---")
    
    # Sample data if too large (for faster computation)
    if len(df_scaled) > 10000:
        print(f"Dataset is large ({len(df_scaled)} samples). Sampling 10,000 points for model comparison...")
        # Use pandas sample to maintain DataFrame structure
        sample_indices = np.random.RandomState(42).choice(len(df_scaled), 
                                                         size=min(10000, len(df_scaled)), 
                                                         replace=False)
        df_sample = df_scaled.iloc[sample_indices]
        if isinstance(pca_result, np.ndarray) and len(pca_result) == len(df_scaled):
            pca_sample = pca_result[sample_indices]
        else:
            print("Warning: PCA result size doesn't match dataframe. Using full PCA result.")
            pca_sample = pca_result
    else:
        df_sample = df_scaled
        pca_sample = pca_result
    
        
    models = { 'K-means': KMeans(n_clusters=optimal_k, random_state=42, n_init=10) }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(df_scaled)
        
        # Get labels
        if name == 'Gaussian Mixture':
            labels = model.predict(df_scaled)
        else:
            labels = model.labels_
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(df_scaled, labels)
            calinski = calinski_harabasz_score(df_scaled, labels)
            davies_bouldin = davies_bouldin_score(df_scaled, labels)
            
            print(f"{name} metrics:")
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Calinski-Harabasz Index: {calinski:.3f}")
            print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
            
            results[name] = {
                'model': model,
                'labels': labels,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies_bouldin': davies_bouldin
            }
            
            # Visualize clusters
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=labels, cmap='viridis', alpha=0.8, s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'{name} Clustering Results', fontsize=14)
            plt.xlabel(f'PC1', fontsize=12)
            plt.ylabel(f'PC2', fontsize=12)
            plt.grid(alpha=0.3)
            plt.savefig(f'{name.lower().replace(" ", "_")}_clusters.png')
            plt.close()
            print(f"{name} clusters visualization saved as '{name.lower().replace(' ', '_')}_clusters.png'")
            
        except Exception as e:
            print(f"Error calculating metrics for {name}: {e}")
            results[name] = {
                'model': model,
                'labels': labels,
                'silhouette': np.nan,
                'calinski': np.nan,
                'davies_bouldin': np.nan
            }
    
    # Compare model performance
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Silhouette Score': [results[model].get('silhouette', np.nan) for model in results],
        'Calinski-Harabasz Index': [results[model].get('calinski', np.nan) for model in results],
        'Davies-Bouldin Index': [results[model].get('davies_bouldin', np.nan) for model in results]
    })
    
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Save comparison to file
    model_comparison.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to 'model_comparison.csv'")
    
    # Visualize model comparison
    plt.figure(figsize=(12, 8))
    
    # Silhouette Score (higher is better)
    plt.subplot(1, 3, 1)
    sns.barplot(x='Model', y='Silhouette Score', data=model_comparison)
    plt.title('Silhouette Score\n(higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Calinski-Harabasz Index (higher is better)
    plt.subplot(1, 3, 2)
    sns.barplot(x='Model', y='Calinski-Harabasz Index', data=model_comparison)
    plt.title('Calinski-Harabasz Index\n(higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Davies-Bouldin Index (lower is better)
    plt.subplot(1, 3, 3)
    sns.barplot(x='Model', y='Davies-Bouldin Index', data=model_comparison)
    plt.title('Davies-Bouldin Index\n(lower is better)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    plt.close()
    print("Model comparison visualization saved as 'model_comparison.png'")
    
    return results

# Main function to run the analysis
def main():
    """
    Main function to run the analysis
    """
    print("=== Metabolic Pathway Network Analysis ===")
    
    # Load data
    file_path = "Reaction Network (Undirected).data"
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Preprocess data
    df_scaled, pathway_names, scaler = preprocess_data(df)
    
    # Perform EDA
    pca, pca_result = perform_eda(df, df_scaled)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(df_scaled)
    
    # Run K-means clustering - now passing pca as an argument
    kmeans, labels = run_kmeans(df_scaled, optimal_k, pca_result, pca)
    
    # Analyze cluster characteristics
    analyze_clusters(df, pathway_names, labels, optimal_k)
    
    # Compare different models - consider passing pca here too if needed
    results = compare_models(df_scaled, optimal_k, pca_result)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated CSV files and images for results.")
if __name__ == "__main__":
    main()
