# Metabolic Pathway Network Analysis

This project performs clustering analysis on metabolic pathway network data to identify meaningful patterns and groupings among metabolic pathways based on their network characteristics.

## Project Overview

The analysis pipeline includes:

1. Data loading and preprocessing
2. Exploratory data analysis with visualizations
3. Determination of optimal cluster numbers
4. K-means clustering implementation
5. Cluster analysis and characterization
6. Clustering model comparison and evaluation

## Requirements

- Python 3.6+
- pip (Python package installer)

## Setup Instructions

### 1. Clone the Repository

First, clone this repository to your local machine or download the source code files.

```bash
git clone https://github.com/satuiro/kegg-undirected
cd kegg-undirected
```

### 2. Create a Virtual Environment

Creating a virtual environment is recommended to isolate the project dependencies.

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating that the virtual environment is active.

### 3. Install Dependencies

Install all required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


### 4. Prepare Data

Place your metabolic pathway network data file named `Reaction Network (Undirected).data` in the project root directory. The script expects data in a specific format with columns as described in the code comments.

### 5. Run the Analysis

Execute the main script:

```bash
python metabolic_pathway_analysis.py
```

## Output Files

The script generates several output files:

- **correlation_matrix.png**: Heatmap visualizing the correlation between different features
- **pca_visualization.png**: PCA visualization of the metabolic pathway data
- **optimal_clusters_analysis.png**: Analysis plots for determining the optimal number of clusters
- **kmeans_clusters.png**: Visualization of the K-means clustering results
- **cluster_assignments.csv**: CSV file containing the pathway names and their assigned clusters
- **model_comparison.csv**: Comparison of different clustering models and their performance metrics
- **model_comparison.png**: Visual comparison of clustering model performance

## Customizing the Analysis

To customize the analysis, you can modify the following in `metabolic_pathway_analysis.py`:

- Change the input file path in the `main()` function
- Adjust the range of clusters to test in the `find_optimal_clusters()` function
- Add additional clustering algorithms in the `compare_models()` function

## Troubleshooting

### Common Issues:

1. **Missing data file**: Ensure that the data file `Reaction Network (Undirected).data` is in the correct location.

2. **Import errors**: Make sure all dependencies are correctly installed using the provided pip command.

3. **Memory issues**: If you encounter memory errors with large datasets, try increasing the sample size reduction in the code or run the analysis on a machine with more RAM.

4. **Visualization errors**: If you encounter issues with visualizations, ensure matplotlib and seaborn are correctly installed and try updating them to the latest versions.
