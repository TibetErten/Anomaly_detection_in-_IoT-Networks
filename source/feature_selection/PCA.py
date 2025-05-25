
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from source.helper_functions.save_outputs import save_plot_to_file, save_output_to_file

def standardize_features(X):
    """
    Standardize the input features using StandardScaler.
    
    Args:
        X (np.ndarray): Input features matrix
        
    Returns:
        np.ndarray: Standardized features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def calculate_optimal_components(pca, variance_threshold):
    """
    Calculate the optimal number of components based on variance threshold.
    
    Args:
        pca: Fitted PCA object
        variance_threshold (float): Minimum cumulative explained variance ratio
        
    Returns:
        tuple: (optimal number of components, cumulative variance ratio)
    """
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components_threshold = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    return n_components_threshold, cumulative_variance_ratio

def plot_variance_ratio(cumulative_variance_ratio, n_components_threshold, variance_threshold):
    """
    Plot the cumulative explained variance ratio.
    
    Args:
        cumulative_variance_ratio (np.ndarray): Cumulative explained variance ratios
        n_components_threshold (int): Optimal number of components
        variance_threshold (float): Variance threshold used
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio, 'bo-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=n_components_threshold, color='g', linestyle='--')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)

def save_pca_results(pca, n_components_threshold, variance_threshold, save_dir):
    """
    Save PCA analysis results and plots.
    
    Args:
        pca: Fitted PCA object
        n_components_threshold (int): Optimal number of components
        variance_threshold (float): Variance threshold used
        save_dir (str): Directory to save outputs
    """
    save_plot_to_file(plt.gcf(), "pca_explained_variance.png", save_dir)
    
    report = (f"Total number of components: {len(pca.explained_variance_ratio_)}\n"
             f"Components needed for {variance_threshold*100}% variance: {n_components_threshold}\n"
             f"Explained variance ratios:\n{pca.explained_variance_ratio_}")
    save_output_to_file(report, "pca_report.txt", save_dir)

def apply_pca(X, n_components=None, variance_threshold=0.95, save_dir=None):
    """
    Apply PCA to the input data and visualize the explained variance ratio.
    
    Args:
        X (np.ndarray): Input features matrix
        n_components (int, optional): Number of components to keep. If None, use variance_threshold
        variance_threshold (float, optional): Minimum cumulative explained variance ratio
        save_dir (str, optional): Directory to save the plots and reports
        
    Returns:
        tuple: (transformed data, fitted PCA object, explained variance ratio)
    """
    # Standardize features
    X_scaled = standardize_features(X)
    
    # Initialize and fit PCA
    if n_components is None:
        n_components = min(X_scaled.shape)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate optimal components
    n_components_threshold, cumulative_variance_ratio = calculate_optimal_components(
        pca, variance_threshold
    )
    
    # Plot results
    plot_variance_ratio(cumulative_variance_ratio, n_components_threshold, variance_threshold)
    
    # Save results if directory provided
    if save_dir:
        save_pca_results(pca, n_components_threshold, variance_threshold, save_dir)
    
    plt.show()
    
    # Apply optimal PCA transformation
    pca_optimal = PCA(n_components=n_components_threshold)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)
    
    return X_pca_optimal, pca_optimal, pca.explained_variance_ratio_