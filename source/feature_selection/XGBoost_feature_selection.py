import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from source.helper_functions.save_outputs import save_plot_to_file, save_output_to_file

def train_xgboost_model(X, y):
    """
    Train an XGBoost classifier.
    
    Args:
        X (np.ndarray): Input features matrix
        y (np.ndarray): Target labels
        
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    print("Training XGBoost model...")
    model.fit(X, y)
    return model

def create_importance_dataframe(importance_scores, feature_names):
    """
    Create a DataFrame with feature importance scores.
    
    Args:
        importance_scores (np.ndarray): Feature importance scores
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with feature names and importance scores
    """
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)

def plot_feature_importance(top_features, n_top_features):
    """
    Plot feature importance scores.
    
    Args:
        top_features (pd.DataFrame): DataFrame with top features and their importance
        n_top_features (int): Number of top features to display
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_features)), top_features['Importance'])
    plt.xticks(range(len(top_features)), top_features['Feature'], rotation=45, ha='right')
    plt.title(f'Top {n_top_features} Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()

def save_results(feature_importance, top_features, feature_names, n_top_features, save_dir):
    """
    Save feature importance analysis results.
    
    Args:
        feature_importance (pd.DataFrame): Full feature importance DataFrame
        top_features (pd.DataFrame): Top N features DataFrame
        feature_names (list): List of feature names
        n_top_features (int): Number of top features
        save_dir (str): Directory to save outputs
    """
    # Save plot
    save_plot_to_file(plt.gcf(), "feature_importance_plot.png", save_dir)
    
    # Save feature importance scores
    report = (f"Feature Importance Analysis\n"
             f"Total features: {len(feature_names)}\n"
             f"Top {n_top_features} features:\n\n"
             f"{top_features.to_string()}")
    save_output_to_file(report, "feature_importance_report.txt", save_dir)

def analyze_feature_importance(X, y, feature_names=None, n_top_features=20, save_dir=None):
    """
    Analyze feature importance using XGBoost and visualize results.
    
    Args:
        X (np.ndarray): Input features matrix
        y (np.ndarray): Target labels
        feature_names (list, optional): List of feature names
        n_top_features (int, optional): Number of top features to display
        save_dir (str, optional): Directory to save outputs
        
    Returns:
        tuple: (feature importance DataFrame, trained XGBoost model)
    """
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Train model and get importance scores
    model = train_xgboost_model(X, y)
    importance_scores = model.feature_importances_
    
    # Create and sort feature importance DataFrame
    feature_importance = create_importance_dataframe(importance_scores, feature_names)
    top_features = feature_importance.head(n_top_features)
    
    # Create visualization
    plot_feature_importance(top_features, n_top_features)
    
    # Save results if directory provided
    if save_dir:
        save_results(feature_importance, top_features, feature_names, n_top_features, save_dir)
    
    plt.show()
    return feature_importance, model