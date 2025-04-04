#!/usr/bin/env python3
"""
Machine Learning Model Trainer for Quantum Spin Network Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from datetime import datetime

def load_and_explore_data(csv_file):
    """
    Load and perform initial data exploration
    
    Parameters:
    -----------
    csv_file : str - Path to CSV file with quantum simulation data
    
    Returns:
    --------
    pandas.DataFrame : Loaded and processed dataframe
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of configurations: {len(df)}")
    print(f"Number of features: {df.shape[1]}")
    
    # Display column names and types
    print("\nColumn names and types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found")
    
    return df

def prepare_data_for_ml(df, target_features, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning by splitting into training and testing sets
    
    Parameters:
    -----------
    df : pandas.DataFrame - Input dataframe
    target_features : list - Features to predict (K-coupling values)
    test_size : float - Fraction of data to use for testing
    random_state : int - Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"\nWarning: Found {nan_count} NaN values in the dataset")
        print("Columns with NaN values:")
        print(df.isna().sum()[df.isna().sum() > 0])
        
        # Option 1: Drop rows with NaN values
        df_cleaned = df.dropna()
        print(f"Dropped {len(df) - len(df_cleaned)} rows with NaN values")
        print(f"Remaining rows: {len(df_cleaned)}")
        
        if len(df_cleaned) < 10:
            print("Error: Too few samples remaining after dropping NaN values")
            print("Attempting to replace NaN values with median instead")
            # Option 2: Replace NaN values with median
            df_cleaned = df.copy()
            for col in df.columns:
                if df[col].isna().sum() > 0 and col not in ['config_id'] + target_features:
                    median_val = df[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    print(f"  - Replaced NaN values in column '{col}' with median: {median_val}")
            
            # Check if we still have NaNs in target features
            if df_cleaned[target_features].isna().sum().sum() > 0:
                print("Error: NaN values in target features. These rows must be dropped.")
                df_cleaned = df_cleaned.dropna(subset=target_features)
                print(f"Rows after dropping NaN targets: {len(df_cleaned)}")
        
        df = df_cleaned
    
    # Separate input features from target features
    if 'config_id' in df.columns:
        X = df.drop(columns=target_features + ['config_id'])
    else:
        X = df.drop(columns=target_features)
    
    y = df[target_features]
    
    feature_names = X.columns.tolist()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, target_features):
    """
    Train multiple regression models for each target variable separately
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : Training and testing data
    feature_names : list - Names of input features
    target_features : list - Names of target features
    
    Returns:
    --------
    dict : Dictionary of trained models and their performance metrics
    """
    # Initialize dictionary to store models and their metrics
    models = {}
    
    # Train separate models for each target feature
    for i, target in enumerate(target_features):
        print(f"\n=== Training models for target: {target} ===")
        
        # Extract this specific target's values
        y_train_target = y_train[target]
        y_test_target = y_test[target]
        
        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train_target)
        rf_pred = rf_model.predict(X_test)
        
        rf_mse = mean_squared_error(y_test_target, rf_pred)
        rf_r2 = r2_score(y_test_target, rf_pred)
        print(f"Mean Squared Error: {rf_mse:.4f}")
        print(f"R² Score: {rf_r2:.4f}")
        
        # Feature importance for Random Forest
        rf_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 important features (Random Forest):")
        print(rf_importances.head(10))
        
        # Train Gradient Boosting Regressor
        print("\nTraining Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train_target)
        gb_pred = gb_model.predict(X_test)
        
        gb_mse = mean_squared_error(y_test_target, gb_pred)
        gb_r2 = r2_score(y_test_target, gb_pred)
        print(f"Mean Squared Error: {gb_mse:.4f}")
        print(f"R² Score: {gb_r2:.4f}")
        
        # Train Neural Network (MLP)
        print("\nTraining Neural Network (MLP)...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            activation='relu',
            solver='adam',
            random_state=42
        )
        nn_model.fit(X_train, y_train_target)
        nn_pred = nn_model.predict(X_test)
        
        nn_mse = mean_squared_error(y_test_target, nn_pred)
        nn_r2 = r2_score(y_test_target, nn_pred)
        print(f"Mean Squared Error: {nn_mse:.4f}")
        print(f"R² Score: {nn_r2:.4f}")
        
        # Store models and metrics for this target
        models[target] = {
            'random_forest': {
                'model': rf_model,
                'predictions': rf_pred,
                'mse': rf_mse,
                'r2': rf_r2,
                'importances': rf_importances
            },
            'gradient_boosting': {
                'model': gb_model,
                'predictions': gb_pred,
                'mse': gb_mse,
                'r2': gb_r2
            },
            'neural_network': {
                'model': nn_model,
                'predictions': nn_pred,
                'mse': nn_mse,
                'r2': nn_r2
            }
        }
        
        # Visualize results for this target
        plot_prediction_results_single_target(y_test_target, rf_pred, gb_pred, nn_pred, target)
    
    return models

def plot_prediction_results_single_target(y_true, rf_pred, gb_pred, nn_pred, target_name):
    """
    Visualize prediction results for a single target
    
    Parameters:
    -----------
    y_true : pandas.Series - True target values
    rf_pred, gb_pred, nn_pred : numpy.ndarray - Predictions from different models
    target_name : str - Name of the target feature
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Random Forest
    axs[0].scatter(y_true, rf_pred, alpha=0.7)
    axs[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    axs[0].set_xlabel(f'True {target_name}')
    axs[0].set_ylabel(f'Predicted {target_name}')
    axs[0].set_title(f'Random Forest: {target_name}')
    
    # Gradient Boosting
    axs[1].scatter(y_true, gb_pred, alpha=0.7)
    axs[1].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    axs[1].set_xlabel(f'True {target_name}')
    axs[1].set_ylabel(f'Predicted {target_name}')
    axs[1].set_title(f'Gradient Boosting: {target_name}')
    
    # Neural Network
    axs[2].scatter(y_true, nn_pred, alpha=0.7)
    axs[2].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    axs[2].set_xlabel(f'True {target_name}')
    axs[2].set_ylabel(f'Predicted {target_name}')
    axs[2].set_title(f'Neural Network: {target_name}')
    
    plt.tight_layout()
    plt.savefig(f'prediction_results_{target_name}.png', dpi=300)
    plt.show()

def visualize_feature_importance(models, feature_names, top_n=20):
    """
    Visualize feature importance from Random Forest model for each target
    
    Parameters:
    -----------
    models : dict - Dictionary of trained models
    feature_names : list - Names of input features
    top_n : int - Number of top features to display
    """
    # Check if we have enough features
    max_features = min(top_n, len(feature_names))
    
    # Create a multi-target feature importance visualization
    plt.figure(figsize=(15, 10))
    
    # For each target feature
    for i, (target, target_models) in enumerate(models.items()):
        rf_model = target_models['random_forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Ensure we don't try to plot more features than we have
        n_to_plot = min(max_features, len(importances))
        
        # Create subplot for this target
        plt.subplot(len(models), 1, i+1)
        plt.title(f'Feature Importance for {target} (Random Forest)')
        plt.bar(range(n_to_plot), importances[indices[:n_to_plot]], align='center')
        plt.xticks(range(n_to_plot), [feature_names[i] for i in indices[:n_to_plot]], rotation=90)
        plt.tight_layout()
    
    plt.savefig('feature_importance_all_targets.png', dpi=300)
    plt.show()
    
    # Also create individual plots for each target
    for target, target_models in models.items():
        rf_model = target_models['random_forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Ensure we don't try to plot more features than we have
        n_to_plot = min(max_features, len(importances))
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance for {target} (Random Forest)')
        plt.bar(range(n_to_plot), importances[indices[:n_to_plot]], align='center')
        plt.xticks(range(n_to_plot), [feature_names[idx] for idx in indices[:n_to_plot]], rotation=90)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{target}.png', dpi=300)
        plt.show()

def save_models(models, scaler, output_dir='models'):
    """
    Save trained models and scaler to disk
    
    Parameters:
    -----------
    models : dict - Dictionary of trained models per target
    scaler : sklearn.preprocessing.StandardScaler - Fitted scaler
    output_dir : str - Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models for each target
    for target, target_models in models.items():
        target_dir = os.path.join(output_dir, target)
        os.makedirs(target_dir, exist_ok=True)
        
        for model_name, model_data in target_models.items():
            model_path = os.path.join(target_dir, f"{model_name}_{timestamp}.joblib")
            joblib.dump(model_data['model'], model_path)
            print(f"Saved {target} {model_name} to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

def main():
    """Main function to run the ML training pipeline"""
    print("Quantum Spin Network - ML Training Pipeline")
    print("==========================================")
    
    # Hardcoded path to your CSV file - replace with your actual file path
    csv_file = "quantum_spin_data_20250401_191820_final.csv"
    
    # Uncomment the following line if you want to be prompted for the file path instead
    # csv_file = input("Enter path to CSV file with quantum simulation data: ")
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        return
    
    # Load and explore data
    df = load_and_explore_data(csv_file)
    
    # Define target features (K-coupling values)
    target_features = ['k01', 'k23', 'k45']
    
    # Prepare data for ML
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_ml(
        df, target_features, test_size=0.2
    )
    
    # Train and evaluate models
    models = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names, target_features
    )
    
    # Visualize feature importance
    visualize_feature_importance(models, feature_names)
    
    # Ask if user wants to save models
    save_model = input("\nSave trained models? (y/n): ").lower() == 'y'
    if save_model:
        save_models(models, scaler)
    
    print("\nML training pipeline complete!")

if __name__ == "__main__":
    main()