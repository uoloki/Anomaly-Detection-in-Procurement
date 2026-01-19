import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import utils

def run_eda(data_path):
    utils.setup_plotting_style()
    df = utils.load_data(data_path)
    
    # We assume data_path is the processed one, but EDA usually runs on RAW data to show distributions of everything.
    # However, the plan says "Refactor eda_workflow.py to generate and save key charts".
    # Let's check if we should use raw or processed. EDA typically explores the whole dataset.
    # But if we use processed, we miss the dropped columns. 
    # Let's stick to the raw data for EDA as is standard, or use the processed if we want to visualize features only.
    # Given the previous context, let's use the processed file to verify feature engineering, 
    # OR the original file to replicate `eda_workflow.py`. 
    # Let's use the original file to be safe and comprehensive, as EDA usually precedes processing.
    
    # Correcting: The plan says "Extract cleaning... to process_data.py" and "Refactor eda... to save charts".
    # So EDA script should likely work on the raw data to show the "before" picture or the cleaned features.
    # Let's use the raw data in data/GTI_labelled_cartel_data_NOV2023.csv
    
    target_col = 'is_cartel'
    
    # 1. Target Distribution
    fig1 = plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=target_col, data=df, palette='coolwarm')
    plt.title("Class Balance (Is Cartel?)")
    utils.save_plot(fig1, "class_balance.png")
    
    # 2. Missing Values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)
    
    if not missing_df.empty:
        fig2 = plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_df.index, y=missing_df['Missing %'], palette='viridis')
        plt.title("Percentage of Missing Values by Feature")
        plt.xticks(rotation=45, ha='right')
        utils.save_plot(fig2, "missing_values.png")

    # 3. Numeric Distributions (Top 4 interesting ones)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col, errors='ignore')
    # Pick a few key ones mentioned in analysis
    key_cols = ['lot_bidscount', 'buyer_avg_bidder_yearly', 'benfords_market_yearly_avg', 'singleb_avg']
    key_cols = [c for c in key_cols if c in df.columns]
    
    if key_cols:
        fig3, axes = plt.subplots(len(key_cols), 2, figsize=(14, 4 * len(key_cols)))
        for i, col in enumerate(key_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0], color='skyblue')
            axes[i, 0].set_title(f"Distribution of {col}")
            sns.boxplot(x=df[target_col], y=df[col], ax=axes[i, 1], palette='Set2')
            axes[i, 1].set_title(f"{col} by Class")
        plt.tight_layout()
        utils.save_plot(fig3, "key_numeric_features.png")

    # 4. Correlation Heatmap
    if len(numeric_cols) > 0:
        corr_matrix = df[list(numeric_cols) + [target_col]].corr()
        fig4 = plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title("Correlation Heatmap")
        utils.save_plot(fig4, "correlation_heatmap.png")

    # 5. PCA
    print("Running PCA...")
    pca_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2))
    ])
    
    # Use only numeric columns and drop rows with all NaNs if any, or let imputer handle
    X_pca = df[numeric_cols]
    try:
        X_pca_trans = pca_pipeline.fit_transform(X_pca)
        pca_df = pd.DataFrame(data=X_pca_trans, columns=['PC1', 'PC2'])
        pca_df['Target'] = df[target_col].values
        
        fig5 = plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, alpha=0.6, palette='coolwarm')
        plt.title(f"PCA (2 Components) - Variance Explained: {np.sum(pca_pipeline['pca'].explained_variance_ratio_):.2%}")
        utils.save_plot(fig5, "pca_projection.png")
    except Exception as e:
        print(f"PCA failed: {e}")

if __name__ == "__main__":
    # We use the RAW data for EDA to show the initial state
    run_eda("data/GTI_labelled_cartel_data_NOV2023.csv")


