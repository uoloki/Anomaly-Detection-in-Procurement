import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data Loaded Successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise

def setup_plotting_style():
    """
    Sets up the default plotting style for the project.
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (12, 6)
    pd.set_option("display.max_columns", None)

def save_plot(fig, filename, output_dir="output/charts"):
    """
    Saves a matplotlib figure to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {path}")


