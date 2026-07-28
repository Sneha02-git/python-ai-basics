#import all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from IPython.display import display

try:
    import ipywidgets as widgets
    from ipywidgets import interact
    widgets_available = True
except ImportError:
    widgets_available = False

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")
# %matplotlib inline

print("Setup complete. You can start the experiment.")

# Basic Python data structures
numbers = [1, 2, 3, 4, 5]

squared_numbers = [value ** 2 for value in numbers]
cubed_numbers = [value ** 3 for value in numbers]

display(pd.DataFrame({
    "number": numbers,
    "square": squared_numbers,
    "cube": cubed_numbers,
}))

# Basic Python data structures (dictionaries)
student_profile = {
    "name": "Amina",
    "semester": 7,
    "cgpa": 8.6,
    "interests": ["AI", "ML", "Data Science"],
}

print("Dictionary example:")
display(student_profile)
print("Try changing the numbers list and run this cell again.")

# NumPy array operations
vector = np.array(numbers)

multiplier = 2
bias = 1
scaled_vector = vector * multiplier + bias

print("Original vector:", vector)
print("Scaled vector:", scaled_vector)

matrix = np.arange(1, 10).reshape(3, 3)
print("\nMatrix:")
display(pd.DataFrame(matrix))

print("Row means:", matrix.mean(axis=1))

print("\nBroadcast addition:")
display(pd.DataFrame(matrix + np.array([10, 20, 30])))

# Load Iris dataset
iris = load_iris(as_frame=True)
iris_df = iris.frame.copy()
iris_df["species"] = iris_df["target"].map(dict(enumerate(iris.target_names)))
iris_df = iris_df.drop(columns=["target"])

print("Dataset loaded successfully.")
print("Shape:", iris_df.shape)
print("Columns:", list(iris_df.columns))
# display(iris_df.head())

if widgets_available:
    def show_rows(n=5):
        display(iris_df.head(n))

    interact(show_rows, n=widgets.IntSlider(value=5, min=1, max=15, step=1))

# Missing values and summary statistics
missing_values = iris_df.isna().sum()

print("Missing values per column:")
display(missing_values.to_frame("missing_count"))

print("Descriptive statistics:")
display(iris_df.describe(include="all"))

# Class-wise feature averages
feature_means = iris_df.groupby("species").mean(numeric_only=True)
display(feature_means.round(3))

feature_gap = feature_means.max() - feature_means.min()
print("Feature with largest class-wise average difference:", feature_gap.idxmax())

if widgets_available:
    def compare_feature(feature):
        display(feature_means[[feature]].sort_values(feature, ascending=False).round(3))

    interact(compare_feature, feature=list(feature_means.columns))

# Feature distribution plots
numeric_columns = iris.feature_names

if widgets_available:
    def plot_histogram(feature=numeric_columns[0], bins=15):
        plt.figure(figsize=(7, 4))
        plt.hist(iris_df[feature], bins=bins, color="#2a9d8f", edgecolor="black")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    interact(
        plot_histogram,
        feature=numeric_columns,
        bins=widgets.IntSlider(value=15, min=5, max=30, step=1),
    )
else:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, column in zip(axes.ravel(), numeric_columns):
        axis.hist(iris_df[column], bins=15, color="#2a9d8f", edgecolor="black")
        axis.set_title(column)
    plt.tight_layout()
    plt.show()

# Scatter plot for class separation
color_map = {"setosa": "#e76f51", "versicolor": "#2a9d8f", "virginica": "#264653"}

def scatter_features(x_feature="petal length (cm)", y_feature="petal width (cm)"):
    plt.figure(figsize=(7, 5))
    for species, group in iris_df.groupby("species"):
        plt.scatter(
            group[x_feature],
            group[y_feature],
            label=species,
            alpha=0.8,
            color=color_map[species],
        )
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{x_feature} vs {y_feature}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if widgets_available:
    interact(scatter_features, x_feature=numeric_columns, y_feature=numeric_columns)
else:
    scatter_features()

# Correlation heatmap
correlation_matrix = iris_df.drop(columns=["species"]).corr()

plt.figure(figsize=(7, 5))
plt.imshow(correlation_matrix, cmap="viridis", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)

for row in range(len(correlation_matrix.index)):
    for col in range(len(correlation_matrix.columns)):
        plt.text(col, row, f"{correlation_matrix.iloc[row, col]:.2f}", ha="center", va="center", color="white")

plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.show()

# Final summary table
summary_table = pd.DataFrame({
    "mean": iris_df.drop(columns=["species"]).mean(),
    "std": iris_df.drop(columns=["species"]).std(),
    "min": iris_df.drop(columns=["species"]).min(),
    "max": iris_df.drop(columns=["species"]).max(),
})

display(summary_table.round(3))

