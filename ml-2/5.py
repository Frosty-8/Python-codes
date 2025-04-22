import pandas as pd
import seaborn as sns
data = sns.load_dataset('iris')

# Load dataset
data = sns.load_dataset('iris')

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Handle missing values ONLY for numeric columns
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Print dataset info and description
print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize distributions and relationships
sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.show()

# Correlation Heatmap (select only numeric columns)
numeric_data = data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

from ydata_profiling import ProfileReport

profile = ProfileReport(data, title="Iris Dataset EDA Report", explorative=True)
profile.to_file("Iris_EDA_Report.html")

# jupyter nbconvert --to html 5.ipynb
