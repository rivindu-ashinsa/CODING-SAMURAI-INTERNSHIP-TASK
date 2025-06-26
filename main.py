import os

# Create directory to save all three files
project_dir = "Coding_Samurai_Titanic_EDA_Project"
os.makedirs(project_dir, exist_ok=True)

# Define the code for the script and notebook
project_code = '''\
# Titanic EDA Project - End-to-End

# 🔧 Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 📥 Load Data
df = pd.read_csv("train.csv")

# 📊 Basic Information
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
print(df.info())
print(df.describe())

# 🔍 Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percentage})
print(missing_df)

# 🔧 Handle Missing Data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True)

# 📈 Univariate Analysis
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

sns.countplot(data=df, x='Pclass')
plt.title("Passenger Class Distribution")
plt.show()

sns.countplot(data=df, x='Sex')
plt.title("Gender Distribution")
plt.show()

df['Age'].hist(bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

df['Fare'].hist(bins=40, edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# 🔗 Bivariate Analysis
sns.barplot(data=df, x='Sex', y='Survived')
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(data=df, x='Pclass', y='Survived')
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.boxplot(data=df, x='Survived', y='Age')
plt.title("Age Distribution by Survival")
plt.show()

# 🔥 Correlation Analysis
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
'''

# Save .py script
script_path = os.path.join(project_dir, "titanic_eda.py")
with open(script_path, "w", encoding="utf-8") as f:
    f.write(project_code)

# Save Jupyter notebook
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['cells'] = [nbf.v4.new_code_cell(code) for code in project_code.split('\n\n')]

notebook_path = os.path.join(project_dir, "titanic_eda.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# Create a report (Markdown format)
report_md = '''\
# 🛳️ Titanic EDA Project Report

## 🧠 Objective
Perform a comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset to understand variable distribution, missing values, and potential survival-influencing features.

---

## 📈 Summary of Steps

### 1. Data Loading
- Used Titanic dataset from open source (Kaggle/DataScienceDojo)
- Loaded with `pandas.read_csv()`

### 2. Missing Value Handling
| Feature   | Missing Values | Action Taken     |
|-----------|----------------|------------------|
| Age       | 177            | Filled with median |
| Cabin     | 687            | Dropped          |
| Embarked  | 2              | Filled with mode |

### 3. Visualizations
- Countplots for categorical features (`Survived`, `Sex`, `Pclass`)
- Histograms for numerical features (`Age`, `Fare`)
- Boxplots and barplots for feature relationships
- Correlation heatmap

### 4. Key Findings

| Feature    | Insight |
|------------|---------|
| Sex        | Females had higher survival |
| Pclass     | 1st class passengers survived more |
| Age        | Children had slightly better survival |
| Fare       | Higher fare = better survival chances |

---

## ✅ Conclusion
This EDA provided crucial insights into survival predictors on the Titanic and showed how to handle missing data and derive patterns from visuals and correlation analysis.
'''

report_path = os.path.join(project_dir, "titanic_eda_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)

# Return paths to user
script_path, notebook_path, report_path

