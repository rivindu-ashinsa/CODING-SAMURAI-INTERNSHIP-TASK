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
