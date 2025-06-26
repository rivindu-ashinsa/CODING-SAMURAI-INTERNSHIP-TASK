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
