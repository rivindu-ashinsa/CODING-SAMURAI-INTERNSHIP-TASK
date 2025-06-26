# Re-import necessary libraries after code execution environment reset
import os
import nbformat as nbf

# Recreate project directory
project_dir = "project_4"
os.makedirs(project_dir, exist_ok=True)

# Define Logistic Regression code
logistic_code = '''\
# Titanic Project - Logistic Regression

# 🔧 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 📥 Load Data
df = pd.read_csv("train.csv")

# 🔍 Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# 🎯 Encode Categorical Variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# 🎯 Feature Selection and Target Variable
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔃 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🚀 Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 📈 Predictions and Evaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))
'''

# Save .py script
logistic_script_path = os.path.join(project_dir, "titanic_logistic_regression.py")
with open(logistic_script_path, "w", encoding="utf-8") as f:
    f.write(logistic_code)

# Save Jupyter notebook
logistic_nb = nbf.v4.new_notebook()
logistic_nb['cells'] = [nbf.v4.new_code_cell(code) for code in logistic_code.split('\n\n')]
logistic_notebook_path = os.path.join(project_dir, "titanic_logistic_regression.ipynb")
with open(logistic_notebook_path, "w", encoding="utf-8") as f:
    nbf.write(logistic_nb, f)

# Save report markdown
logistic_report_md = '''\
# 🧪 Logistic Regression on Titanic Dataset

## 🎯 Objective
Build a logistic regression model to predict passenger survival based on features like age, sex, and passenger class.

---

## 🛠️ Preprocessing Steps

1. **Handle Missing Values**
   - `Age`: Filled with median
   - `Embarked`: Filled with mode
   - `Cabin`: Dropped

2. **Drop Unnecessary Columns**: `PassengerId`, `Ticket`, `Name`

3. **Label Encoding**: Categorical variables like `Sex` and `Embarked`

4. **Feature Scaling**: Standardized numerical features using `StandardScaler`

---

## 📊 Model Building

- Model: Logistic Regression (from Scikit-learn)
- Target: `Survived`
- Features: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`
- Train/Test Split: 80/20

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## ✅ Conclusion

The logistic regression model provides a simple yet effective baseline for classifying survival. Feature importance can be interpreted using model coefficients.
'''

logistic_report_path = os.path.join(project_dir, "titanic_logistic_regression_report.md")
with open(logistic_report_path, "w", encoding="utf-8") as f:
    f.write(logistic_report_md)

logistic_script_path, logistic_notebook_path, logistic_report_path

