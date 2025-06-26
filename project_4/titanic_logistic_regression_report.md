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
