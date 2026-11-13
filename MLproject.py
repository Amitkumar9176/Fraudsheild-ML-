
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



# 1. DATASET LOADING
df = pd.read_csv("C:/Users/HP/Downloads/creditcard.csv")
print("\nFIRST FIVE ROWS")
print(df.head())
print("\nDATASET SHAPE")
print(df.shape)
print("\nDATASET INFORMATION")
print(df.info())

#2.DATA PREPROCESSING
print(df.isnull().sum().sum())
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print(df.shape)


print(df["Class"].value_counts())
print("\nCLASS PERCENTAGE")
print(df["Class"].value_counts(normalize=True) * 100)

# 3.EXPLORATORY DATA ANALYSIS

# Fraud vs Genuine transactions
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Class")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# Transaction amount distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.show()

# Fraud and genuine amount comparison
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Class", y="Amount")
plt.title("Fraud vs Genuine Transaction Amount")
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#4. FEATURE SELECTION
X = df.drop("Class", axis=1)
y = df["Class"]

# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTRAINING DATA SHAPE")
print(X_train.shape)
print("\nTESTING DATA SHAPE")
print(X_test.shape)

# 6. FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. LOGISTIC REGRESSION
print("\n========== LOGISTIC REGRESSION ==========")
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_lr, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_lr, zero_division=0))


print("\nClassification Report")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# 8. K-NEAREST NEIGHBORS (KNN)
print("\n========== K NEAREST NEIGHBORS ==========")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_knn, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_knn, zero_division=0))
print("\nClassification Report")
print(classification_report(y_test, y_pred_knn, zero_division=0))

# 9. DECISION TREE
print("\n========== DECISION TREE ==========")
dt = DecisionTreeClassifier(
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_dt, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_dt, zero_division=0))

print("\nClassification Report")
print(classification_report(y_test, y_pred_dt, zero_division=0))

# 10.RANDOM FOREST
print("\n========== RANDOM FOREST ==========")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_rf, zero_division=0))
print("\nClassification Report")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# 11. MODEL ACCURACY COMPARISON
lr_accuracy = accuracy_score(y_test, y_pred_lr)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n========== MODEL COMPARISON ==========")
print("Logistic Regression:", lr_accuracy * 100, "%")
print("KNN:", knn_accuracy * 100, "%")
print("Decision Tree:", dt_accuracy * 100, "%")
print("Random Forest:", rf_accuracy * 100, "%")

# Accuracy graph
models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
accuracy = [lr_accuracy, knn_accuracy, dt_accuracy, rf_accuracy]
plt.figure(figsize=(9, 5))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=25)
plt.show()

# 12. CONFUSION MATRIX - RANDOM FOREST
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# 13. FINAL RESULT
print("CREDIT CARD FRAUD DETECTION COMPLETED")
print("\nTOTAL TRANSACTIONS:", len(df))
print("\nFINAL MODEL ACCURACY")
print("Logistic Regression:", round(lr_accuracy * 100, 2), "%")
print("KNN:", round(knn_accuracy * 100, 2), "%")
print("Decision Tree:", round(dt_accuracy * 100, 2), "%")
print("Random Forest:", round(rf_accuracy * 100, 2), "%")
print("\nPROJECT EXECUTED SUCCESSFULLY!")
