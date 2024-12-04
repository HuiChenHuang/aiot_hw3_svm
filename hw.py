import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a simple 1D dataset
X, y = make_classification(
    n_samples=200, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression and SVM models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate models
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
print(f"SVM Accuracy: {accuracy_svm:.2f}")

# Plot decision boundaries
x_range = np.linspace(X.min() - 1, X.max() + 1, 500).reshape(-1, 1)
y_prob_log_reg = log_reg.predict_proba(x_range)[:, 1]
y_decision_svm = svm.decision_function(x_range)

plt.figure(figsize=(10, 6))

# Logistic Regression decision boundary
plt.plot(x_range, y_prob_log_reg, label='Logistic Regression Probability', color='blue')

# SVM decision boundary
plt.plot(x_range, y_decision_svm, label='SVM Decision Function', color='red')

# Data points
plt.scatter(X_train, y_train, color='green', label='Train Data', alpha=0.6, edgecolor='k')
plt.scatter(X_test, y_test, color='orange', label='Test Data', alpha=0.6, edgecolor='k')

plt.axhline(0.5, color='gray', linestyle='--', label='Decision Threshold (Logistic Regression)')
plt.axhline(0, color='black', linestyle=':', label='Decision Boundary (SVM)')

plt.title('Logistic Regression vs SVM on 1D Dataset')
plt.xlabel('Feature Value')
plt.ylabel('Probability / Decision Function Value')
plt.legend()
plt.show()

