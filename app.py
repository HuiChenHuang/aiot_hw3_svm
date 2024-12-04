import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# Streamlit App
st.title("2D SVM with 3D Visualization")
st.write("This application demonstrates a 2D SVM classifier with a circular dataset and 3D visualization.")

# Generate a circular dataset
X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM with RBF kernel
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# Create a 3D grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
xx, yy = np.meshgrid(x_range, y_range)
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict decision function values
z = svm.decision_function(grid).reshape(xx.shape)

# Plot the results in 3D
fig = go.Figure()

# Add decision surface
fig.add_trace(go.Surface(z=z, x=xx, y=yy, colorscale='Viridis', opacity=0.8))

# Add training points
fig.add_trace(go.Scatter3d(
    x=X_train[:, 0], y=X_train[:, 1], z=svm.decision_function(X_train),
    mode='markers', marker=dict(size=5, color=y_train, colorscale='Viridis', line_width=1),
    name="Training Data"
))

# Add test points
fig.add_trace(go.Scatter3d(
    x=X_test[:, 0], y=X_test[:, 1], z=svm.decision_function(X_test),
    mode='markers', marker=dict(size=5, color=y_test, colorscale='Plasma', line_width=1),
    name="Test Data"
))

# Customize the layout
fig.update_layout(
    title="SVM Decision Surface",
    scene=dict(
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        zaxis_title="Decision Function Value"
    )
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Model accuracy
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

st.write(f"Training Accuracy: {train_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")
