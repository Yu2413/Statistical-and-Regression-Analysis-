import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(50) * 10  # Feature (independent variable)
y = 2 * X + 1 + np.random.randn(50) * 2  # Target (dependent variable) with some noise

# Reshape X to 2D array (required by scikit-learn)
X = X.reshape(-1, 1)

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values using the model
y_pred = model.predict(X)

# Plot the data and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data Points')  # Scatter plot of the data
plt.plot(X, y_pred, color='red', label='Regression Line')  # Regression line
plt.title('Linear Regression Example', fontsize=16)
plt.xlabel('X (Independent Variable)', fontsize=14)
plt.ylabel('y (Dependent Variable)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()