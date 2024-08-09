import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# Separate the features and the target
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

# Add a column of ones to X for the intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Calculate the coefficients using the matrix method
B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Make predictions using the calculated coefficients
y_pred = X.dot(B)

# Plot the first 10 actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y[:10], 'o-', color='yellow', label='Actual')
plt.plot(y_pred[:10], 'o-', color='orange', label='Predicted')

plt.title('Actual vs Predicted Values for First 10 Data Points')
plt.xlabel('Data Point Index')
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)
plt.show()
