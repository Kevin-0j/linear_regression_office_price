import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('Nairobi Office Price Ex.csv')

# 2. Extract the relevant columns ('SIZE' and 'PRICE')
size = data['SIZE'].values
price = data['PRICE'].values

# Normalize the data (optional but recommended for gradient descent)
size = (size - np.mean(size)) / np.std(size)
price = (price - np.mean(price)) / np.std(price)

# 3. Initialize random values for slope (m) and intercept (c)
m = np.random.randn()  # Random initial slope
c = np.random.randn()  # Random initial intercept
learning_rate = 0.01  # Step size for Gradient Descent
epochs = 1000  # Increased number of epochs for better convergence

# 4. Define Mean Squared Error (MSE) function
def compute_mse(size, price, m, c):
    n = len(size)
    predictions = m * size + c
    mse = (1 / n) * np.sum((price - predictions) ** 2)
    return mse

# 5. Define Gradient Descent function
def gradient_descent(size, price, m, c, learning_rate):
    n = len(size)
    predictions = m * size + c

    # Compute gradients
    dm = -(2 / n) * np.sum(size * (price - predictions))
    dc = -(2 / n) * np.sum(price - predictions)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    return m, c

# 6. Train the model using Gradient Descent
for epoch in range(epochs):
    m, c = gradient_descent(size, price, m, c, learning_rate)
    mse = compute_mse(size, price, m, c)
    if epoch % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {epoch + 1}, Slope: {m}, Intercept: {c}, MSE: {mse}")

# 7. Make predictions (e.g., predict the price for a 100 sq. ft office)
predicted_price = m * (100 - np.mean(size)) / np.std(size) + c
print(f"Predicted price for 100 sq. ft office: {predicted_price}")

# 8. Plot the data and the line of best fit
plt.scatter(size, price, color='blue', label='Data Points')
plt.plot(size, m * size + c, color='red', label='Best Fit Line')
plt.xlabel('Office Size (normalized)')
plt.ylabel('Office Price (normalized)')
plt.title('Linear Regression: Office Price vs Size')
plt.legend()
plt.show()
