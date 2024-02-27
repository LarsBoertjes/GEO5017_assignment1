import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from leastsquares import least_squares


df = pd.read_csv('data/positions.csv')

x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])
t = np.arange(0, len(x))

learning_rate = 0.0000001
iterations = 1000000

print(y)
print(t)

k = 4

def gradient_solver(y, t, learning_rate, max_iter, tolerance, k):
    coefficients = np.zeros(k + 1)

    for _ in range(max_iter):
        y_pred = sum([coefficients[i] * (t ** i) for i in range(k + 1)])
        error = y - y_pred
        gradients = np.array([-2 * sum(error * (t ** i)) for i in range(k + 1)])

        step_size = learning_rate * gradients
        if sum(step_size) < tolerance:
            break

        coefficients -= step_size

    return tuple(coefficients)

coefficients = gradient_solver(y, t, learning_rate, iterations, 0.0001, k)
coefficients_ls = least_squares(t, y, 4)[1]

print(coefficients)
print(coefficients_ls)

plt.scatter(t, y, color='red', label='Data Points')

# Plot the regression line
t_line = np.linspace(min(t), max(t), 500)
y_line = sum([coefficients[i] * (t_line ** i) for i in range(k + 1)])
y_ls = sum([coefficients_ls[i] * (t_line ** i) for i in range(k + 1)])
plt.plot(t_line, y_line, color='blue', label='Regression Line')
plt.plot(t_line, y_ls, color='green', label='least squares')

plt.xlabel('t')
plt.ylabel('y')
plt.title('Scatter Plot of t vs y and Regression Line')
plt.legend()
plt.show()

