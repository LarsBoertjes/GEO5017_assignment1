import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

df = pd.read_csv('data/positions.csv')
x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])

t = np.arange(1, len(x) + 1)

xs = []
ys = []


def eval_2nd_degree(coeffs, x):
    a = (coeffs[0]*(x*x))
    b = (coeffs[1]*x)
    c = coeffs[2]
    return a + b + c


def loss_mse(true_vals, predicted_vals):
    return sum((true_vals - predicted_vals) * (true_vals - predicted_vals)) / len(true_vals)


def compute_gradient(coeffs, input_vals, output_vals, learning_rate):
    a_s = []
    b_s = []
    c_s = []

    predicted_vals = eval_2nd_degree(coeffs, input_vals)

    for x,y,predicted_val in list(zip(input_vals, output_vals, predicted_vals)):
        partial_a = (x ** 2) * (y - predicted_val)
        a_s.append(partial_a)
        partial_b = x * (y - predicted_val)
        b_s.append(partial_b)
        partial_c = (y - predicted_val)
        c_s.append(partial_c)

    num = [i for i in predicted_vals]
    n = len(num)

    gradient_a = (-2 / n) * sum(a_s)
    gradient_b = (-2 / n) * sum(b_s)
    gradient_c = (-2 / n) * sum(c_s)

    a_new = coeffs[0] - learning_rate * gradient_a
    b_new = coeffs[1] - learning_rate * gradient_b
    c_new = coeffs[2] - learning_rate * gradient_c

    new_model_coeffs = (a_new, b_new, c_new)

    new_predicted_values = eval_2nd_degree(new_model_coeffs, input_vals)

    updated_model_loss = loss_mse(output_vals, new_predicted_values)

    return updated_model_loss, new_model_coeffs, new_predicted_values


def gradient_descent(iterations, input_vals, output_vals, learning_rate):
    losses = []
    rand_coeffs_to_test = (random.randrange(-10, 10), random.randrange(-10, 10), random.randrange(-10, 10))

    for i in range(iterations):
        loss = compute_gradient(rand_coeffs_to_test, input_vals, output_vals, learning_rate)
        rand_coeffs_to_test = loss[1]
        losses.append(loss[0])

    print(losses)

    return loss[0], loss[1], loss[2], losses


rand_coeffs = (random.randrange(-5, 5), random.randrange(-5,5), random.randrange(-5,5))
GD = gradient_descent(10000, t, x, 0.001)

plt.figure(figsize=(20, 10))
plt.plot(t, x, 'g+', label='original')
plt.plot(t, GD[2], label='final prediction')
plt.title('Original data points vs Final prediction after Gradient Descent')
plt.legend(loc="lower right")
plt.show()