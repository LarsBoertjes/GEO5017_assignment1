import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/positions.csv')

x = df['X']
y = df['Y']
z = df['Z']

# 2.1 Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Trajectory Plot')
plt.show()

# 2.2 Drone speed
t = np.arange(1, len(x) + 1)
x = np.array(x)
y = np.array(y)
z = np.array(z)


def least_squares(input_values, output_values):
    # least squares function for polynomial with k = 2
    D = np.column_stack((np.ones(len(input_values)), input_values, input_values**2))

    coeffs = np.linalg.inv(D.T @ D) @ D.T @ output_values
    predicted_values = [coeffs[0] + coeffs[1] * value + coeffs[2] * value ** 2 for value in input_values]
    residuals = sum((output_values - predicted_values)**2)

    return (residuals, coeffs, predicted_values)


x_predicted = least_squares(t, x)
print(f"test: {x_predicted[1]} , residuals: {x_predicted[2]}, new coefficients: {x_predicted[0]}")

y_predicted = least_squares(t, y)

z_predicted = least_squares(t, z)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# plot actual vs predicted
axs[0].scatter(t, x, label='True X values', color='blue', marker='o')
axs[0].plot(t, x_predicted[2], label='Predicted X values', linestyle='--', color='red')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X Position')
axs[0].set_title('True vs Predicted X Positions')
axs[0].legend()

# Plot for y
axs[1].scatter(t, y, label='True Y values', color='green', marker='o')
axs[1].plot(t, y_predicted[2], label='Predicted Y values', linestyle='--', color='orange')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Y Position')
axs[1].set_title('True vs Predicted Y Positions')
axs[1].legend()

# Plot for z
axs[2].scatter(t, z, label='True Z values', color='purple', marker='o')
axs[2].plot(t, z_predicted[2], label='Predicted Z values', linestyle='--', color='brown')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Z Position')
axs[2].set_title('True vs Predicted Z Positions')
axs[2].legend()

plt.tight_layout()
plt.show()

# Print the arrays to the console


# 2.4 Predict next position

# speed test
matrix = df.to_numpy()
print(matrix)
time = range(len(matrix))
print(time)

speed_array = np.zeros(matrix.shape)
for i in time:
    if i - 1 >= 0:
        speed_array[i] = matrix[i] - matrix[i - 1]

print(speed_array)
speed = np.linalg.norm(speed_array, axis=1)
print(speed)

plt.plot(time, speed)
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Speed over Time')
plt.show()


