import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# will get warning if you don't have pyarrow on your system / venv

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

k = 2

D = np.column_stack((np.ones(len(t)), t, t**k))

coef_x = np.linalg.inv(D.T @ D) @ D.T @ x
x_predicted = [coef_x[0] + coef_x[1] * t + coef_x[2] * t ** 2 for t in t]
x_residuals = sum((x - x_predicted)**2)

coef_y = np.linalg.inv(D.T @ D) @ D.T @ y
y_predicted = [coef_y[0] + coef_y[1] * t + coef_y[2] * t ** 2 for t in t]
y_residuals = sum((y - y_predicted)**2)

coef_z = np.linalg.inv(D.T @ D) @ D.T @ z
z_predicted = [coef_z[0] + coef_z[1] * t + coef_z[2] * t ** 2 for t in t]
z_residuals = sum((z - z_predicted)**2)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# plot actual vs predicted
axs[0].scatter(t, x, label='True X values', color='blue', marker='o')
axs[0].plot(t, x_predicted, label='Predicted X values', linestyle='--', color='red')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X Position')
axs[0].set_title('True vs Predicted X Positions')
axs[0].legend()

# Plot for y
axs[1].scatter(t, y, label='True Y values', color='green', marker='o')
axs[1].plot(t, y_predicted, label='Predicted Y values', linestyle='--', color='orange')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Y Position')
axs[1].set_title('True vs Predicted Y Positions')
axs[1].legend()

# Plot for z
axs[2].scatter(t, z, label='True Z values', color='purple', marker='o')
axs[2].plot(t, z_predicted, label='Predicted Z values', linestyle='--', color='brown')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Z Position')
axs[2].set_title('True vs Predicted Z Positions')
axs[2].legend()

plt.tight_layout()
plt.show()

# Print the arrays to the console
print("Residual value for X is: ", x_residuals)
print("Residual value for Y is: ", y_residuals)
print("Residual value for Z is: ", z_residuals)


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


