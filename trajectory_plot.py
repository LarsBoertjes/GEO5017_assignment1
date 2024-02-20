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

alpha = np.linalg.inv(D.T @ D) @ D.T @ x
x_predicted = [alpha[0] + alpha[1] * t + alpha[2] * t ** 2 for t in t]
x_residuals = sum((x - x_predicted)**2)

# plot actual vs predicted
plt.figure()
plt.scatter(t, x, label='True X values')
plt.plot(t, x_predicted, label='Predicted X values', linestyle='--')
plt.xlabel('Time')
plt.ylabel('X Position')
plt.title('True vs Predicted X Positions')
plt.legend()
plt.show()

# Print the arrays to the console
print("True X values:", x)
print("Predicted X values:", x_predicted)
print("Residual value for X is: ", x_residuals)


# 2.4 Predict next position