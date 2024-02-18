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
positions = np.column_stack((x, y, z))
time_intervals = np.arange(1, 7)

final_position = positions[-1]
initial_position = positions[0]

# -- velocity vector for each axis
velocity_x = (final_position[0] - initial_position[0]) / (time_intervals[-1] - time_intervals[0])
velocity_y = (final_position[1] - initial_position[1]) / (time_intervals[-1] - time_intervals[0])
velocity_z = (final_position[2] - initial_position[2]) / (time_intervals[-1] - time_intervals[0])

# -- take the norm of the combined velocities
velocity_vector = np.array([velocity_x, velocity_y, velocity_z])

speed = np.linalg.norm(velocity_vector)

print("Speed of drone is:", np.round(speed, 2), "units per second")

# -- to calculate residuals make a prediction for the position based on calculated velocity
predicted_positions = np.array([initial_position + velocity_vector * t for t in time_intervals])

# -- find differences between predicted and actual positions
differences = predicted_positions - positions

# -- square and sum them to find the residual error
residual_error = np.sum(differences ** 2)

print("Residual error:", np.round(residual_error, 2))

# 2.3 Drone acceleration
# -- With constant acceleration, speed is linear, position is polynomial. Thus fit for use with Polynomial Regression
# -- I think for now it is ok to chose k = 2, so y = a0 + a1*x + a2*x**2
# -- a0 initial position
# -- a1 initial velocity
# -- a2 the acceleration

# -- next determine coefficients a0, a1, a2 using polynomial regression

# -- fit a polynomial for each axis (x, y, z)

# -- calculate residual error by summing residual error for each axis

# -- compare with error from 2.2

# 2.4 Predict next position