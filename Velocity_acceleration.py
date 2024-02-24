import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# will get warning if you don't have pyarrow on your system / venv

df = pd.read_csv('data/positions.csv')

x_pos = np.array(df['X'])
y_pos = np.array(df['Y'])
z_pos = np.array(df['Z'])
t = np.arange(1, len(x_pos) + 1)

fig, axs = plt.subplots(3, 3, figsize=(20, 15))

# make sure al x-axis are the same
for ax_row in axs:
    for ax in ax_row:
        ax.set_xlim(0.8, 6.2)
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)


# making the position plots
axs[0, 0].scatter(t, x_pos, label='Position of X', color='blue', marker='o')
axs[0, 0].plot(t, x_pos, color='lightblue', linestyle='--')
axs[0, 0].set_xlabel('time')
axs[0, 0].set_ylabel('x_position')

axs[0, 1].scatter(t, y_pos, label='Position of Y', color='red', marker='o')
axs[0, 1].plot(t, y_pos, color='salmon', linestyle='--')
axs[0, 1].set_xlabel('time')
axs[0, 1].set_ylabel('y_position')

axs[0, 2].scatter(t, z_pos, label='Position of Z', color='green', marker='o')
axs[0, 2].plot(t, z_pos, color='lightgreen', linestyle='--')
axs[0, 2].set_xlabel('time')
axs[0, 2].set_ylabel('z_position')

# making the velocity plots
x_velocity = np.diff(x_pos) / np.diff(t)
y_velocity = np.diff(y_pos) / np.diff(t)
z_velocity = np.diff(z_pos) / np.diff(t)
t_first_derivative = t[:-1] + 0.5

axs[1, 0].scatter(t_first_derivative, x_velocity, label='Velocity of X', color='blue', marker='o')
axs[1, 0].plot(t_first_derivative, x_velocity, color='lightblue', linestyle='--')
axs[1, 0].set_xlabel('time')
axs[1, 0].set_ylabel('x_velocity')

axs[1, 1].scatter(t_first_derivative, y_velocity, label='Velocity of Y', color='red', marker='o')
axs[1, 1].plot(t_first_derivative, y_velocity, color='salmon', linestyle='--')
axs[1, 1].set_xlabel('time')
axs[1, 1].set_ylabel('y_velocity')

axs[1, 2].scatter(t_first_derivative, z_velocity, label='Velocity of Z', color='green', marker='o')
axs[1, 2].plot(t_first_derivative, z_velocity, color='lightgreen', linestyle='--')
axs[1, 2].set_xlabel('time')
axs[1, 2].set_ylabel('z_velocity')

# making the acceleration plots
x_acceleration = np.diff(x_velocity) / np.diff(t_first_derivative)
y_acceleration = np.diff(y_velocity) / np.diff(t_first_derivative)
z_acceleration = np.diff(z_velocity) / np.diff(t_first_derivative)
t_second_derivative = t_first_derivative[:-1] + 0.5

axs[2, 0].scatter(t_second_derivative, x_acceleration, label='Acceleration of X', color='blue', marker='o')
axs[2, 0].plot(t_second_derivative, x_acceleration, color='lightblue', linestyle='--')
axs[2, 0].set_xlabel('time')
axs[2, 0].set_ylabel('x_acceleration')

axs[2, 1].scatter(t_second_derivative, y_acceleration, label='Acceleration of Y', color='red', marker='o')
axs[2, 1].plot(t_second_derivative, y_acceleration, color='salmon', linestyle='--')
axs[2, 1].set_xlabel('time')
axs[2, 1].set_ylabel('y_acceleration')

axs[2, 2].scatter(t_second_derivative, z_acceleration, label='Acceleration of Z', color='green', marker='o')
axs[2, 2].plot(t_second_derivative, z_acceleration, color='lightgreen', linestyle='--')
axs[2, 2].set_xlabel('time')
axs[2, 2].set_ylabel('z_acceleration')
axs[2, 2].axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
