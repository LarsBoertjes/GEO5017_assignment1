import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PolynomialGradientSolver import gradient_descent
from trajectory_plot import least_squares

df = pd.read_csv('data/positions.csv')

x_pos = np.array(df['X'])
y_pos = np.array(df['Y'])
z_pos = np.array(df['Z'])
t = np.arange(1, len(x_pos) + 1)

fig, axs = plt.subplots(5, 3, figsize=(20, 25))

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

# fit polynomial regression line through velocity plot
# gradient descent x velocity
GDxv = gradient_descent(10000, t_first_derivative, x_velocity, 0.001)
# least squares x velocity
LSxv = least_squares(t_first_derivative, x_velocity)

axs[2, 0].scatter(t_first_derivative, x_velocity, label='Velocity of X', color='blue', marker='o')
axs[2, 0].plot(t_first_derivative, x_velocity, color='lightblue', linestyle='--')
axs[2, 0].plot(t_first_derivative, GDxv[2], color='black', label='gradient prediction')
axs[2, 0].plot(t_first_derivative, LSxv[2], color='grey', label='least squares prediction')
axs[2, 0].set_xlabel(f"GD polynomial: {np.round(GDxv[1][0], 2)} t**2 + {np.round(GDxv[1][1], 2)}t + {np.round(GDxv[1][2], 2)}, residuals: {np.round(GDxv[0], 2)}\n"
                     f"LS polynomial: {np.round(LSxv[1][0], 2)} t**2 + {np.round(LSxv[1][1], 2)}t + {np.round(LSxv[1][2], 2)}, residuals: {np.round(LSxv[0], 2)}")
axs[2, 0].set_ylabel('x_velocity')
axs[2, 0].legend(loc="lower right")

# gradient descent x velocity
GDyv = gradient_descent(10000, t_first_derivative, y_velocity, 0.001)
# least squares y velocity
LSyv = least_squares(t_first_derivative, y_velocity)

axs[2, 1].scatter(t_first_derivative, y_velocity, label='Velocity of Y', color='red', marker='o')
axs[2, 1].plot(t_first_derivative, y_velocity, color='salmon', linestyle='--')
axs[2, 1].plot(t_first_derivative, GDyv[2], color='black', label='gradient prediction')
axs[2, 1].plot(t_first_derivative, LSyv[2], color='grey', label='least squares prediction')
axs[2, 1].set_xlabel(f"GD polynomial: {np.round(GDyv[1][0], 2)} t**2 + {np.round(GDyv[1][1], 2)}t + {np.round(GDyv[1][2], 2)}, residuals: {np.round(GDyv[0], 2)}\n"
                     f"LS polynomial: {np.round(LSyv[1][0], 2)} t**2 + {np.round(LSyv[1][1], 2)}t + {np.round(LSyv[1][2], 2)}, residuals: {np.round(LSyv[0], 2)}")
axs[2, 1].set_ylabel('y_velocity')
axs[2, 1].legend(loc="upper right")

# gradient descent z velocity
GDzv = gradient_descent(10000, t_first_derivative, z_velocity, 0.001)
# least squares y velocity
LSzv = least_squares(t_first_derivative, z_velocity)

axs[2, 2].scatter(t_first_derivative, z_velocity, label='Velocity of Z', color='green', marker='o')
axs[2, 2].plot(t_first_derivative, z_velocity, color='lightgreen', linestyle='--')
axs[2, 2].plot(t_first_derivative, GDzv[2], color='black', label='gradient prediction')
axs[2, 2].plot(t_first_derivative, LSzv[2], color='grey', label='least squares prediction')
axs[2, 2].set_xlabel(f"GD polynomial: {np.round(GDzv[1][0], 2)} t**2 + {np.round(GDzv[1][1], 2)}t + {np.round(GDzv[1][2], 2)}, residuals: {np.round(GDzv[0], 2)}\n"
                     f"LS polynomial: {np.round(LSzv[1][0], 2)} t**2 + {np.round(LSzv[1][1], 2)}t + {np.round(LSzv[1][2], 2)}, residuals: {np.round(LSzv[0], 2)}")
axs[2, 2].set_ylabel('z_velocity')
axs[2, 2].legend(loc="lower right")

# making the acceleration plots
x_acceleration = np.diff(x_velocity) / np.diff(t_first_derivative)
y_acceleration = np.diff(y_velocity) / np.diff(t_first_derivative)
z_acceleration = np.diff(z_velocity) / np.diff(t_first_derivative)
t_second_derivative = t_first_derivative[:-1] + 0.5

axs[3, 0].scatter(t_second_derivative, x_acceleration, label='Acceleration of X', color='blue', marker='o')
axs[3, 0].plot(t_second_derivative, x_acceleration, color='lightblue', linestyle='--')
axs[3, 0].set_xlabel('time')
axs[3, 0].set_ylabel('x_acceleration')

axs[3, 1].scatter(t_second_derivative, y_acceleration, label='Acceleration of Y', color='red', marker='o')
axs[3, 1].plot(t_second_derivative, y_acceleration, color='salmon', linestyle='--')
axs[3, 1].set_xlabel('time')
axs[3, 1].set_ylabel('y_acceleration')

axs[3, 2].scatter(t_second_derivative, z_acceleration, label='Acceleration of Z', color='green', marker='o')
axs[3, 2].plot(t_second_derivative, z_acceleration, color='lightgreen', linestyle='--')
axs[3, 2].set_xlabel('time')
axs[3, 2].set_ylabel('z_acceleration')

# fit polynomial regression line through acceleration plot

# gradient descent x acceleration
GDxa = gradient_descent(10000, t_second_derivative, x_acceleration, 0.001)
# least squares x acceleration
LSxa = least_squares(t_second_derivative, x_acceleration)

axs[4, 0].scatter(t_second_derivative, x_acceleration, label='Acceleration of X', color='blue', marker='o')
axs[4, 0].plot(t_second_derivative, x_acceleration, color='lightblue', linestyle='--')
axs[4, 0].plot(t_second_derivative, GDxa[2], color='black', label='gradient prediction')
axs[4, 0].plot(t_second_derivative, LSxa[2], color='grey', label='least squares prediction')
axs[4, 0].set_xlabel(f"GD polynomial: {np.round(GDxa[1][0], 2)} t**2 + {np.round(GDxa[1][1], 2)}t + {np.round(GDxa[1][2], 2)}, residuals: {np.round(GDxa[0], 2)}\n"
                     f"LS polynomial: {np.round(LSxa[1][0], 2)} t**2 + {np.round(LSxa[1][1], 2)}t + {np.round(LSxa[1][2], 2)}, residuals: {np.round(LSxa[0], 2)}")
axs[4, 0].set_ylabel('x_acceleration')
axs[4, 0].legend(loc="lower right")

# gradient descent y acceleration
GDya = gradient_descent(10000, t_second_derivative, y_acceleration, 0.001)
# least squares y acceleration
LSya = least_squares(t_second_derivative, y_acceleration)

axs[4, 1].scatter(t_second_derivative, y_acceleration, label='Acceleration of Y', color='red', marker='o')
axs[4, 1].plot(t_second_derivative, y_acceleration, color='salmon', linestyle='--')
axs[4, 1].plot(t_second_derivative, GDya[2], color='black', label='gradient prediction')
axs[4, 1].plot(t_second_derivative, LSya[2], color='grey', label='least squares prediction')
axs[4, 1].set_xlabel(f"GD polynomial: {np.round(GDya[1][0], 2)} t**2 + {np.round(GDya[1][1], 2)}t + {np.round(GDya[1][2], 2)}, residuals: {np.round(GDya[0], 2)}\n"
                     f"LS polynomial: {np.round(LSya[1][0], 2)} t**2 + {np.round(LSya[1][1], 2)}t + {np.round(LSya[1][2], 2)}, residuals: {np.round(LSya[0], 2)}")
axs[4, 1].set_ylabel('y_acceleration')
axs[4, 1].legend(loc="lower right")

# gradient descent z acceleration
GDza = gradient_descent(10000, t_second_derivative, z_acceleration, 0.001)
# least squares z acceleration
LSza = least_squares(t_second_derivative, z_acceleration)

axs[4, 2].scatter(t_second_derivative, z_acceleration, label='Acceleration of Z', color='green', marker='o')
axs[4, 2].plot(t_second_derivative, z_acceleration, color='lightgreen', linestyle='--')
axs[4, 2].plot(t_second_derivative, GDza[2], color='black', label='gradient prediction')
axs[4, 2].plot(t_second_derivative, LSza[2], color='grey', label='least squares prediction')
axs[4, 2].set_xlabel(f"GD polynomial: {np.round(GDza[1][0], 2)} t**2 + {np.round(GDza[1][1], 2)}t + {np.round(GDza[1][2], 2)}, residuals: {np.round(GDza[0], 2)}\n"
                     f"LS polynomial: {np.round(LSza[1][0], 2)} t**2 + {np.round(LSza[1][1], 2)}t + {np.round(LSza[1][2], 2)}, residuals: {np.round(LSza[0], 2)}")
axs[4, 2].set_ylabel('z_acceleration')
axs[4, 2].legend(loc="lower right")









plt.tight_layout()
plt.show()
