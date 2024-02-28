import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def predict_pos_speed(a0, a1, t):
    return a0 + a1 * t


def predict_pos_acc(a0, a1, a2, t):
    return a0 + a1 * t + a2 * t**2


def plot_trajectory(x, y, z):
    """
    Plot the trajectory based on given arrays.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    ax.view_init(elev=30, azim=50)
    ax.plot(x, y, z)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Trajectory Plot')

    plt.show()


def plot_positions_with_constant_speed(t, x, y, z, p):
    """
    Plot the measured and predicted positions of X, Y, and Z when constant speed is assumed.
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Position of X, Y, and Z')

    for ax in axs:
        ax.set_xlim(-0.1, 6.1)
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

    axs[0].scatter(t, x, label='Position of X', color='green', marker='o')
    axs[0].plot(t, predict_pos_speed(p[0], p[1], t), color='red', label='Gradient Descent')
    axs[0].plot(t, predict_pos_speed(p[6], p[7], t), color='blue', label='Linear Regression')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('x_position')
    axs[0].legend()

    axs[1].scatter(t, y, label='Position of Y', color='green', marker='o')
    axs[1].plot(t, predict_pos_speed(p[2], p[3], t), color='red', label='Gradient Descent')
    axs[1].plot(t, predict_pos_speed(p[8], p[9], t), color='blue', label='Linear Regression')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('y_position')
    axs[1].legend()

    axs[2].scatter(t, z, label='Position of Z', color='green', marker='o')
    axs[2].plot(t, predict_pos_speed(p[4], p[5], t), color='red', label='Gradient Descent')
    axs[2].plot(t, predict_pos_speed(p[10], p[11], t), color='blue', label='Linear Regression')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('z_position')
    axs[2].legend()

    plt.show()


def plot_positions_with_constant_acceleration(t, x, y, z, p):
    """
    Plot the measured and predicted positions of X, Y, and Z when constant acceleration is assumed.
    """
    fig, axs2 = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Position of X, Y, and Z')

    for ax in axs2:
        ax.set_xlim(-0.1, 6)
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

    t_smooth = np.linspace(-0.1, 6, 1000)

    # Plotting x

    axs2[0].scatter(t, x, label='Position of X', color='blue', marker='o')
    axs2[0].plot(t_smooth, p[0] + p[1] * t_smooth + p[2] * t_smooth ** 2, color='blue',
                 label='Polynomial Regression')
    axs2[0].plot(t_smooth, p[9] + p[10] * t_smooth + p[11] * t_smooth ** 2, color='red', label='Gradient Descent')
    axs2[0].set_xlabel('time')
    axs2[0].set_ylabel('x_position')
    axs2[0].legend()

    # Plotting y
    axs2[1].scatter(t, y, label='Position of Y', color='red', marker='o')
    axs2[1].plot(t_smooth, p[3] + p[4] * t_smooth + p[5] * t_smooth ** 2, color='blue',
                 label='Polynomial Regression')
    axs2[1].plot(t_smooth, p[12] + p[13] * t_smooth + p[14] * t_smooth ** 2, color='red', label='Gradient Descent')
    axs2[1].set_xlabel('time')
    axs2[1].set_ylabel('y_position')
    axs2[1].legend()

    # Plotting z
    axs2[2].scatter(t, z, label='Position of Z', color='green', marker='o')
    axs2[2].plot(t_smooth, p[6] + p[7] * t_smooth + p[8] * t_smooth ** 2, color='blue',
                 label='Polynomial Regression')
    axs2[2].plot(t_smooth, p[15] + p[16] * t_smooth + p[17] * t_smooth ** 2, color='red', label='Gradient Descent')
    axs2[2].set_xlabel('time')
    axs2[2].set_ylabel('z_position')
    axs2[2].legend()

    plt.show()
