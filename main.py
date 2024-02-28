import pandas as pd
import numpy as np
from leastsquares import least_squares
from gradient_solver import gradient_solver_a1, gradient_solver_a2
from plotting import plot_trajectory, plot_positions_with_constant_speed,  plot_positions_with_constant_acceleration


def predict_pos_speed(a0, a1, t):
    return a0 + a1 * t


def predict_pos_acc(a0, a1, a2, t):
    return a0 + a1 * t + a2 * t**2


def loss_olse(y, y_pred):
    return sum((y - y_pred)**2)


# 2.0 Reading the given position data
df = pd.read_csv('data/positions.csv')

x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])
t = np.arange(0, len(x))

# 2.1 Plot trajectory
plot_trajectory(x, y, z)

# 2.2a assume constant speed linear regression with gradient descent
# if speed is constant then a * t^2 = 0 therefore
# f(t) = ax_0 + ax_1 * t
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti))^2

learning_rate = 0.001
iterations = 10000

ax_0, ax_1 = gradient_solver_a1(x, t, learning_rate, iterations, 0.000001)
ay_0, ay_1 = gradient_solver_a1(y, t, learning_rate, iterations, 0.000001)
az_0, az_1 = gradient_solver_a1(z, t, learning_rate, iterations, 0.000001)

error_x_speed = loss_olse(x, predict_pos_speed(ax_0, ax_1, t))
error_y_speed = loss_olse(y, predict_pos_speed(ay_0, ay_1, t))
error_z_speed = loss_olse(z, predict_pos_speed(az_0, az_1, t))

# 2.2a assume constant speed linear regression with polynomial regression
# if speed is constant then a * t^2 = 0 therefore
# f(t) = ax_0 + ax_1 * t
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti))^2
ax_0_ls, ax_1_ls = least_squares(t, x, 1)[1]
ay_0_ls, ay_1_ls = least_squares(t, y, 1)[1]
az_0_ls, az_1_ls = least_squares(t, z, 1)[1]

error_x_speed_ls = least_squares(t, x, 1)[0]
error_y_speed_ls = least_squares(t, y, 1)[0]
error_z_speed_ls = least_squares(t, z, 1)[0]


# 2.2b assume constant acceleration thus polynomial form f(x) = ax_0 + ax_1 * t + ax_2 * t^2
# gradient descent method
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti + ax_2 * ti^2 ))^2

learning_rate = 0.0005
iterations = 5000

bx_0, bx_1, bx_2 = gradient_solver_a2(x, t, learning_rate, iterations, 0.00000001)
by_0, by_1, by_2 = gradient_solver_a2(y, t, learning_rate, iterations, 0.00000001)
bz_0, bz_1, bz_2 = gradient_solver_a2(z, t, learning_rate, iterations, 0.00000001)

error_x_acc = loss_olse(x, predict_pos_acc(bx_0, bx_1, bx_2, t))
error_y_acc = loss_olse(y, predict_pos_acc(by_0, by_1, by_2, t))
error_z_acc = loss_olse(z, predict_pos_acc(bz_0, bz_1, bz_2, t))

# 2.2b assume constant acceleration thus polynomial form f(x) = ax_0 + ax_1 * t + ax_2 * t^2
# polynomial regression method
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti + ax_2 * ti^2 ))^2

bx_0_ls, bx_1_ls, bx_2_ls = least_squares(t, x, 2)[1]
by_0_ls, by_1_ls, by_2_ls = least_squares(t, y, 2)[1]
bz_0_ls, bz_1_ls, bz_2_ls = least_squares(t, z, 2)[1]

error_x_acc_ls = least_squares(t, x, 2)[0]
error_y_acc_ls = least_squares(t, y, 2)[0]
error_z_acc_ls = least_squares(t, z, 2)[0]

# 2.2c next position

next_position = (predict_pos_acc(bx_0, bx_1, bx_2, 6),
                 predict_pos_acc(by_0, by_1, by_2, 6),
                 predict_pos_acc(bz_0, bz_1, bz_2, 6))

x_next = np.append(x, next_position[0])
y_next = np.append(y, next_position[1])
z_next = np.append(z, next_position[2])

plot_trajectory(x_next, y_next, z_next)



# Plotting the X, Y and Z measured positions and predicted positions
# For assumed constant speed
parameters_speed = [ax_0, ax_1, ay_0, ay_1, az_0, az_1, ax_0_ls, ax_1_ls, ay_0_ls, ay_1_ls, az_0_ls, az_1_ls]
plot_positions_with_constant_speed(t, x, y, z, parameters_speed)

# For assumed constant acceleration
parameters_acc = [bx_0, bx_1, bx_2, by_0, by_1, by_2, bz_0, bz_1, bz_2,
                  bx_0_ls, bx_1_ls, bx_2_ls, by_0_ls, by_1_ls, by_2_ls, bz_0_ls, bz_1_ls, bz_2_ls]
plot_positions_with_constant_acceleration(t, x, y, z, parameters_acc)


# Printing all output data
def print_speed_constant_linear(dim, a0, a1, error):
    print(f"Speed Constant Linear Formula for {dim}:")
    print(f"y = {round(a0, 3)} + {round(a1, 3)}*t")
    print(f"Speed  = {round(a1, 3)} m/s ")
    print(f"Error: {round(error, 3)}\n")

def print_speed_constant_acc(dim, a0, a1, a2, error):
    print(f"Acceleration Constant Formula for {dim}:")
    print(f"y = {round(a0, 3)} + {round(a1, 3)}*t + {round(a2, 3)}*t^2")
    print(f"Acceleration = {round((2 * a2), 3)} m/s^2 ")
    print(f"Error: {round(error, 3)}\n")

print('Assuming constant speed and using gradient descent to solve:')
print_speed_constant_linear('X', ax_0, ax_1, error_x_speed)
print_speed_constant_linear('Y', ay_0, ay_1, error_y_speed)
print_speed_constant_linear('Z', az_0, az_1, error_z_speed)

print('\n\nAssuming constant acceleration and using gradient descent to solve:')
print_speed_constant_acc('X', bx_0, bx_1, bx_2, error_x_acc)
print_speed_constant_acc('Y', by_0, by_1, by_2, error_y_acc)
print_speed_constant_acc('Z', bz_0, bz_1, bz_2, error_z_acc)
print(f'\nThe predicted next position (x, y, z) = ({round(next_position[0], 3)}, '
      f'{round(next_position[1], 3)}, {round(next_position[2], 3)}) ')