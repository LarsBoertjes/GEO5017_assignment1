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

positions = [x, y, z]
linear_models = []
errors = []

for position in positions:
    error_speed_lin, linear_coefficients, positions = least_squares(t, position, 1)
    a0_lin, a1_lin = linear_coefficients
    linear_models.append((a0_lin, a1_lin))
    errors.append(error_speed_lin)

(ax_0_ls, ax_1_ls), (ay_0_ls, ay_1_ls), (az_0_ls, az_1_ls) = linear_models
error_x_speed_lin, error_y_speed_lin, error_z_speed_lin = errors


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

error_x_acc2 = least_squares(t, x, 2)[0]
error_y_acc2 = least_squares(t, y, 2)[0]
error_z_acc2 = least_squares(t, z, 2)[0]

# 2.2c next position



# Plotting the X, Y and Z measured positions and predicted positions

# For assumed constant speed
parameters_speed = [ax_0, ax_1, ay_0, ay_1, az_0, az_1, ax_0_ls, ax_1_ls, ay_0_ls, ay_1_ls, az_0_ls, az_1_ls]
plot_positions_with_constant_speed(t, x, y, z, parameters_speed)

# For assumed constant acceleration

parameters_acc = [bx_0, bx_1, bx_2, by_0, by_1, by_2, bz_0, bz_1, bz_2,
                  bx_0_ls, bx_1_ls, bx_2_ls, by_0_ls, by_1_ls, by_2_ls, bz_0_ls, bz_1_ls, bz_2_ls]
plot_positions_with_constant_acceleration(t, x, y, z, parameters_acc)

# printing all output data
print('ax_0 = ', ax_0, ' ax_1 = ', ax_1, 'ay_0 = ', ay_0, 'ay_1 = ', ay_1, 'az_0 = ', az_0, 'az_1 = ', az_1)
print('velocity in x-direction = ',ax_1, '\nvelocity in y-direction =',ay_1, '\nvelocity in z-direction = ', az_1,)
print('errors x y z: ', error_x_speed, error_y_speed, error_z_speed)
def print_speed_constant_linear(a0, a1, error):
    print("Speed Constant Linear Formula:")
    print(f"y = {round(a0, 3)} + {round(a1, 3)}*t")
    print(f"Error: {error}")

print_speed_constant_linear(ax_0_ls, ax_1_ls, error_x_speed_lin)
print_speed_constant_linear(ay_0_ls, ay_1_ls, error_y_speed_lin)
print_speed_constant_linear(az_0_ls, az_1_ls, error_z_speed_lin)

print("\nVelocities in Each Direction:")
print(f"Velocity in X-direction: {ax_1_ls}")
print(f"Velocity in Y-direction: {ay_1_ls}")
print(f"Velocity in Z-direction: {az_1_ls}")

print('gradient descent polynomial')
print('ax_0 = ', bx_0, 'ax_1 = ', bx_1, 'ax_2 = ', bx_2,
      'ay_0 = ', by_0, 'ay_1 = ', by_1, 'ay_2 = ', by_2,
      'az_0 = ', bz_0, 'az_1 = ', bz_1, 'az_2 = ', bz_2, )
print('acceleration in x-direction = ',bx_2, '\nacceleration in y-direction =',by_2, '\nacceleration in z-direction = ', bz_2,)
print('errors x y z: ', error_x_acc, error_y_acc, error_z_acc)
print('next position on t = 6 will be (x,y,z): \n',
    predict_pos_acc(bx_0, bx_1, bx_2, 6), predict_pos_acc(by_0, by_1, by_2, 6), predict_pos_acc(bz_0, bz_1, bz_2, 6))

print('polynomial regression')
print('errors x y z: ', error_x_acc2, error_y_acc2, error_z_acc2)
