import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/positions.csv')

x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])
t = np.arange(0, len(x))

# 2.1 Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Trajectory Plot')
plt.show()


# 2.2a assume constant speed linear regression with gradient descent
# if speed is constant then a * t^2 = 0 therefore
# f(t) = ax_0 + ax_1 * t
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti))^2

def coefficient_finder(y, t, learning_rate, iterations):
    a0 = 0
    a1 = 0

    n = float(len(t))

    for i in range(iterations):
        y_pred = a1 * t + a0  # current predicted value of y
        D_a1 = -2  * sum(t * (y - y_pred))  # partial derivative of a1
        D_a0 = -2  * sum(y - y_pred)  # partial derivative of a0
        a1 = a1 - learning_rate * D_a1  # update a1
        a0 = a0 - learning_rate * D_a0  # update a0

    return a0, a1

def predict_speed(a0, a1, t):
    return a0 + a1 * t


learning_rate = 0.01
iterations = 100

ax_0, ax_1 = coefficient_finder(x, t, learning_rate, iterations)
ay_0, ay_1 = coefficient_finder(y, t, learning_rate, iterations)
az_0, az_1 = coefficient_finder(z, t, learning_rate, iterations)

error_x_speed = sum((x - predict_speed(ax_0, ax_1, t))**2)
error_y_speed = sum((y - predict_speed(ay_0, ay_1, t))**2)
error_z_speed = sum((z - predict_speed(az_0, az_1, t))**2)


print('ax_0 = ', ax_0, ' ax_1 = ', ax_1, 'ay_0 = ', ay_0, 'ay_1 = ', ay_1, 'az_0 = ', az_0, 'az_1 = ', az_1)
print('velocity in x-direction = ',ax_1, '\nvelocity in y-direction =',ay_1, '\nvelocity in z-direction = ', az_1,)
print('errors x y z: ', error_x_speed, error_y_speed, error_z_speed)

# 2.2a assume constant speed linear regression with linear regression
# if speed is constant then a * t^2 = 0 therefore
# f(t) = ax_0 + ax_1 * t
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti))^2

def calculate_coefficients_linear(t, y):
    cov_ty = np.cov(t, y, ddof=1)[0][1]
    var_t = np.var(t, ddof=1, dtype=float)
    a1 = cov_ty / var_t
    a0 = np.mean(y) - a1 * np.mean(t)
    # print(np.cov(t, y, bias=True), 'cov', cov_ty, 'var_t', var_t, 'a0', a0, 'a1', a1, 'mean', np.mean(t))
    return a0, a1

ax_0_lin, ax_1_lin = calculate_coefficients_linear(t, x)
ay_0_lin, ay_1_lin = calculate_coefficients_linear(t, y)
az_0_lin, az_1_lin = calculate_coefficients_linear(t, z)

error_x_speed_lin = sum((x - predict_speed(ax_0_lin, ax_1_lin, t))**2)
error_y_speed_lin = sum((y - predict_speed(ay_0_lin, ay_1_lin, t))**2)
error_z_speed_lin = sum((z - predict_speed(az_0_lin, az_1_lin, t))**2)

print('speed constant linear')
print('ax_0 = ', ax_0_lin, ' ax_1 = ', ax_1_lin, 'ay_0 = ', ay_0_lin, 'ay_1 = ', ay_1_lin, 'az_0 = ', az_0_lin, 'az_1 = ', az_1_lin)
print('velocity in x-direction = ',ax_1_lin, '\nvelocity in y-direction =',ay_1_lin, '\nvelocity in z-direction = ', az_1_lin,)
print('errors x y z: ', error_x_speed_lin, error_y_speed_lin, error_z_speed_lin)


# 2.2b assume constant acceleration thus polynomial form f(x) = ax_0 + ax_1 * t + ax_2 * t^2
# gradient descent method
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti + ax_2 * ti^2 ))^2

learning_rate = 0.0005
iterations = 5000
def gradient_solver(y, t, learning_rate, max_iter, tolerance):
    a0 = 0
    a1 = 0
    a2 = 0

    for i in range(max_iter):
        y_pred = a0 + a1 * t + a2 * t**2  # current predicted value of y
        D_a2 = -2 * sum(t**2 * (y - y_pred))  # partial derivative of a2
        D_a1 = -2 * sum(t * (y - y_pred))  # partial derivative of a1
        D_a0 = -2 * sum(y - y_pred)  # partial derivative of a0
        a2 = a2 - learning_rate * D_a2  # update a2
        a1 = a1 - learning_rate * D_a1  # update a1
        a0 = a0 - learning_rate * D_a0  # update a0

    return a0, a1, a2

def gradient_a2(t, y, y_pred):
    return -2 * sum(t**2 * (y - y_pred))

def gradient_a1(t, y, y_pred):
    return -2 * sum(t * (y - y_pred))

def gradient_a0(t, y, y_pred):
    return -2 * sum(y - y_pred)

def gradient_solver2(t, y, learning_rate, max_iter, tolerance, coefficient):
    if coefficient == 0:
        a0 = 0
        y_pred = a0
        gradient = gradient_a0(t, y, y_pred)

    if coefficient == 1:
        a1 = 0
        y_pred = a1 * t
        gradient = gradient_a1(t, y, y_pred)

    if coefficient == 2:
        a2 = 0
        y_pred = a1 * t**2
        gradient = gradient_a2(t, y, y_pred)

    for i in range(max_iter):
        a =
    return


def loss_olse(y, y_pred):
    return sum((y - y_pred)**2)

def predict_acc(a0, a1, a2, t):
    return a0 + a1 * t + a2 * t**2

bx_0, bx_1, bx_2 = gradient_solver(x, t, learning_rate, iterations)
by_0, by_1, by_2 = gradient_solver(y, t, learning_rate, iterations)
bz_0, bz_1, bz_2 = gradient_solver(z, t, learning_rate, iterations)

error_x_acc = loss_olse(x, predict_acc(bx_0, bx_1, bx_2, t))
error_y_acc = loss_olse(y, predict_acc(by_0, by_1, by_2, t))
error_z_acc = loss_olse(z, predict_acc(bz_0, bz_1, bz_2, t))

print('gradient descent polynomial')
print('ax_0 = ', bx_0, 'ax_1 = ', bx_1, 'ax_2 = ', bx_2,
      'ay_0 = ', by_0, 'ay_1 = ', by_1, 'ay_2 = ', by_2,
      'az_0 = ', bz_0, 'az_1 = ', bz_1, 'az_2 = ', bz_2, )
print('acceleration in x-direction = ',bx_2, '\nacceleration in y-direction =',by_2, '\nacceleration in z-direction = ', bz_2,)
print('errors x y z: ', error_x_acc, error_y_acc, error_z_acc)
print('next position on t = 6 will be (x,y,z): \n',
    predict_acc(bx_0, bx_1, bx_2, 6), predict_acc(by_0, by_1, by_2, 6), predict_acc(bz_0, bz_1, bz_2, 6))


# 2.2b assume constant acceleration thus polynomial form f(x) = ax_0 + ax_1 * t + ax_2 * t^2
# polynomial regression method
# objective function/loss is ordinary squared error  = (sum (xi - (ax_0 + ax_1 * ti + ax_2 * ti^2 ))^2

def polynomial_regression(t, y):
    D = np.vstack((t**0, t**1, t**2)).T
    y_up = y.T
    coefficients = np.linalg.inv(D.T @ D) @ D.T @ y_up
    return coefficients[0], coefficients[1], coefficients[2]

bx_0_poly, bx_1_poly, bx_2_poly = polynomial_regression(t, x)
by_0_poly, by_1_poly, by_2_poly = polynomial_regression(t, y)
bz_0_poly, bz_1_poly, bz_2_poly = polynomial_regression(t, z)

error_x_acc2 = loss_olse(x, predict_acc(bx_0_poly, bx_1_poly, bx_2_poly, t))
error_y_acc2 = loss_olse(y, predict_acc(by_0_poly, by_1_poly, by_2_poly, t))
error_z_acc2 = loss_olse(z, predict_acc(bz_0_poly, bz_1_poly, bz_2_poly, t))

print('polynomial regression')
print('errors x y z: ', error_x_acc2, error_y_acc2, error_z_acc2)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Position of X, Y, and Z')

for ax in axs:
    ax.set_xlim(-0.1, 6.1)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)


axs[0].scatter(t, x, label='Position of X', color='blue', marker='o')
axs[0].plot([min(t), max(t)], [predict_speed(ax_0, ax_1_lin, -0.1), predict_speed(ax_0, ax_1_lin, 6.1)], color='red', label='Gradient Descent')
axs[0].plot([min(t), max(t)], [predict_speed(ax_0_lin, ax_1_lin, -0.1), predict_speed(ax_0_lin, ax_1_lin, 6.1)], color='blue', label='Linear Regression')
axs[0].set_xlabel('time')
axs[0].set_ylabel('x_position')
axs[0].legend()

axs[1].scatter(t, y, label='Position of Y', color='red', marker='o')
axs[1].plot([min(t), max(t)], [predict_speed(ay_0, ay_1, -0.1), predict_speed(ay_0, ay_1, 6.1)], color='red', label='Gradient Descent')
axs[1].plot([min(t), max(t)], [predict_speed(ay_0_lin, ay_1_lin, -0.1), predict_speed(ay_0_lin, ay_1_lin, 6.1)], color='blue', label='Linear Regression')
axs[1].set_xlabel('time')
axs[1].set_ylabel('y_position')
axs[1].legend()

axs[2].scatter(t, z, label='Position of Z', color='green', marker='o')
axs[2].plot([min(t), max(t)], [predict_speed(az_0, az_1, -0.1), predict_speed(az_0, az_1, 6.1)], color='red', label='Gradient Descent')
axs[2].plot([min(t), max(t)], [predict_speed(az_0_lin, az_1_lin, -0.1), predict_speed(az_0_lin, az_1_lin, 6.1)], color='blue', label='Linear Regression')
axs[2].set_xlabel('time')
axs[2].set_ylabel('z_position')
axs[2].legend()

plt.show()

fig, axs2 = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Position of X, Y, and Z')

for ax in axs2:
    ax.set_xlim(-0.1, 6)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

t_smooth = np.linspace(min(t), max(t), 1000)

# Plotting x

axs2[0].scatter(t, x, label='Position of X', color='blue', marker='o')
x_poly = bx_0 + bx_1 * t + bx_2 * t ** 2  # Polynomial function
x_poly2 = bx_0_poly + bx_1_poly * t + bx_2_poly * t ** 2
axs2[0].plot(t_smooth, bx_0_poly + bx_1_poly * t_smooth + bx_2_poly * t_smooth ** 2, color='blue', label='Polynomial Regression')
axs2[0].plot(t_smooth, bx_0 + bx_1 * t_smooth + bx_2 * t_smooth ** 2, color='red', label='Gradient Descent')
axs2[0].set_xlabel('time')
axs2[0].set_ylabel('x_position')
axs2[0].legend()

# Plotting y
axs2[1].scatter(t, y, label='Position of Y', color='red', marker='o')
y_poly = by_0 + by_1 * t + by_2 * t ** 2  # Polynomial function
y_poly2 = by_0_poly + by_1_poly * t + by_2_poly * t ** 2
axs2[1].plot(t_smooth, by_0_poly + by_1_poly * t_smooth + by_2_poly * t_smooth ** 2, color='blue', label='Polynomial Regression')
axs2[1].plot(t_smooth, by_0 + by_1 * t_smooth + by_2 * t_smooth ** 2, color='red', label='Gradient Descent')
axs2[1].set_xlabel('time')
axs2[1].set_ylabel('y_position')
axs2[1].legend()

# Plotting z
axs2[2].scatter(t, z, label='Position of Z', color='green', marker='o')
z_poly = bz_0 + bz_1 * t + bz_2 * t ** 2  # Polynomial function
z_poly2 = bz_0_poly + bz_1_poly * t + bz_2_poly * t ** 2
axs2[2].plot(t_smooth, bz_0_poly + bz_1_poly * t_smooth + bz_2_poly * t_smooth ** 2, color='blue', label='Polynomial Regression')
axs2[2].plot(t_smooth, bz_0 + bz_1 * t_smooth + bz_2 * t_smooth ** 2, color='red', label='Gradient Descent')
axs2[2].set_xlabel('time')
axs2[2].set_ylabel('z_position')
axs2[2].legend()

plt.show()