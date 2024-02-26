import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stat

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
        D_a1 = -2 * sum(t * (y - y_pred))  # partial derivative of a1
        D_a0 = -2 * sum(y - y_pred)  # partial derivative of a0
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

# 2.2a assume constant speed linear regression with polynomial regression
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




# 2.2b assume constant acceleration

# f(x) = ax_0 + ax_1 * t + ax_2 * t^2

# making the position plots


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

for ax in axs:
    ax.set_xlim(-0.1, 6.1)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

axs[0].scatter(t, x, label='Position of X', color='blue', marker='o')
axs[0].plot([min(t), max(t)], [predict_speed(ax_0, ax_1_lin, -0.1), predict_speed(ax_0, ax_1_lin, 6.1)], color='red')
axs[0].plot([min(t), max(t)], [predict_speed(ax_0_lin, ax_1_lin, -0.1), predict_speed(ax_0_lin, ax_1_lin, 6.1)], color='orange')
axs[0].set_xlabel('time')
axs[0].set_ylabel('x_position')

axs[1].scatter(t, y, label='Position of Y', color='red', marker='o')
axs[1].plot([min(t), max(t)], [predict_speed(ay_0, ay_1, -0.1), predict_speed(ay_0, ay_1, 6.1)], color='red')
axs[1].plot([min(t), max(t)], [predict_speed(ay_0_lin, ay_1_lin, -0.1), predict_speed(ay_0_lin, ay_1_lin, 6.1)], color='orange')
axs[1].set_xlabel('time')
axs[1].set_ylabel('y_position')

axs[2].scatter(t, z, label='Position of Z', color='green', marker='o')
axs[2].plot([min(t), max(t)], [predict_speed(az_0, az_1, -0.1), predict_speed(az_0, az_1, 6.1)], color='red')
axs[2].plot([min(t), max(t)], [predict_speed(az_0_lin, az_1_lin, -0.1), predict_speed(az_0_lin, az_1_lin, 6.1)], color='orange')
axs[2].set_xlabel('time')
axs[2].set_ylabel('z_position')

plt.show()