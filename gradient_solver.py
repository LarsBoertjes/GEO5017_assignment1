import numpy as np
def gradient_solver(y, t, learning_rate, max_iter, tolerance, k):
    parameters = [0] * (k+1)

    for _ in range(max_iter):
        y_pred = 0

        for i in range(k + 1):
            y_pred += parameters[i] * (t ** i)

        gradients = [0] * (k + 1)
        for i in range(k + 1):
            gradients[i] = -2 * sum(t**i * (y - y_pred))

        step_size = learning_rate * np.array(gradients)
        if np.linalg.norm(step_size) < tolerance:
            break
        for i in range(k + 1):
            parameters[i] -= step_size[i]

    return tuple(parameters)

def gradient_solver_a1(y, t, learning_rate, max_iter, tolerance):
    a0 = 0
    a1 = 0

    for i in range(max_iter):
        y_pred = a0 + a1 * t  # current predicted value of y
        D_a1 = -2 * sum(t * (y - y_pred))  # partial derivative of a1
        D_a0 = -2 * sum(y - y_pred)  # partial derivative of a0

        step_size_a1 = learning_rate * D_a1
        step_size_a0 = learning_rate * D_a0

        if abs(step_size_a1 + step_size_a0) < tolerance:
            break

        a1 = a1 - step_size_a1  # update a1
        a0 = a0 - step_size_a0  # update a0

    return a0, a1

def gradient_solver_a2(y, t, learning_rate, max_iter, tolerance):
    a0 = 0
    a1 = 0
    a2 = 0

    for i in range(max_iter):
        y_pred = a0 + a1 * t + a2 * t ** 2  # current predicted value of y
        D_a2 = -2 * sum(t ** 2 * (y - y_pred))  # partial derivative of a2
        D_a1 = -2 * sum(t * (y - y_pred))  # partial derivative of a1
        D_a0 = -2 * sum(y - y_pred)  # partial derivative of a0

        ss2 = learning_rate * D_a2
        ss1 = learning_rate * D_a1
        ss0 = learning_rate * D_a0

        if abs(ss2 + ss1 + ss0) < tolerance:
            break

        a2 = a2 - ss2  # update a2
        a1 = a1 - ss1  # update a1
        a0 = a0 - ss0  # update a0

    return a0, a1, a2

