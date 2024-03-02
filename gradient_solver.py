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


def gradient_solver_including_errors(y, t, learning_rate, max_iter, tolerance, k):
    parameters = [0] * (k+1)
    residuals = []

    y = np.array(y)
    t = np.array(t)

    for _ in range(max_iter):
        y_pred = np.sum([parameters[i] * (t ** i) for i in range(k + 1)], axis=0)

        residual = y - y_pred
        residuals.append(np.round(np.sum(residual ** 2), 4))

        gradients = [0] * (k + 1)
        for i in range(k + 1):
            gradients[i] = -2 * sum(t**i * residual)

        step_size = learning_rate * np.array(gradients)
        if np.linalg.norm(step_size) < tolerance:
            break

        parameters = np.array(parameters) - step_size

    return tuple(parameters), residuals

