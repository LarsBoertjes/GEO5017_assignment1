import numpy as np


def gradient_solver(y, t, learning_rate, max_iter, tolerance, k):
    parameters = [0] * (k)

    for _ in range(max_iter):
        y_pred = 0

        for i in range(k):
            y_pred += parameters[i] * (t ** i)

        gradients = [0] * (k)
        for i in range(k):
            gradients[i] = -2 * sum(t**i * (y - y_pred))

        step_size = learning_rate * np.array(gradients)
        if np.linalg.norm(step_size) < tolerance:
            break
        for i in range(k):
            parameters[i] -= step_size[i]

    return tuple(parameters)


def gradient_solver_including_errors(y, t, learning_rate, max_iter, tolerance, k):
    parameters = [0] * (k)
    residuals = []

    y = np.array(y)
    t = np.array(t)

    for _ in range(max_iter):
        y_pred = np.sum([parameters[i] * (t ** i) for i in range(k)], axis=0)

        residual = y - y_pred
        residuals.append(np.round(np.sum(residual ** 2), 4))

        gradients = [0] * (k)
        for i in range(k):
            gradients[i] = -2 * sum(t**i * residual)

        step_size = learning_rate * np.array(gradients)
        if np.linalg.norm(step_size) < tolerance:
            break

        parameters = np.array(parameters) - step_size

    return tuple(parameters), residuals

