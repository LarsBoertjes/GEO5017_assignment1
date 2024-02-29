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
