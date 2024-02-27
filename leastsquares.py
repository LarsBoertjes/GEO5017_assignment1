import numpy as np


def least_squares(input_values, output_values, k):
    d = np.column_stack([input_values ** i for i in range(k+1)])

    coeffs = np.linalg.inv(d.T @ d) @ d.T @ output_values

    predicted_values = d @ coeffs

    residuals = sum((output_values - predicted_values)**2)

    return residuals, coeffs, predicted_values





























