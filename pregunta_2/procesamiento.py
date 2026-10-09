from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    # La referencia se conserva en escala de grises y en el rango [0, 1].
    image = Image.open(Path(path)).convert("L")
    return np.asarray(image, dtype=float) / 255.0


def add_gaussian_noise(image, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, sigma, image.shape)


def rmse(reference, estimation):
    return np.sqrt(np.mean((reference - estimation) ** 2))


def neighbors(image):
    padded = np.pad(image, 1, mode="edge")

    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    return north, south, west, east


def derivatives(image):
    north, south, west, east = neighbors(image)

    gradient = np.sqrt(((east - west) / 2) ** 2 + ((south - north) / 2) ** 2)
    laplacian = north + south + west + east - 4 * image

    return gradient, laplacian


def gaussian_blur(image, sigma):
    if sigma == 0:
        return image.copy()

    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)

    result = image.astype(float, copy=True)

    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        filtered = np.zeros_like(result)

        for offset, weight in enumerate(kernel):
            if axis == 0:
                filtered += weight * padded[offset:offset + image.shape[0], :]
            else:
                filtered += weight * padded[:, offset:offset + image.shape[1]]

        result = filtered

    return result


def coefficient_map(image, method, epsilon=0.05, k=0.10, alpha=0.25,
                    sigma_pre=1.0, c_floor=0.05):
    if method == "tv":

        gradient, _ = derivatives(image)
        return 1 / np.sqrt(gradient**2 + epsilon**2)

    if method == "borde":
        gradient, _ = derivatives(image)
        return np.maximum(0, 1 - (gradient / k) ** 2) ** 2

    if method == "laplaciano":

        # El pre-suavizado reduce la respuesta de la segunda derivada al ruido.
        smoothed = gaussian_blur(image, sigma_pre)
        gradient, laplacian = derivatives(smoothed)
        indicator = np.sqrt(gradient**2 + (alpha * laplacian) ** 2)
        return c_floor + (1 - c_floor) / (1 + (indicator / k) ** 4)
    return None


def stability_limit(method, epsilon=0.05):
    # La suma de los cuatro pesos de flujo no excede 4*c_max.
    # TV: c_max <= 1/epsilon. Las otras propuestas: c_max <= 1.
    if method == "tv":
        return epsilon / 4

    if method in ("borde", "laplaciano"):
        return 0.25

    return None


def diffuse(image, method, lambda_, iterations, **parameters):

    result = image.astype(float, copy=True)

    for _ in range(iterations):
        coefficient = coefficient_map(result, method, **parameters)
        north, south, west, east = neighbors(result)
        cn, cs, cw, ce = neighbors(coefficient)

        flux = (
            (coefficient + cn) / 2 * (north - result)
            + (coefficient + cs) / 2 * (south - result)
            + (coefficient + cw) / 2 * (west - result)
            + (coefficient + ce) / 2 * (east - result)
        )

        result = result + lambda_ * flux

    return result
