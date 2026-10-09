import numpy as np
from scipy.signal import convolve2d

np.random.seed(0)

def base_image():
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    base_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 0.15, dtype=float)

    # Cuadrado centrado de 128x128

    inicio = 64
    fin = 192

    square_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    square_mask[inicio:fin, inicio:fin] = True

    base_image[square_mask] = 0.45

    # Cicrulo centrado de radio 32

    yy, xx = np.indices((IMAGE_HEIGHT, IMAGE_WIDTH))

    cx = (IMAGE_WIDTH - 1) / 2
    cy = (IMAGE_HEIGHT - 1) / 2

    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 32 ** 2

    base_image[circle_mask] = 0.75

    background_mask = ~square_mask
    square_mask = square_mask & ~circle_mask

    # Ruido Poisson

    N = 40
    noise_base_image = np.random.poisson(N * base_image) / N

    return base_image, noise_base_image, square_mask, circle_mask, background_mask

# Kernel Gaussiano

def gaussian_kernel(sigma):
    if (sigma == 0):
        return np.array([[1.0]])

    radius = int(np.ceil(3 * sigma))

    coords = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(coords, coords)

    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel) # Normalizacion

    return kernel

def gaussian_fiilter(image, sigma):
    if (sigma == 0):
        return image

    kernel = gaussian_kernel(sigma)

    return convolve2d(
        image, kernel, mode='same', boundary='symm'
    )

def rmse(referencia, estimacion, mask=None):
    if mask is None:
        error = referencia - estimacion
    else:
        error = referencia[mask] - estimacion[mask]

    return np.sqrt(np.mean(error**2))

def rmse_curve(image, noise_image, mask, sigmas):

    rmses = []

    for sigma in sigmas:
        filtered = gaussian_fiilter(noise_image, sigma)

        rmses.append(rmse(image, filtered, mask))

    rmses = np.array(rmses)

    min = np.argmin(rmses)
    sigma_minimo = sigmas[min]

    return rmses, min, sigma_minimo

def spatial_adaptive_filter(image, noise_image, intensities, optimal_sigmas):
    sigma_mu = 2.0

    mu_hat = gaussian_fiilter(noise_image, sigma_mu)

    sigma_map = np.interp(
        mu_hat,
        intensities,
        optimal_sigmas,
        left=optimal_sigmas[0],
        right=optimal_sigmas[-1]
    )

    max_sigma = np.max(sigma_map)

    max_radius = int(np.ceil(3 * max_sigma))

    padded_image = np.pad(
        image, max_radius, mode="symmetric"
    )

    result = np.zeros_like(image)

    IMAGE_HEIGHT, IMAGE_WIDTH = image.shape

    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            sigma = sigma_map[i, j]
            kernel = gaussian_kernel(sigma)
            radius = kernel.shape[0] // 2

            ip = i + max_radius
            jp = j + max_radius

            patch = padded_image[
                ip - radius:ip + radius + 1,
                jp - radius:jp + radius + 1
            ]

            result[i, j] = np.sum(patch * kernel)

    return result