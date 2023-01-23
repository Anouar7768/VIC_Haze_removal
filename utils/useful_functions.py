import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import gaussian_filter
import cv2

# Step 1
def dark_channel_prior(img, patch_size=4):
    """ Computes the dark channel of an image

    Inputs:
        img:           The input image
        patch_size:    The size of the neighborhood considered

    Outputs:
        img_dark       The dark image
    """

    h, w, c = img.shape

    pad_img = np.array([np.pad(img[:, :, c], pad_width=patch_size) for c in range(img.shape[-1])])

    img_dark = np.zeros((h, w))

    for k in range(patch_size, h + patch_size):
        for j in range(patch_size, w + patch_size):
            img_dark[k - patch_size, j - patch_size] = np.nanmin(
                pad_img[:, k - patch_size:k + patch_size + 1, j - patch_size:j + patch_size + 1])

    # Mask the dark channel so it has nans where original image has
    img_dark[np.isnan(img[:, :, 0])] = np.nan

    return img_dark


# Step 2
def global_atmospheric_light(img, img_dark, patch_size=4):
    """ Computes the global atmospheric light of an image
        given its dark channel prior

    Inputs:
        img:           The input image
        img_dark:      The dark channel prior of the image

    Outputs:
        A              The global atmospheric light
    """

    xk = np.nanargmax(img_dark)
    xk = np.unravel_index(xk, img_dark.shape)

    # A = img[xk[1], xk[0]]
    A = img[xk]

    return A


# Step 3
def normalize(img, A):
    """ Normalizes the image by its global atmospheric light
        and stretches the values to [0,1]

    Inputs:
        img:           The input image
        A              The global atmospheric light

    Outputs:
        img_n          The image normalized by A and stretched to [0,1]
    """

    # Normalize by global atmospheric light
    img_n = img / A
    # Stretch to [0,1]
    img_n = img_n / np.nanmax(img_n, axis=(0, 1))

    return img_n


# Steps 4 & 5
def atmospheric_veil(img_n, sigma=2):
    """ Estimates the atmospheric veil of the
        normalized image and smooths it

    Inputs:
        img_n:           The normalized image

    Outputs:
        V                The estimated atmospheric veil
    """

    V = np.nanmin(img_n, axis=2)
    V = gaussian_filter(V, sigma=sigma)
    V = np.expand_dims(V, axis=-1)

    return V


def transmission(V):
    t = 1 - V
    return t


# Step 6
def restore_radiance(img, A, V, t, t0=1, k=1):
    """ Restore scene radiance after haze removal

    Inputs:
        img:           The input image
        A:             The global atmospheric light
        V:             The estimated atmospheric veil
        t:             The estimated transmission
        t0:            Lower bound on transmission t
        k:             Parameter to keep a small amount of haze
                       from distant objects

    Outputs:
        J              The restored scene radiance
    """

    J = A * ((img / A) - k * V) / np.maximum(t, t0)

    return J


# Wrap up
def dehazing(img, patch_size=4, sigma=2, t0=0.4, k=0.7):
    # Dark channel prior
    img_dark = dark_channel_prior(img, patch_size)
    # Global atmospheric light
    A = global_atmospheric_light(img, img_dark)
    # Normalization
    img_n = normalize(img, A)
    # Atmospheric veil
    V = atmospheric_veil(img_n, sigma=sigma)
    # Transmission
    t = transmission(V)
    # Restored scene radiance
    J = restore_radiance(img, A, V, t, t0=t0, k=k)

    return J


# Helper functions to display the original and its computed restored radiance
def plot_steps(rgb, patch_size=4, sigma=2, t0=0.4, k=0.7):
    # Dark channel prior
    dark_channel_rgb = dark_channel_prior(rgb, patch_size)
    # Global atmospheric light
    A = global_atmospheric_light(rgb, dark_channel_rgb)
    # print(A)
    # Normalization
    img_n = normalize(rgb, A)
    # Atmospheric veil
    V = atmospheric_veil(img_n, sigma=sigma)
    # Transmission
    t = transmission(V)
    # Restored scene radiance
    J = restore_radiance(rgb, A, V, t, t0, k)
    # Normalization
    J = J / np.nanmax(J, axis=(0, 1))

    # Plot the different steps
    fig, axes = plt.subplots(1, 6, figsize=(20, 10))
    axes[0].imshow(rgb)
    axes[0].set_title("Original rgb image")
    axes[1].imshow(dark_channel_rgb, cmap='gray')
    axes[1].set_title("Darkchannel prior")
    axes[2].imshow(img_n)
    axes[2].set_title("Normalized image")
    axes[3].imshow(V, cmap='gray')
    axes[3].set_title("Atmospheric veil")
    axes[4].imshow(t, cmap='gray')
    axes[4].set_title("Transmission")
    axes[5].imshow(J)
    axes[5].set_title("Restored scene radiance")
    plt.suptitle("Different steps of the algorithm", y=0.7)


def plot_comparison(img, patch_size=4, sigma=2, t0=0.4, k=0.7):
    # Compute resotred radiance
    J = dehazing(img, patch_size, sigma, t0, k)
    # Normalization
    J = J / np.nanmax(J, axis=(0, 1))

    # Print original image vs result
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original rgb image")
    axes[0, 1].imshow(J)
    axes[0, 1].set_title("Restored scene radiance")

    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([(img * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        histJ = cv2.calcHist([(J * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        axes[1, 0].plot(histr, color=col)
        axes[1, 1].plot(histJ, color=col)


def plot_comparison2(img1, img2, location, patch_size=4, sigma=2, t0=0.4, k=0.7):
    # Compute restored radiance
    J = dehazing(img1, patch_size, sigma, t0, k)
    # Normalization
    J = J / np.nanmax(J, axis=(0, 1))

    # Print original image vs result
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f"Original rgb hazy image - {location}")
    axes[0, 1].imshow(J)
    axes[0, 1].set_title("Restored scene radiance")
    axes[0, 2].imshow(img2)
    axes[0, 2].set_title("Same place without haze at a close date")

    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv2.calcHist([(img1 * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        histr2 = cv2.calcHist([(img2 * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        histJ = cv2.calcHist([(J * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        axes[1, 0].plot(histr, color=col)
        axes[1, 1].plot(histJ, color=col)
        axes[1, 2].plot(histr2, color=col)
    
    plt.suptitle(f"Parameters : sigma={sigma}, t0={t0}, k={k}")

    return None


def plot_results(input_img, expected_img, img_type, location, patch_size=4, sigma=2, t0=0.4, k=0.7, steps=False, comparison=False):
    """
    Function that plots the different steps of the algorithm, and compares the result with an image without haze

    Inputs:

    input_img:       path of the original image, with haze
    expected_img:    path of an image of the same place, without haze
    img_type:        type of the image : 'raster' or 'rgb'
    location:        location of the satellite image

    Outputs:

    None

    """

    if img_type == "rgb":

        hazy_img = cv2.imread(input_img)
        clean_img = cv2.imread(expected_img)


    elif img_type == "raster":

        # Import array from raster (tif) file
        with rasterio.open(input_img) as src:
            input_img = src.read()

        # Set 0 values to nan to ignore them afterwards
        input_img = np.where(input_img == 0, np.nan, input_img)

        # Transpose to get an (.,.,10) image with rgb bands
        input_img = np.moveaxis(input_img, source=0, destination=-1)
        # print(input_img.shape)

        # Extract rgb bands
        hazy_img = input_img[:, :, [2, 1, 0]]
        # print(np.isnan(hazy_img).sum())
        hazy_img[np.isnan(hazy_img)] = np.nanmean(hazy_img)

        # Normalize bands
        hazy_img = hazy_img / np.nanmax(hazy_img, axis=(0, 1))

        # Import array from raster (tif) file
        with rasterio.open(expected_img) as src:
            expected_img = src.read()
        # Set 0 values to nan to ignore them afterwards
        expected_img = np.where(expected_img == 0, np.nan, expected_img)
        # Transpose to get an (.,.,10) image with rgb bands
        expected_img = np.moveaxis(expected_img, source=0, destination=-1)
        # print(expected_img.shape)
        # Extract rgb bands
        clean_img = expected_img[:, :, [2, 1, 0]]

        # Normalize bands
        clean_img = clean_img / np.nanmax(clean_img, axis=(0, 1))

    if steps:
        # # Plot steps of the algorithm
        plot_steps(hazy_img, patch_size, sigma, t0, k)

    if comparison:
        plot_comparison2(hazy_img, clean_img, location, patch_size=patch_size, sigma=sigma, t0=t0, k=k)

    return None


def plot_img_vs_histogram(img_name, location, img_type):
    if img_type == "raster":

        # Import array from raster (tif) file
        with rasterio.open(img_name) as src:
            img = src.read()

        # Set 0 values to nan to ignore them afterwards
        img = np.where(img == 0, np.nan, img)

        # Transpose to get an (.,.,10) image with rgb bands
        img = np.moveaxis(img, source=0, destination=-1)

        print(img.shape)

        # Extract rgb bands
        rgb = img[:, :, [2, 1, 0]]

        # Normalize bands
        # rgb = (rgb-np.nanmin(rgb, axis=(0,1)))/(np.nanmax(rgb, axis=(0,1)))
        rgb = rgb / (np.nanmax(rgb, axis=(0, 1)))

    elif img_type == "rgb":

        rgb = plt.imread(img_name)

    colors = ('b', 'g', 'r')

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Original rgb image - {location}")

    for i, col in enumerate(colors):
        histr = cv2.calcHist([(rgb * 255).astype(np.uint16)], [i], None, [256], [0, 256])
        axes[1].plot(histr, color=col)
        axes[1].set_title(f"Histogram of rgb color bands")
    return None
