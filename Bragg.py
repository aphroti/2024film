import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import correlation
import os


def load_tiff_image(path):
    """ Load the TIFF image from the given path. """
    return Image.open(path)


def calculate_optical_density(image):
    """ Convert the image to grayscale, set white pixels as maximum intensity, and calculate optical density. """
    if image.mode != 'L':
        image = image.convert('L')

    image_array = np.array(image)

    # Set white pixels (value 255 in 8-bit images) as maximum intensity before OD calculation
    max_intensity = 255
    image_array[image_array == 255] = max_intensity

    # Prevent division by zero for any zero pixel values
    image_array[image_array == 0] = 1

    # Calculate optical density
    I0 = max_intensity  # Assuming the light source's maximum intensity corresponds to 255
    optical_density = np.log10(I0 / image_array)
    return optical_density


def find_line_of_symmetry(optical_density):
    """ Find the horizontal line that best splits the image into two symmetrical parts. """
    num_rows = optical_density.shape[0]
    min_correlation = np.inf
    best_line = None

    for i in range(num_rows // 2):  # Only need to check up to the middle
        upper_half = optical_density[:i, :]
        lower_half = optical_density[-i:, :]

        if upper_half.shape[0] != lower_half.shape[0]:
            continue

        current_correlation = correlation(upper_half.mean(axis=0), np.flipud(lower_half).mean(axis=0))

        if current_correlation < min_correlation:
            min_correlation = current_correlation
            best_line = i

    return best_line


def export_data_to_txt(data, filename):
    """ Export the given data (x, y) to a text file. """
    with open(filename, 'w') as file:
        for x, y in data:
            file.write(f"{x}\t{y}\n")


def plot_optical_density_curve(optical_density, line_position, dpi, filename):
    """ Plot the optical density along the specified horizontal line in the image with x-axis in mm and export data. """
    od_line = optical_density[line_position, :]
    non_zero_start = np.where(od_line > 0)[0][0]  # Find the first non-zero optical density
    od_line = od_line[non_zero_start:]  # Skip initial zeros
    pixels = np.arange(len(od_line)) + non_zero_start
    mm = pixels / dpi * 25.4  # Convert pixel position to mm

    data_pairs = list(zip(mm, od_line))  # Prepare data for export
    data_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_OD_data.txt"
    # export_data_to_txt(data_pairs, data_filename)
    print(f"Data exported to {data_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(mm, od_line, label='Optical Density')
    plt.title(os.path.basename(filename))
    plt.xlabel('Position (mm)')
    plt.ylabel('Optical Density')
    plt.grid(True)
    plt.show()


def plot_image_with_symmetry_line(image, line_position):
    """ Plot the original image with the symmetry line marked. """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.axhline(y=line_position, color='red', linestyle='--')
    plt.title('Original Image with Symmetry Line')
    plt.show()