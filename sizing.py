import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def analyze_radiochromic_film(tiff_path):
    # Load the TIFF image
    img = Image.open(tiff_path).convert('RGB')

    # Split image into RGB channels and convert to grayscale intensity (0-255)
    r, g, b = img.split()
    r_gray = np.array(r) / 255.0
    g_gray = np.array(g) / 255.0
    b_gray = np.array(b) / 255.0

    # Compute the average grayscale image from the three channels
    avg_gray = (r_gray + g_gray + b_gray) / 3.0

    # Define function to calculate optical density (OD)
    def calculate_optical_density(gray_channel):
        # Avoid division by zero by setting a minimum intensity level
        gray_channel = np.clip(gray_channel, 1e-3, 1.0)
        return -np.log10(gray_channel)

    # Calculate optical density for the averaged grayscale image
    od_avg = calculate_optical_density(avg_gray)

    # Find the maximum OD value in the image
    max_od = np.max(od_avg)

    # Step 1: Identify Set A (within 20% of max OD)
    set_a_mask = (od_avg >= 0.8 * max_od)
    od_img_a = np.where(set_a_mask, od_avg, 0)
    od_img_a = (od_img_a * 255).astype(np.uint8)

    # Apply Gaussian blur to reduce noise within Set A
    blurred_a = cv2.GaussianBlur(od_img_a, (5, 5), 0)

    # Use Canny edge detection to find edges in the Set A region
    edges_a = cv2.Canny(blurred_a, 50, 150)

    # Find contours in the Set A edge-detected image
    contours_a, _ = cv2.findContours(edges_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fit an ellipse to the largest contour in Set A
    largest_contour_a = max(contours_a, key=cv2.contourArea) if contours_a else None
    ellipse_center = None

    if largest_contour_a is not None and len(largest_contour_a) >= 5:  # At least 5 points needed for ellipse fitting
        ellipse = cv2.fitEllipse(largest_contour_a)
        ellipse_center = (int(ellipse[0][0]), int(ellipse[0][1]))

    # Step 2: Identify Set B (within 2% of max OD)
    set_b_mask = (od_avg >= 0.98 * max_od)
    set_b_indices = np.column_stack(np.where(set_b_mask))

    # Find the pixel in Set B closest to the ellipse center
    closest_pixel = None
    if ellipse_center is not None and len(set_b_indices) > 0:
        distances = np.sqrt(
            (set_b_indices[:, 1] - ellipse_center[0]) ** 2 + (set_b_indices[:, 0] - ellipse_center[1]) ** 2)
        closest_idx = np.argmin(distances)
        radiation_center = (set_b_indices[closest_idx][1], set_b_indices[closest_idx][0])

    # Plot optical density of the averaged grayscale image with the ellipse and center marked
    plt.figure(figsize=(8, 6))
    plt.imshow(od_avg, cmap='gray')
    plt.title('Optical Density - Averaged Grayscale with Elliptical Mark')
    plt.axis('off')

    # Plot the fitted ellipse from Set A
    if largest_contour_a is not None and ellipse_center is not None:
        ellipse_box = cv2.ellipse(np.zeros_like(od_img_a), ellipse, 255, 2)
        plt.contour(ellipse_box, colors='red', linewidths=0.5)

        # Mark the ellipse center
        plt.plot(ellipse_center[0], ellipse_center[1], 'bo', markersize=10, label="Ellipse Center")

    # Plot the radiation center in Set B
    if radiation_center is not None:
        plt.plot(radiation_center[0], radiation_center[1], 'go', markersize=8, label="Radiation Center")

        # Draw vertical and horizontal lines through the radiation center
        plt.axvline(x=radiation_center[0], color='yellow', linestyle='--', linewidth=1)
        plt.axhline(y=radiation_center[1], color='yellow', linestyle='--', linewidth=1)

    plt.legend()
    plt.show()

    # Step 3: Extract pixel values along the vertical and horizontal lines at the radiation center
    if radiation_center is not None:
        vertical_line = od_avg[:, radiation_center[0]]
        horizontal_line = od_avg[radiation_center[1], :]

        # Calculate distances from the radiation center for x-axis values
        vertical_distances = np.arange(len(vertical_line)) - radiation_center[1]
        horizontal_distances = np.arange(len(horizontal_line)) - radiation_center[0]

        # Plot the pixel values along these lines with distance to radiation center as x-axis
        plt.figure(figsize=(10, 5))
        plt.plot(vertical_distances, vertical_line, label='Vertical Line', color='blue')
        plt.plot(horizontal_distances, horizontal_line, label='Horizontal Line', color='green')
        plt.xlabel('Distance to Radiation Center (pixels)')
        plt.ylabel('Optical Density')
        plt.title('Optical Density Along Vertical and Horizontal Lines at Radiation Center')
        plt.legend()
        plt.show()

        # Step 4: Calculate the widths at 10% of max OD
        threshold = 0.1 * max_od

        # For vertical line
        vertical_positions = np.where(vertical_line >= threshold)[0]
        if len(vertical_positions) > 0:
            vertical_width = vertical_positions[-1] - vertical_positions[0]
            print(f"Vertical width at 90% max OD: {vertical_width} pixels")

        # For horizontal line
        horizontal_positions = np.where(horizontal_line >= threshold)[0]
        if len(horizontal_positions) > 0:
            horizontal_width = horizontal_positions[-1] - horizontal_positions[0]
            print(f"Horizontal width at 90% max OD: {horizontal_width} pixels")
