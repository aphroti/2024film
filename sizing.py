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

    # Create a mask for OD values within 10% of the maximum OD
    mask = (od_avg >= 0.8 * max_od)

    # Convert the masked OD to an 8-bit grayscale image for contour detection
    od_img = (od_avg * 255).astype(np.uint8)
    masked_od_img = np.where(mask, od_img, 0).astype(np.uint8)

    # Apply Gaussian blur to reduce noise within the masked image
    blurred = cv2.GaussianBlur(masked_od_img, (5, 5), 0)

    # Use Canny edge detection to find edges in the masked image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fit an ellipse to the largest contour within the masked region
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Plot optical density of the averaged grayscale image with the ellipse and center marked
    plt.figure(figsize=(8, 6))
    plt.imshow(od_avg, cmap='gray')
    plt.title('Optical Density - Averaged Grayscale with Elliptical Mark')
    plt.axis('off')

    # If a valid contour is found, fit an ellipse and mark the center
    if largest_contour is not None and len(largest_contour) >= 5:  # Need at least 5 points for ellipse fitting
        ellipse = cv2.fitEllipse(largest_contour)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))

        # Plot the ellipse
        ellipse_box = cv2.ellipse(np.zeros_like(od_img), ellipse, 255, 2)
        plt.contour(ellipse_box, colors='red', linewidths=0.5)

        # Mark the center point
        plt.plot(center[0], center[1], 'bo', markersize=10, label="Elliptical Center")
        plt.legend()

    plt.show()

