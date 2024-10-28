
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# Load the scanned .tiff image
image_path = "W:\\RadOnc\\Research\\2024_flash\\Flash_film_scan\\test061.tif"
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert("L")
image_array = np.array(gray_image)

# Thresholding to differentiate exposed and unexposed regions
# Adjust this threshold based on the characteristics of your image.
threshold_value = 200
_, binary_image = cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour, assuming it is the exposed region
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest contour (the exposed region)
mask = np.zeros_like(image_array, dtype=np.uint8)
cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)

# Extract pixel values from the exposed region using the mask
exposed_region_pixels = image_array[mask == 255]

# Calculate the average pixel value of an unexposed region (e.g., the background)
unexposed_roi = image_array[0:100, 0:100]  # Adjust as needed
I_0 = np.mean(unexposed_roi)

# Define a small epsilon to avoid log(0)
epsilon = 1e-10

# Ensure no zero or negative inputs to the logarithm
exposed_region_pixels = np.clip(exposed_region_pixels, epsilon, None)
I_0 = max(I_0, epsilon)

# Calculate optical density (OD) for each pixel in the exposed region
od_values = -np.log10(exposed_region_pixels / I_0)

# Find the maximum OD value in the exposed region
max_od = np.max(od_values)

# Find the location of the pixels with the maximum OD value
max_od_indices = np.argwhere(mask == 255)
max_od_locations = max_od_indices[od_values == max_od]

# Calculate the center of the detected exposed region (using image moments)
M = cv2.moments(largest_contour)
center_x = int(M["m10"] / M["m00"])
center_y = int(M["m01"] / M["m00"])
center = np.array([center_y, center_x])

# Select the closest pixel to the center as the single maximum point
distances = np.linalg.norm(max_od_locations - center, axis=1)
closest_index = np.argmin(distances)
max_od_yx = max_od_locations[closest_index]
max_y, max_x = max_od_yx

# Extract the OD profile along the x and y directions passing through the maximum point
od_x_profile = -np.log10(np.clip(image_array[max_y, :] / I_0, epsilon, None))
od_y_profile = -np.log10(np.clip(image_array[:, max_x] / I_0, epsilon, None))

# Convert pixels to millimeters
dpi = 266
mm_per_pixel = 25.4 / dpi

# Create x-axis for the OD profile in mm, centered around the maximum point
x_pixels = np.arange(image_array.shape[1]) - max_x
y_pixels = np.arange(image_array.shape[0]) - max_y

x_mm = x_pixels * mm_per_pixel
y_mm = y_pixels * mm_per_pixel

# Create a plot for the OD profiles with distances in mm
plt.figure(figsize=(10, 5))
plt.plot(x_mm, od_x_profile, label='OD along X direction', color='blue')
plt.plot(y_mm, od_y_profile, label='OD along Y direction', color='red')
plt.xlabel('Distance (mm)')
plt.ylabel('Optical Density (OD)')
plt.title(image_path)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the original image with lines indicating where the profiles were taken
plt.figure(figsize=(6, 6))
plt.imshow(image_array, cmap='gray')
plt.axhline(y=max_y, color='blue', linestyle='--', label='OD along X direction')
plt.axvline(x=max_x, color='red', linestyle='--', label='OD along Y direction')
plt.scatter(max_x, max_y, color='yellow', marker='x', s=100, label='Max OD Point')
plt.title(image_path)
plt.axis('off')
plt.legend(loc='upper right')
plt.show()
