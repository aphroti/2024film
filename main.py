# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import Bragg
from PIL import Image

import sizing

# bragg
'''
dpi = 266  # Dots Per Inch
path_to_tiff = "W:\\RadOnc\\Research\\2024_flash\\Flash_film_scan\\test002crop.tif"  # Update this path
filename = path_to_tiff.split('/')[-1]  # Extract the filename from the path
image = Bragg.load_tiff_image(path_to_tiff)
optical_density = Bragg.calculate_optical_density(image)
symmetry_line = Bragg.find_line_of_symmetry(optical_density)

Bragg.plot_optical_density_curve(optical_density, line_position=symmetry_line, dpi=dpi, filename=filename)
Bragg.plot_image_with_symmetry_line(image, line_position=symmetry_line)
'''

# beam profile
'''
image_path = "W:\\RadOnc\\Research\\2024_flash\\Flash_film_scan\\test003.tif"
grayscale = beamProf.read_and_convert_to_grayscale(image_path)
circular_region = beamProf.find_circular_region(grayscale)
optical_density = beamProf.calculate_optical_density(grayscale, circular_region)
beamProf.plot_optical_density(optical_density)
'''
'''
# Usage
dpi = 266  # Dots Per Inch
path_to_tiff = "W:\\RadOnc\\Research\\2024_flash\\Flash_film_scan\\test001crop.tif"  # Update this path
image = Bragg.load_tiff_image(path_to_tiff)
optical_density = Bragg.calculate_optical_density(image)
symmetry_line = Bragg.find_line_of_symmetry(optical_density)

Bragg.plot_optical_density_curve(optical_density, line_position=symmetry_line, dpi=dpi, filename=path_to_tiff)
Bragg.plot_image_with_symmetry_line(image, line_position=symmetry_line)
'''

# Usage
tiff_path = "W:\\RadOnc\\Research\\2024_flash\\Flash_film_scan\\Cali002.tif"
sizing.analyze_radiochromic_film(tiff_path)
