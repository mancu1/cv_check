import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image, num_segments=12):
    """
    This function segments the given image into smaller parts.

    Parameters:
    image (numpy.ndarray): The input image to be segmented.
    num_segments (int): The number of segments to divide the image into. Default is 4.

    Returns:
    list: A list of segmented parts of the image.
    """
    segment_height = image.shape[0] // num_segments
    segment_width = image.shape[1] // num_segments

    segments = []
    for i in range(num_segments):
        for j in range(num_segments):
            segment = image[i * segment_height:(i + 1) * segment_height, j * segment_width:(j + 1) * segment_width]
            segments.append(segment)
    return segments

def correct_illumination(segment):
    """
    This function corrects the illumination of a given image segment.

    Parameters:
    segment (numpy.ndarray): The input image segment to correct the illumination of.

    Returns:
    numpy.ndarray: The image segment with corrected illumination.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    return clahe.apply(segment)

# Load the image in grayscale
image = cv2.imread('C:\\Users\\zakir\\Downloads\\Stars.jpg', cv2.IMREAD_GRAYSCALE)

# Segment the image
segments = segment_image(image)

# Correct the illumination of each segment
corrected_segments = [correct_illumination(segment) for segment in segments]

def integrate_segments(segments, num_segments=12):
    """
    This function integrates the corrected segments back into a single image.

    Parameters:
    segments (list): The list of corrected image segments.
    num_segments (int): The number of segments the image was divided into. Default is 4.

    Returns:
    numpy.ndarray: The integrated image.
    """
    height, width = segments[0].shape
    integrated_image = np.zeros((height * num_segments, width * num_segments), dtype=np.uint8)

    for i in range(num_segments):
        for j in range(num_segments):
            integrated_image[i * height:(i + 1) * height, j * width:(j + 1) * width] = segments[i * num_segments + j]

    return integrated_image

# Integrate the corrected segments back into a single image
integrated_image = integrate_segments(corrected_segments)

# Display the original and corrected images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Corrected Image")
plt.imshow(integrated_image, cmap='gray')
plt.show()