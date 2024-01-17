import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def wavelet_transform(image, mode='db1', level=1):
    coeffs = pywt.wavedec2(image, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # High frequencies set to 0
    return pywt.waverec2(coeffs_H, mode)

def process_image_lab(img):
    # Load image

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Apply gamma correction
    l_gamma_corrected = gamma_correction(l, gamma=1.2)

    # Apply bilateral filter
    l_filtered = apply_bilateral_filter(l_gamma_corrected)

    # Apply wavelet transform
    l_wavelet = wavelet_transform(l_filtered)

    # Merge and convert back to BGR
    processed_lab = cv2.merge([l_wavelet, a, b])
    processed_img = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

    return processed_img
# Apply the processing

 # Replace with your image path

img = cv2.imread('C:\\Users\\zakir\\Downloads\\Stars.jpg')
processed_image = process_image_lab(img)
# Display the result

plt.figure(figsize=(48, 24))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Corrected Image")
plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
plt.show()