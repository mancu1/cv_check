import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.0):
    """
    Применяет гамма-коррекцию к изображению.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Применение гамма-коррекции
    return cv2.LUT(image, table)

def apply_morphological_filters(image, kernel_size=(2, 2)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def correct_illumination_color(img, clip_limit=2.0, tile_grid_size=(16, 16), gamma=1.0):
    """
    Корректирует освещение цветного изображения, работая с каналом яркости в цветовом пространстве LAB.
    """
    # Преобразование изображения в цветовое пространство LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Применение CLAHE к каналу L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    corrected_l = clahe.apply(l)

    # Применение гамма-коррекции
    # corrected_l = gamma_correction(corrected_l, gamma=gamma)
    # corrected_l = apply_morphological_filters(corrected_l)

    # Объединение каналов обратно
    corrected_lab = cv2.merge([corrected_l, a, b])
    corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_image

    #Загрузка и обработка цветного изображения

image = cv2.imread('C:\\Users\\zakir\\Downloads\\Stars.jpg');
# image = cv2.imread('C:\\Users\\zakir\\Downloads\\tt.png');
corrected_image = correct_illumination_color(image)

cv2.imwrite("change.jpg", corrected_image);
# Отображение результатов

plt.figure(figsize=(48, 24))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Corrected Image")
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.show()