import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=4.0):
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

def calculate_histogram_flatness(image):
    """
    Вычисляет "плоскостность" гистограммы, которая может служить индикатором равномерности распределения яркости.
    Возвращает значение, которое стремится быть ниже для более равномерных распределений.
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
    hist_norm = hist / hist.sum()
    return np.std(hist_norm)  # Стандартное отклонение гистограммы

def apply_bilateral_filter(image, d=9, sigmaColor=25, sigmaSpace=25):
    """
    Применяет билатеральный фильтр для сглаживания изображения при сохранении краев.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

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
    corrected_l = gamma_correction(corrected_l, gamma=gamma)
    # corrected_l = apply_morphological_filters(corrected_l)
    # corrected_l = apply_bilateral_filter(corrected_l)
    # Объединение каналов обратно
    corrected_lab = cv2.merge([corrected_l, a, b])
    corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_image

    #Загрузка и обработка цветного изображения

# image = cv2.imread('1.jpg');
# image = cv2.imread('C:\\Users\\zakir\\Downloads\\Stars.jpg');
image = cv2.imread('C:\\Users\\zakir\\Downloads\\tt.png');

best_flatness = float('inf')
best_params = None
best_image = None

for clip_limit in np.linspace(1, 3, 15):  # Примерный диапазон для clip_limit
    for tile_grid_size in [(4, 4), (8, 8), (16, 16), (32, 32)]:  # Несколько вариантов для tile_grid_size
        corrected_image = correct_illumination_color(image, clip_limit, tile_grid_size)
        flatness = calculate_histogram_flatness(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY))
        if flatness < best_flatness:
            best_flatness = flatness
            best_params = (clip_limit, tile_grid_size)
            best_image = corrected_image

print(f"Best CLAHE parameters: Clip Limit = {best_params[0]}, Tile Grid Size = {best_params[1]}")

corrected_image = correct_illumination_color(image)

cv2.imwrite("change.jpg", best_image);
# Отображение результатов

plt.figure(figsize=(48, 24))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Corrected Image")
plt.imshow(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
plt.show()