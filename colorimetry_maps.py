import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_image():
    """Загрузка изображения через диалоговое окно"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    
    root.destroy()
    
    if not file_path:
        return None, None
    
    image = cv2.imread(file_path)
    return image, os.path.basename(file_path)


def create_ab_composite_simple(image):
    """
    Простая функция создания композита a→R, b→G
    
    Args:
        a_channel: канал a из Lab (0-255)
        b_channel: канал b из Lab (0-255)
    
    Returns:
        composite: BGR изображение с a в R, b в G
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # Получение каналов a и b
    _, a_channel, b_channel = cv2.split(lab)

    height, width = a_channel.shape
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    
    # BGR порядок в OpenCV: composite[:,:,0] = B, composite[:,:,1] = G, composite[:,:,2] = R
    composite[:, :, 1] = a_channel  # Красный канал = a
    composite[:, :, 2] = b_channel  # Зеленый канал = b
    # Синий канал остается 0
    
    return composite


def create_chromacity_heatmap(image, a_param=2, b_param=1):
    """Создание тепловой карты цветности"""
    # Конвертация в RGB и затем в Lab
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # Получение каналов a и b
    _, a, b = cv2.split(lab)
    
    # Нормализация в диапазон [-128, 127]
    a_norm = a.astype(np.float32) - 128
    b_norm = b.astype(np.float32) - 128
    
    # Расчет цветности
    chroma = np.sqrt(a_norm**2 + b_norm**2)
    vessel_params = a_norm*a_param - b_norm*b_param
    # Нормализация для визуализации
    chroma_norm = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vessel_norm = cv2.normalize(vessel_params, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    anti_vessel_norm = 255 - vessel_norm
    return anti_vessel_norm

def normalize_image_for_display(image):
    """Нормализация изображения для отображения"""
    # Конвертируем в float32 для нормализации
    img_float = image.astype(np.float32)
    
    # Нормализуем каждый канал отдельно или всё изображение целиком?
    # Для цветного изображения лучше нормализовать каждый канал отдельно,
    # чтобы сохранить цветовой баланс
    normalized = np.zeros_like(img_float)
    for i in range(3):
        channel = img_float[:, :, i]
        if channel.max() > channel.min():  # избегаем деления на ноль
            normalized[:, :, i] = 255 * (channel - channel.min()) / (channel.max() - channel.min())
        else:
            normalized[:, :, i] = channel
    
    return normalized.astype(np.uint8)

def main():
    # Загрузка изображения
    image, filename = load_image()
    
    if image is None:
        print("Изображение не выбрано")
        return
    
    a_param, b_param = 4, 3
    # Создание тепловой карты
    heatmap = create_chromacity_heatmap(image, a_param=a_param, b_param=b_param)
    composite = create_ab_composite_simple(image)
    
    mask, _ = load_image()
    
    # Нормализуем оригинальное изображение для отображения
    normalized_original = normalize_image_for_display(image)
    
    # Визуализация
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Оригинал (нормализованный)
    ax1.imshow(cv2.cvtColor(normalized_original, cv2.COLOR_BGR2GRAY), cmap='gray')
    ax1.set_title('Оригинал')
    ax1.axis('off')
    
    # Маска
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Маска')
    ax2.axis('off')
    
    '''
    # Тепловая карта
    im = ax3.imshow(heatmap, cmap='gray')
    ax3.set_title(f'Проекция компонент цветности a:b в соотношении (-{a_param}):{b_param}')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3)
    '''

    im = ax3.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Композит каналов a, b')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()