import os
import sys
import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.v2 as tfs_v2

# Добавляем папку со скриптом в путь поиска модулей
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Теперь можно импортировать
from test_model import UNetModel

def simple_circular_mask_otsu_only(image):
    """
    Максимально простой метод выделения круга через Оцу
    """
    # 1. Конвертируем в gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Оцу для выделения светлой области
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Морфологическая очистка
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 4. Находим контур светлой области
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # 5. Берем самый большой контур
    largest = max(contours, key=cv2.contourArea)
    
    # 6. Аппроксимируем круг через эллипс
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        radius = int(min(ellipse[1]) / 2)
        
        # Создаем маску
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        return mask, center, radius
    
    return None, None, None

def prepare_single_image(image_path, padding=20, target_size=(256, 256)):
    """
    Обрабатывает одно тестовое изображение:
    1. Выделяет круговую маску через Otsu
    2. Обрезает по квадрату с отступом
    3. Ресайзит до target_size
    
    Returns:
        prepared_image: подготовленное изображение (numpy array, RGB)
        roi_coords: координаты вырезанной области (x_min, y_min, x_max, y_max)
        circular_mask: круговая маска (для визуализации)
    """
    # Загружаем изображение
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Не удалось загрузить {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = image_rgb.shape[:2]
    
    # Получаем круговую маску
    circular_mask, center, radius = simple_circular_mask_otsu_only(image_bgr)
    
    if circular_mask is None:
        raise ValueError("Не удалось выделить круг!")
    
    # Находим границы маски
    rows, cols = np.where(circular_mask > 0)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    # Добавляем отступ
    y_min = max(0, y_min - padding)
    y_max = min(img_height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(img_width, x_max + padding)
    
    # Делаем область квадратной
    height = y_max - y_min
    width = x_max - x_min
    
    if height > width:
        diff = height - width
        x_min = max(0, x_min - diff // 2)
        x_max = min(img_width, x_max + diff // 2 + diff % 2)
    else:
        diff = width - height
        y_min = max(0, y_min - diff // 2)
        y_max = min(img_height, y_max + diff // 2 + diff % 2)
    
    # Вырезаем область
    roi = image_rgb[y_min:y_max, x_min:x_max]
    
    # Ресайзим
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    
    roi_coords = (x_min, y_min, x_max, y_max)
    
    return roi_resized, roi_coords, circular_mask

def prepare_and_predict(model, image_path, transform_img, padding=20, target_size=(256, 256)):
    """
    Полный пайплайн для тестового изображения:
    1. Подготовка изображения (обрезка + resize)
    2. Применение трансформаций
    3. Предсказание модели
    4. Возврат маски и визуализаций
    """
    # Подготавливаем изображение
    prepared_img, roi_coords, circular_mask = prepare_single_image(
        image_path, padding=padding, target_size=target_size
    )
    
    # Конвертируем в PIL для трансформаций
    img_pil = Image.fromarray(prepared_img)
    img_tensor = transform_img(img_pil).unsqueeze(0)  # добавляем batch dimension

    # Предсказание
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor).squeeze(0)  # убираем batch
        prediction = torch.sigmoid(prediction)     # вероятности
    
    # Конвертируем в numpy для визуализации
    pred_mask = prediction.squeeze(0).cpu().numpy()  # убираем channel dimension
    
    return {
        'original_image': cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
        'prepared_image': prepared_img,
        'prediction': pred_mask,
        'circular_mask': circular_mask,
        'roi_coords': roi_coords
    }

def visualize_prediction(result, threshold=0.5):
    """
    Визуализирует результаты предсказания
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Оригинал с ROI
    axes[0, 0].imshow(result['original_image'])
    x_min, y_min, x_max, y_max = result['roi_coords']
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                         fill=False, edgecolor='red', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Оригинал с ROI')
    axes[0, 0].axis('off')
    
    # Круговая маска
    axes[0, 1].imshow(result['circular_mask'], cmap='gray')
    axes[0, 1].set_title('Круговая маска (служебная)')
    axes[0, 1].axis('off')
    
    # Подготовленное изображение
    axes[0, 2].imshow(result['prepared_image'])
    axes[0, 2].set_title(f'Подготовленное {result["prepared_image"].shape[:2]}')
    axes[0, 2].axis('off')
    
    # Предсказание (вероятности)
    axes[1, 0].imshow(result['prediction'], cmap='hot')
    axes[1, 0].set_title('Предсказание (вероятности)')
    axes[1, 0].axis('off')
    
    # Бинарное предсказание
    binary = (result['prediction'] > threshold).astype(np.uint8)
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f'Бинарное (порог={threshold})')
    axes[1, 1].axis('off')
    
    # Наложение на подготовленное
    prepared = result['prepared_image'].astype(np.float32) / 255.0
    
    # Проверяем размерность prepared
    if len(prepared.shape) == 3:
        # Если уже RGB
        overlay = prepared.copy()
    else:
        # Если grayscale, конвертируем в RGB
        overlay = np.stack([prepared, prepared, prepared], axis=-1)
    
    # Создаём маску для наложения (зелёный цвет)
    mask_overlay = binary > 0
    # Применяем зелёный цвет только к каналам, где маска True
    overlay[mask_overlay] = [0, 1, 0]  # зелёный в RGB
    
    axes[1, 2].imshow(np.clip(overlay, 0, 1))
    axes[1, 2].set_title('Наложение')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Пример использования:
if __name__ == "__main__":
    # Трансформации (те же, что при обучении)
    tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
    
    # Загружаем модель из папки models
    model = UNetModel()
    
    # Используем абсолютный путь на основе папки со скриптом
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'model_unet_RGB.tr')
    
    print(f"Ищем модель по пути: {model_path}")
    model.load_state_dict(torch.load(model_path))
    print(f"Модель загружена из {model_path}")
    
    # Тестируем на одном изображении
    test_image = os.path.join(script_dir, "dataset_seg", "processed", "id3_before_processed.jpg")
    print(f"Тестовое изображение: {test_image}")
    
    result = prepare_and_predict(
        model=model,
        image_path=test_image,
        transform_img=tr_img,
        padding=30,
        target_size=(256, 256)
    )
    
    visualize_prediction(result, threshold=0.5)