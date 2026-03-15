import os
import sys
import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.v2 as tfs_v2
import json

# Добавляем папку со скриптом в путь поиска модулей
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Теперь можно импортировать
from test_model import UNetModel
from model_LAB_kfold_napkin import UNetWithPretrainedEncoder

def simple_circular_mask_otsu_only(image):
    """Максимально простой метод выделения круга через Оцу"""
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

def prepare_single_image(image_path, padding=20, target_size=(256, 256), use_lab=False):
    """
    Обрабатывает одно тестовое изображение:
    1. Выделяет круговую маску через Otsu
    2. Обрезает по квадрату с отступом
    3. Ресайзит до target_size
    
    Returns:
        prepared_image: подготовленное изображение (numpy array)
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

def prepare_and_predict(model, image_path, transform_img, padding=20, target_size=(256, 256), use_lab=False):
    """
    Полный пайплайн для тестового изображения:
    1. Подготовка изображения (обрезка + resize)
    2. Применение трансформаций
    3. Предсказание модели
    4. Возврат маски и визуализаций
    """
    # Подготавливаем изображение
    prepared_img, roi_coords, circular_mask = prepare_single_image(
        image_path, padding=padding, target_size=target_size, use_lab=use_lab
    )
    
    # Конвертируем в PIL для трансформаций
    img_pil = Image.fromarray(prepared_img)
    
    if use_lab:
        # Конвертация RGB в LAB (как в обучении)
        img_np = np.array(img_pil) / 255.0
        img_lab = color.rgb2lab(img_np)
        
        # Нормализация
        img_lab_normalized = np.zeros_like(img_lab)
        img_lab_normalized[..., 0] = img_lab[..., 0] / 100.0
        img_lab_normalized[..., 1] = (img_lab[..., 1] + 128) / 255.0
        img_lab_normalized[..., 2] = (img_lab[..., 2] + 128) / 255.0
        
        # Обратно в PIL для трансформаций
        img_pil = Image.fromarray((img_lab_normalized * 255).astype(np.uint8))
    
    # Применяем трансформации
    img_tensor = transform_img(img_pil).unsqueeze(0)

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
        'roi_coords': roi_coords,
        'use_lab': use_lab
    }

def visualize_prediction(result, threshold=0.5, show_plot=True):
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
    axes[1, 0].imshow(result['prediction'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Предсказание (вероятности)')
    axes[1, 0].axis('off')
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)
    
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
    
    if show_plot:
        plt.show()
    
    return fig

def save_good_prediction(result, image_path, output_dir="good_predictions", threshold=0.5, min_confidence=0.8):
    """
    Сохраняет предсказание, если модель уверена в результате
    
    Args:
        result: результат от prepare_and_predict
        image_path: путь к исходному изображению
        output_dir: папка для сохранения
        threshold: порог бинаризации
        min_confidence: минимальная уверенность для сохранения (0-1)
    """
    # Создаём папки для сохранения
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Извлекаем имя файла без расширения
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Бинарная маска
    binary_mask = (result['prediction'] > threshold).astype(np.uint8)
    
    # Оценка уверенности (средняя вероятность в предсказанных сосудах)
    if binary_mask.sum() > 0:
        confidence = result['prediction'][binary_mask > 0].mean()
    else:
        confidence = 1 - result['prediction'].mean()  # уверенность в отсутствии сосудов
    
    print(f"Уверенность: {confidence:.3f}")
    
    # Сохраняем только если уверенность высокая
    if confidence >= min_confidence:
        # Сохраняем изображение
        img_filename = f"{base_name}_prepared.png"
        img_path = os.path.join(images_dir, img_filename)
        
        # Сохраняем подготовленное изображение
        prepared_bgr = cv2.cvtColor(result['prepared_image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, prepared_bgr)
        
        # Сохраняем маску
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        cv2.imwrite(mask_path, binary_mask * 255)
        
        # Сохраняем метаданные
        metadata = {
            'original_image': image_path,
            'confidence': float(confidence),
            'threshold': threshold,
            'use_lab': result.get('use_lab', False),
            'roi_coords': result['roi_coords']
        }
        
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Сохранено: {img_filename} и {mask_filename} (уверенность: {confidence:.3f})")
        return True
    else:
        print(f"⏭️ Пропущено: низкая уверенность ({confidence:.3f} < {min_confidence})")
        return False

# Функция для пакетной обработки нескольких изображений
def batch_process_images(model, image_paths, transform_img, output_dir="good_predictions", 
                         padding=30, target_size=(256, 256), use_lab=False, 
                         threshold=0.5, min_confidence=0.8, show_viz=False):
    """
    Обрабатывает несколько изображений и сохраняет удачные предсказания
    """
    results = []
    saved_count = 0
    
    for img_path in image_paths:
        print(f"\n--- Обработка: {os.path.basename(img_path)} ---")
        
        try:
            result = prepare_and_predict(
                model=model,
                image_path=img_path,
                transform_img=transform_img,
                padding=padding,
                target_size=target_size,
                use_lab=use_lab
            )
            
            if show_viz:
                visualize_prediction(result, threshold=threshold)
            
            # Сохраняем, если уверенность высокая
            if save_good_prediction(result, img_path, output_dir, threshold, min_confidence):
                saved_count += 1
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {img_path}: {e}")
    
    print(f"\n--- Итог: сохранено {saved_count} из {len(image_paths)} изображений ---")
    return results

# Пример использования:
if __name__ == "__main__":
    # Трансформации (те же, что при обучении)
    tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
    
    # Загружаем модель из папки models
    # model = UNetModel()
    model = UNetWithPretrainedEncoder(in_channels=3, num_classes=1, pretrained=False)
    
    # Используем абсолютный путь на основе папки со скриптом
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'model_best_overall_pretrained.pth')
    
    print(f"Ищем модель по пути: {model_path}")
    model.load_state_dict(torch.load(model_path))
    print(f"Модель загружена из {model_path}")
    
    # Параметры эксперимента
    EXPERIMENT = "LAB"  # или "RGB"
    use_lab = (EXPERIMENT == "LAB")
    
    print(f"\n🔬 Эксперимент: {EXPERIMENT}")
    
    if use_lab:
        print("🧪 Модель обучена на LAB, тестируем в LAB")
    else:
        print("🧪 Модель обучена на RGB, тестируем в RGB")
    
    # Тестируем на одном изображении
    test_image = os.path.join(script_dir, "dataset_seg", "processed", "id3_after_processed.jpg")
    print(f"\n📸 Тестовое изображение: {test_image}")
    
    result = prepare_and_predict(
        model=model,
        image_path=test_image,
        transform_img=tr_img,
        padding=30,
        target_size=(256, 256),
        use_lab=use_lab
    )
    
    visualize_prediction(result, threshold=0.5)
    
    # Для пакетной обработки неразмеченных изображений:
    # unlabeled_images = [
    #     "path/to/unlabeled1.jpg",
    #     "path/to/unlabeled2.jpg",
    #     # ...
    # ]
    # 
    # batch_process_images(
    #     model=model,
    #     image_paths=unlabeled_images,
    #     transform_img=tr_img,
    #     output_dir="good_predictions",
    #     use_lab=use_lab,
    #     min_confidence=0.85,
    #     show_viz=True
    # )