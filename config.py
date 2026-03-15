# config.py
"""
Конфигурационный файл для экспериментов с сегментацией
"""

# Параметры эксперимента
K_FOLDS = 3
EPOCHS = 100
RANDOM_STATE = 999
PATIENCE = 20
MIN_DELTA = 0.001

# Параметры модели
USE_PRETRAINED = False  # False для UNet без предобучения, True для ResNet18
IN_CHANNELS = 3  # 3 для RGB/LAB, 4 для LAB+extra

# Пути к данным
DATASET_PATH = "dataset_seg"
IMG_FOLDER = "combined_images_correction"
VESSEL_FOLDER = "combined_masks_correction"
SKIN_FOLDER = "roi_masks_prepared"

# Параметры обучения
BATCH_SIZE = 2
LEARNING_RATE = 0.001
USE_MASKED_LOSS = False  # Использовать ли MaskedCombinedLoss

# Визуализация
SAVE_PREDICTIONS = True
NUM_VISUALIZATION_SAMPLES = 2