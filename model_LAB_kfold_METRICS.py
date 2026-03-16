import os
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from PIL import Image
from skimage import color 

from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tfs_v2
import torchvision.models as models

import config

class SegmentDataset(data.Dataset):
    def __init__(self, file_triplets, transform=None):
        """
        file_triplets: список кортежей (путь_к_изображению, путь_к_маске_сосудов, путь_к_маске_кожи)
        """
        self.file_triplets = file_triplets
        self.transform = transform
        self.length = len(file_triplets)

    def __getitem__(self, item):
        '''
        Открывает картинки по известному пути, конвертирует, 
        применяет трансформации к изображению и маске,
        бинаризует маску
        '''
        path_img, path_vessel_mask, path_skin_mask = self.file_triplets[item]
        img = Image.open(path_img).convert('RGB')
        vessel_mask = Image.open(path_vessel_mask).convert('L')
        skin_mask = Image.open(path_skin_mask).convert('L')

        # Конвертация RGB в LAB
        img_np = np.array(img) / 255.0  # нормализуем в [0, 1]
        img_lab = color.rgb2lab(img_np)  # LAB: L [0,100], a[-128,128], b[-128,128]

        # Нормализуем для нейросети
        img_lab_normalized = np.zeros_like(img_lab)
        img_lab_normalized[..., 0] = img_lab[..., 0] / 100.0  # L в [0,1]
        img_lab_normalized[..., 1] = (img_lab[..., 1] + 128) / 255.0  # a в [0,1]
        img_lab_normalized[..., 2] = (img_lab[..., 2] + 128) / 255.0  # b в [0,1]

        # Конвертируем обратно в PIL.Image
        img = Image.fromarray((img_lab_normalized * 255).astype(np.uint8))

        # ПРИМЕНЯЕМ ТРАНСФОРМАЦИИ К ПАРЕ (img, mask)
        if self.transform:
            img, vessel_mask, skin_mask = self.transform(img, vessel_mask, skin_mask)
        
        # Бинаризация маски
        vessel_mask = (vessel_mask > 0.5).float()  # если маска уже тензор
        skin_mask = (skin_mask > 0.5).float()
        
        return img, vessel_mask, skin_mask

    def __len__(self):
        return self.length


class Compose:
    """Кастомный Compose для пар (img, mask)"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, vessel_mask, skin_mask):
        for t in self.transforms:
            img, vessel_mask, skin_mask = t(img, vessel_mask, skin_mask)
        return img, vessel_mask, skin_mask


class RandomHorizontalFlip:
    """Синхронный горизонтальный флип"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, vessel_mask, skin_mask):
        if random.random() < self.p:
            return (tfs_v2.functional.horizontal_flip(img), 
                    tfs_v2.functional.horizontal_flip(vessel_mask),
                    tfs_v2.functional.horizontal_flip(skin_mask))
        return img, vessel_mask, skin_mask


class RandomVerticalFlip:
    """Синхронный вертикальный флип"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, vessel_mask, skin_mask):
        if random.random() < self.p:
            return (tfs_v2.functional.vertical_flip(img), 
                    tfs_v2.functional.vertical_flip(vessel_mask),
                    tfs_v2.functional.vertical_flip(skin_mask))
        return img, vessel_mask, skin_mask


class RandomRotation:
    """Синхронный поворот для трёх объектов"""
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, img, vessel_mask, skin_mask):
        angle = random.uniform(-self.degrees, self.degrees)
        return (tfs_v2.functional.rotate(img, angle, interpolation=tfs_v2.InterpolationMode.BILINEAR),
                tfs_v2.functional.rotate(vessel_mask, angle, interpolation=tfs_v2.InterpolationMode.NEAREST),
                tfs_v2.functional.rotate(skin_mask, angle, interpolation=tfs_v2.InterpolationMode.NEAREST))


class ToTensor:
    """Конвертирует PIL в тензор для трёх объектов"""
    def __call__(self, img, vessel_mask, skin_mask):
        img = tfs_v2.functional.to_image(img)
        img = tfs_v2.functional.to_dtype(img, torch.float32, scale=True)
        
        vessel_mask = tfs_v2.functional.to_image(vessel_mask)
        vessel_mask = tfs_v2.functional.to_dtype(vessel_mask, torch.float32, scale=False)
        
        skin_mask = tfs_v2.functional.to_image(skin_mask)
        skin_mask = tfs_v2.functional.to_dtype(skin_mask, torch.float32, scale=False)
        
        return img, vessel_mask, skin_mask


# Визуализация для проверки трансформаций
def visualize_augmentations(dataset, num_samples=3):
    """
    Показывает несколько примеров из датасета с применёнными трансформациями
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    
    for i in range(num_samples):
        # Берём случайный индекс
        idx = np.random.randint(len(dataset))
        img, mask = dataset[idx]  # применяются трансформации!
        
        # Конвертируем тензоры в numpy для визуализации
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        
        # Изображение
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Изображение {i+1}')
        axes[i, 0].axis('off')
        
        # Маска
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Маска {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        
        def forward(self, x):
            return self.model(x)
    
    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x
    
    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            
        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)
            return u
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)
        
        return self.out(x)


import torchvision.models as models

class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super().__init__()
        
        # Загружаем предобученный ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Энкодер - берём слои до avgpool
        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # 64 канала
        self.enc2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,  # 64 канала
        )
        self.enc3 = resnet.layer2  # 128 каналов
        self.enc4 = resnet.layer3  # 256 каналов
        self.enc5 = resnet.layer4  # 512 каналов
        
        # Декодер (в соответствии со схемой)
        self.dec4 = DecoderBlock(768, 256)   # 512(enc5) + 256(enc4) = 768 -> 256
        self.dec3 = DecoderBlock(384, 128)   # 256 + 128(enc3) = 384 -> 128
        self.dec2 = DecoderBlock(192, 64)    # 128 + 64(enc2) = 192 -> 64
        self.dec1 = DecoderBlock(128, 32)    # 64 + 64(enc1) = 128 -> 32
        self.dec0 = DecoderBlock(32, 16)     # 32 -> 16
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Энкодер
        e1 = self.enc1(x)      # 64 канала
        e2 = self.enc2(e1)     # 64 канала
        e3 = self.enc3(e2)     # 128 каналов
        e4 = self.enc4(e3)     # 256 каналов
        e5 = self.enc5(e4)     # 512 каналов
        
        # Декодер с апскейлингом
        d4 = self.dec4(e5, e4)  # 512 + 256 = 768 -> 256
        d3 = self.dec3(d4, e3)  # 256 + 128 = 384 -> 128
        d2 = self.dec2(d3, e2)  # 128 + 64 = 192 -> 64
        d1 = self.dec1(d2, e1)  # 64 + 64 = 128 -> 32
        d0 = self.dec0(d1)       # 32 -> 16
        
        return self.final_conv(d0)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Вход: либо конкатенированные тензоры, либо просто тензор
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip=None):
        # Апскейлинг
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Конкатенация со skip connection, если есть
        if skip is not None:
            # Если размеры не совпадают, подгоняем
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2).sum(dim=1)
        
        dice_coeff = (2 * intersection + self.smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice_coeff.mean()
        return dice_loss

# ================== НОВЫЕ МЕТРИКИ ==================
def dice_coefficient(pred, target, smooth=1e-6):
    """Dice coefficient для бинарной сегментации"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    """IoU (Jaccard index)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target, smooth=1e-6):
    """Точность (Precision): TP / (TP + FP)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision

def recall_score(pred, target, smooth=1e-6):
    """Полнота (Recall): TP / (TP + FN)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

def specificity_score(pred, target, smooth=1e-6):
    """Специфичность: TN / (TN + FP)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    return specificity

# ================== МАСКИРОВАННЫЕ МЕТРИКИ ==================
def dice_coefficient_masked(probs, vessel_targets, skin_masks, smooth=1e-6):
    """
    Dice coefficient только по области кожи
    """
    pred = (probs > 0.5).float()
    
    masked_pred = pred * skin_masks
    masked_target = vessel_targets * skin_masks
    
    num = vessel_targets.size(0)
    m1 = masked_pred.view(num, -1)
    m2 = masked_target.view(num, -1)
    
    intersection = (m1 * m2).sum(dim=1)
    
    dice = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
    return dice.mean()

def iou_score_masked(probs, vessel_targets, skin_masks, smooth=1e-6):
    """
    IoU (Jaccard index) только по области кожи
    """
    pred = (probs > 0.5).float()
    
    masked_pred = pred * skin_masks
    masked_target = vessel_targets * skin_masks
    
    num = vessel_targets.size(0)
    m1 = masked_pred.view(num, -1)
    m2 = masked_target.view(num, -1)
    
    intersection = (m1 * m2).sum(dim=1)
    union = m1.sum(dim=1) + m2.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

# ================== МАСКИРОВАННЫЕ ФУНКЦИИ ПОТЕРЬ ==================
class MaskedDiceLoss(nn.Module):
    """Dice Loss, который считается только по области кожи"""
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, vessel_targets, skin_masks):
        probs = torch.sigmoid(logits)

        # Применяем маску к предсказаниям и истине
        masked_probs = probs * skin_masks
        masked_targets = vessel_targets * skin_masks

        # Сглаживаем для подсчёта Dice
        num = vessel_targets.size(0)
        m1 = masked_probs.view(num, -1)
        m2 = masked_targets.view(num, -1)

        intersection = (m1 * m2).sum(dim=1)

        dice_coeff = (2 * intersection + self.smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice_coeff.mean()
        return dice_loss
    
class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE Loss, который считается только по области кожи"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Без усреднения
    
    def forward(self, logits, vessel_targets, skin_masks):
        # Считаем BCE для всех пикселей
        loss_per_pixel = self.bce(logits, vessel_targets)
        
        # Обнуляем потери вне кожи и усредняем только по области кожи
        masked_loss = loss_per_pixel * skin_masks
        
        # Усредняем по всем пикселям, где есть кожа
        num_skin_pixels = skin_masks.sum() + 1e-8  # избегаем деления на 0
        return masked_loss.sum() / num_skin_pixels
    
 
class MaskedCombinedLoss(nn.Module):
    """Комбинация BCE и Dice с маской кожи"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = MaskedBCEWithLogitsLoss()
        self.dice_loss = MaskedDiceLoss(smooth=smooth)
    
    def forward(self, logits, vessel_targets, skin_masks):
        bce = self.bce_loss(logits, vessel_targets, skin_masks)
        dice = self.dice_loss(logits, vessel_targets, skin_masks)
        return self.bce_weight * bce + self.dice_weight * dice   
  

class ImprovedMaskedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, penalty_weight=0.1, smooth=1):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.penalty_weight = penalty_weight
        self.bce_loss = MaskedBCEWithLogitsLoss()
        self.dice_loss = MaskedDiceLoss(smooth=smooth)
    
    def forward(self, logits, vessel_targets, skin_masks):
        # Основные потери на коже (как раньше)
        bce = self.bce_loss(logits, vessel_targets, skin_masks)
        dice = self.dice_loss(logits, vessel_targets, skin_masks)
        
        # Штраф за предсказания вне кожи
        probs = torch.sigmoid(logits)
        outside_skin = (1 - skin_masks)  # область тени (1 там, где нет кожи)
        
        # Средняя уверенность модели в том, что там сосуды (чем выше, тем больше штраф)
        penalty = (probs * outside_skin).sum() / (outside_skin.sum() + 1e-8)
        
        return self.bce_weight * bce + self.dice_weight * dice + self.penalty_weight * penalty


def create_file_pairs(path, img_folder='roi_images_prepared', mask_folder='roi_masks_prepared'):
    """Создаёт список кортежей (путь_к_изображению, путь_к_маске)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, path)
    
    img_path = os.path.join(dataset_path, img_folder)
    mask_path = os.path.join(dataset_path, mask_folder)
    
    print(f"Поиск изображений в: {img_path}")
    print(f"Поиск масок в: {mask_path}")
    
    # Проверяем существование папок
    if not os.path.exists(img_path):
        print(f"❌ Папка не найдена: {img_path}")
        return []
    if not os.path.exists(mask_path):
        print(f"❌ Папка не найдена: {mask_path}")
        return []
    
    # Получаем все файлы изображений
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Найдено изображений: {len(img_files)}")
    
    file_pairs = []
    for img_file in img_files:
        img_full_path = os.path.join(img_path, img_file)
        
        # Ищем соответствующую маску
        base_name = os.path.splitext(img_file)[0]
        
        # Возможные имена масок (ВАЖНО: заменяем _roi на _mask)
        possible_masks = [
            f"{base_name}_mask.png",  # Теперь ищем с _mask
            f"{base_name}.png",
            img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        ]
        
        mask_found = False
        for pm in possible_masks:
            mask_full_path = os.path.join(mask_path, pm)
            if os.path.exists(mask_full_path):
                file_pairs.append((img_full_path, mask_full_path))
                mask_found = True
                print(f"✓ Найдено: {img_file} -> {pm}")  # Для отладки
                break
        
        if not mask_found:
            print(f"⚠️ Для {img_file} не найдена маска (искали: {possible_masks})")
    
    print(f"Создано {len(file_pairs)} пар изображение-маска")
    return file_pairs

def create_file_triplets(path, 
                         img_folder='combined_images_prepared', 
                         vessel_folder='combined_masks_prepared',
                         skin_folder='roi_masks_prepared'):
    """
    Создаёт список кортежей (путь_к_изображению, путь_к_маске_сосудов, путь_к_маске_кожи)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, path)
    
    img_path = os.path.join(dataset_path, img_folder)
    vessel_path = os.path.join(dataset_path, vessel_folder)
    skin_path = os.path.join(dataset_path, skin_folder)
    
    print(f"Поиск изображений в: {img_path}")
    print(f"Поиск масок сосудов в: {vessel_path}")
    print(f"Поиск масок кожи в: {skin_path}")
    
    # Проверяем существование папок
    for p, name in [(img_path, 'изображения'), (vessel_path, 'маски сосудов'), (skin_path, 'маски кожи')]:
        if not os.path.exists(p):
            print(f"❌ Папка не найдена: {p}")
            return []
    
    # Получаем все файлы изображений
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Найдено изображений: {len(img_files)}")
    
    file_triplets = []
    for img_file in img_files:
        img_full_path = os.path.join(img_path, img_file)
        
        # Извлекаем базовое имя без расширения
        base_name = os.path.splitext(img_file)[0]
        
        # Ищем маску сосудов
        # Пробуем разные варианты
        vessel_candidates = [
            f"{base_name}.png",
            f"{base_name}_mask.png",
        ]
        
        vessel_found = False
        for vs in vessel_candidates:
            vessel_full_path = os.path.join(vessel_path, vs)
            if os.path.exists(vessel_full_path):
                vessel_found = True
                break
        
        if not vessel_found:
            print(f"⚠️ Для {img_file} не найдена маска кожи")
            continue
        
        # Ищем маску кожи
        # Пробуем разные варианты
        skin_candidates = [
            f"{base_name}.png",
            f"{base_name}_mask.png",
            f"skin_{base_name}.png"
        ]
        
        skin_found = False
        for sc in skin_candidates:
            skin_full_path = os.path.join(skin_path, sc)
            if os.path.exists(skin_full_path):
                skin_found = True
                break
        
        if not skin_found:
            print(f"⚠️ Для {img_file} не найдена маска кожи")
            continue
        
        # Если всё нашлось - добавляем тройку
        file_triplets.append((img_full_path, vessel_full_path, skin_full_path))
        print(f"✓ Найдено: {img_file}")
    
    print(f"\nСоздано {len(file_triplets)} троек из {len(img_files)} изображений")
    return file_triplets


def get_image_number(filename):
    """
    Извлекает number из filename вида '{number}-{subnumber}'
    Например: '17-3.png' -> 17
    """
    basename = os.path.basename(filename)
    # Убираем расширение
    name_without_ext = os.path.splitext(basename)[0]
    # Разделяем по дефису и берём первую часть
    try:
        number = int(name_without_ext.split('-')[0])
        return number
    except (ValueError, IndexError):
        print(f"⚠️ Не удалось извлечь номер из {basename}")
        return None

def filter_triplets_by_number(triplets, min_number=17, mode='ge'):
    """
    Фильтрует триплеты по номеру изображения
    
    mode: 'ge' - больше или равно (>= min_number)
          'lt' - меньше (< min_number)
    """
    filtered = []
    for img_path, vessel_path, skin_path in triplets:
        number = get_image_number(img_path)
        if number is None:
            continue
            
        if mode == 'ge' and number >= min_number:
            filtered.append((img_path, vessel_path, skin_path))
        elif mode == 'lt' and number < min_number:
            filtered.append((img_path, vessel_path, skin_path))
    
    return filtered


def visualize_triplet_rgb_transformed(dataset, idx=0):
    """Визуализирует RGB изображение (конвертированное обратно из LAB) и маски после трансформаций"""
    
    # Получаем всё из датасета (изображение уже в LAB, после трансформаций)
    img_lab_tensor, vessel_mask, skin_mask = dataset[idx]
    
    # Конвертируем LAB тензор обратно в RGB для визуализации
    img_lab_np = img_lab_tensor.permute(1, 2, 0).numpy()
    
    # Денормализуем LAB (обратно в диапазоны LAB)
    img_lab_denorm = np.zeros_like(img_lab_np)
    img_lab_denorm[..., 0] = img_lab_np[..., 0] * 100.0  # L back to [0, 100]
    img_lab_denorm[..., 1] = img_lab_np[..., 1] * 255.0 - 128  # a back to [-128, 128]
    img_lab_denorm[..., 2] = img_lab_np[..., 2] * 255.0 - 128  # b back to [-128, 128]
    
    # Конвертируем LAB -> RGB
    img_rgb_np = color.lab2rgb(img_lab_denorm)
    
    # Маски уже в numpy
    vessel_np = vessel_mask.squeeze().numpy()
    skin_np = skin_mask.squeeze().numpy()
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb_np)
    axes[0].set_title('Изображение (RGB, после трансформаций)')
    axes[0].axis('off')
    
    axes[1].imshow(vessel_np, cmap='gray')
    axes[1].set_title('Маска сосудов')
    axes[1].axis('off')
    
    axes[2].imshow(skin_np, cmap='gray')
    axes[2].set_title('Маска кожи')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_rgb_np, vessel_np, skin_np

# ================== НОВАЯ ФУНКЦИЯ ВИЗУАЛИЗАЦИИ ПРЕДСКАЗАНИЙ ==================
def visualize_predictions_with_errors(model, val_loader, device='cpu', num_samples=3, save_path=None):
    """
    Показывает предсказания и раскрашивает ошибки:
    - Зелёный: True Positive (правильно найденные сосуды)
    - Красный: False Positive (ложные срабатывания)
    - Синий: False Negative (пропущенные сосуды)
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    # Если всего один sample, делаем axes 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        samples_shown = 0
        for batch_idx, (x_val, y_val_vessel, _) in enumerate(val_loader):
            if samples_shown >= num_samples:
                break
            
            x_val = x_val.to(device)
            y_val_vessel = y_val_vessel.to(device)
            
            # Предсказание
            predict = model(x_val)
            probs = torch.sigmoid(predict)
            pred_mask = (probs > 0.5).float()
            
            # Для каждого изображения в батче
            for i in range(min(x_val.size(0), num_samples - samples_shown)):
                # Конвертируем в numpy для визуализации
                img = x_val[i].cpu().permute(1, 2, 0).numpy()
                true = y_val_vessel[i].cpu().squeeze().numpy()
                pred = pred_mask[i].cpu().squeeze().numpy()
                
                # Возвращаем из LAB в RGB для показа
                img_lab_denorm = np.zeros_like(img)
                img_lab_denorm[..., 0] = img[..., 0] * 100.0
                img_lab_denorm[..., 1] = img[..., 1] * 255.0 - 128
                img_lab_denorm[..., 2] = img[..., 2] * 255.0 - 128
                img_rgb = color.lab2rgb(img_lab_denorm)
                
                # Создаём карту ошибок
                error_map = np.zeros((*true.shape, 3))
                # TP - зелёный (истина и предсказание совпали = 1)
                tp = (true == 1) & (pred == 1)
                # FP - красный (ложные срабатывания)
                fp = (true == 0) & (pred == 1)
                # FN - синий (пропуски)
                fn = (true == 1) & (pred == 0)
                
                error_map[tp] = [0, 1, 0]  # зелёный
                error_map[fp] = [1, 0, 0]  # красный
                error_map[fn] = [0, 0, 1]  # синий
                
                # Метрики для этого примера
                dice_val = dice_coefficient(probs[i:i+1], y_val_vessel[i:i+1]).item()
                prec_val = precision_score(probs[i:i+1], y_val_vessel[i:i+1]).item()
                rec_val = recall_score(probs[i:i+1], y_val_vessel[i:i+1]).item()
                spec_val = specificity_score(probs[i:i+1], y_val_vessel[i:i+1]).item()
                
                # Визуализация
                row = samples_shown
                axes[row, 0].imshow(img_rgb)
                axes[row, 0].set_title('Исходное изображение')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(true, cmap='gray')
                axes[row, 1].set_title('Истина (врач)')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(pred, cmap='gray')
                axes[row, 2].set_title('Предсказание модели')
                axes[row, 2].axis('off')
                
                axes[row, 3].imshow(img_rgb)
                axes[row, 3].imshow(error_map, alpha=0.5)
                axes[row, 3].set_title('Ошибки: Зел-TP, Крас-FP, Син-FN')
                axes[row, 3].axis('off')
                
                # Подпишем метрики для этого примера
                axes[row, 0].set_ylabel(f'Dice: {dice_val:.3f}\nPrec: {prec_val:.3f}\nRec: {rec_val:.3f}\nSpec: {spec_val:.3f}', 
                                      fontsize=10, rotation=0, labelpad=60)
                
                samples_shown += 1
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ================== ОБНОВЛЁННЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ ==================
def plot_training_history(history, fold_num):
    """Строит графики обучения для одного фолда (с новыми метриками)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = history['epoch']
    
    # График потерь
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Фолд {fold_num}: Функция потерь')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График Dice
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].set_title(f'Фолд {fold_num}: Dice Coefficient')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График IoU
    axes[0, 2].plot(epochs, history['val_iou'], 'g-', label='Val IoU', linewidth=2)
    axes[0, 2].set_xlabel('Эпоха')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].set_title(f'Фолд {fold_num}: IoU (Jaccard)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # График Precision
    axes[1, 0].plot(epochs, history['val_precision'], 'purple', label='Val Precision', linewidth=2)
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title(f'Фолд {fold_num}: Precision (Точность)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # График Recall и Specificity
    axes[1, 1].plot(epochs, history['val_recall'], 'brown', label='Val Recall', linewidth=2)
    axes[1, 1].plot(epochs, history['val_specificity'], 'orange', label='Val Specificity', linewidth=2)
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('Значение')
    axes[1, 1].set_title(f'Фолд {fold_num}: Recall и Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Совмещённый график всех метрик
    axes[1, 2].plot(epochs, history['train_dice'], 'b--', label='Train Dice', alpha=0.5)
    axes[1, 2].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[1, 2].plot(epochs, history['val_iou'], 'g-', label='Val IoU', linewidth=2)
    axes[1, 2].plot(epochs, history['val_precision'], 'purple', label='Val Precision', linewidth=2)
    axes[1, 2].plot(epochs, history['val_recall'], 'brown', label='Val Recall', linewidth=2)
    axes[1, 2].plot(epochs, history['val_specificity'], 'orange', label='Val Specificity', linewidth=2)
    axes[1, 2].set_xlabel('Эпоха')
    axes[1, 2].set_ylabel('Значение')
    axes[1, 2].set_title(f'Фолд {fold_num}: Все метрики')
    axes[1, 2].legend(loc='upper right', fontsize='small')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_fold{fold_num}.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_fold_comparison(fold_results, K_FOLDS):
    """Строит график сравнения результатов по фолдам (с новыми метриками)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    folds = np.arange(1, K_FOLDS + 1)
    
    # График 1: Dice и IoU
    width = 0.35
    axes[0].bar(folds - width/2, fold_results['dice'], width, label='Dice', color='skyblue')
    axes[0].bar(folds + width/2, fold_results['iou'], width, label='IoU', color='lightcoral')
    axes[0].set_xlabel('Фолд')
    axes[0].set_ylabel('Значение')
    axes[0].set_title('Dice и IoU по фолдам')
    axes[0].set_xticks(folds)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # График 2: Precision, Recall, Specificity
    axes[1].bar(folds - width, fold_results['precision'], width, label='Precision', color='purple')
    axes[1].bar(folds, fold_results['recall'], width, label='Recall', color='brown')
    axes[1].bar(folds + width, fold_results['specificity'], width, label='Specificity', color='orange')
    axes[1].set_xlabel('Фолд')
    axes[1].set_ylabel('Значение')
    axes[1].set_title('Precision, Recall, Specificity по фолдам')
    axes[1].set_xticks(folds)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # График 3: Ящик с усами для всех метрик
    data_to_plot = [
        fold_results['dice'], 
        fold_results['iou'], 
        fold_results['precision'], 
        fold_results['recall'], 
        fold_results['specificity']
    ]
    bp = axes[2].boxplot(data_to_plot, labels=['Dice', 'IoU', 'Prec', 'Recall', 'Spec'], patch_artist=True)
    colors = ['skyblue', 'lightcoral', 'purple', 'brown', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[2].set_ylabel('Значение')
    axes[2].set_title('Распределение метрик по фолдам')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fold_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Вывод статистики
    print(f"\n{'='*80}")
    print(f"ИТОГИ {K_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*80}")
    for i in range(K_FOLDS):
        print(f"  Фолд {i+1}:")
        print(f"    Dice={fold_results['dice'][i]:.4f}, IoU={fold_results['iou'][i]:.4f}")
        print(f"    Precision={fold_results['precision'][i]:.4f}, Recall={fold_results['recall'][i]:.4f}, Specificity={fold_results['specificity'][i]:.4f}")
    print(f"\n  Средние значения:")
    print(f"    Dice: {np.mean(fold_results['dice']):.4f} ± {np.std(fold_results['dice']):.4f}")
    print(f"    IoU:  {np.mean(fold_results['iou']):.4f} ± {np.std(fold_results['iou']):.4f}")
    print(f"    Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
    print(f"    Recall: {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
    print(f"    Specificity: {np.mean(fold_results['specificity']):.4f} ± {np.std(fold_results['specificity']):.4f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    
    # Параметры
    K_FOLDS = config.K_FOLDS
    EPOCHS = config.EPOCHS
    RANDOM_STATE = config.RANDOM_STATE
    PATIENCE = config.PATIENCE
    MIN_DELTA = config.MIN_DELTA
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # СОЗДАЁМ ВСЕ ТРОЙКИ
    all_triplets = create_file_triplets("dataset_seg", 
                                        img_folder='combined_images_correction', 
                                        vessel_folder='combined_masks_correction',
                                        skin_folder='roi_masks_prepared')
    
    print(f"{'='*60}")
    print(f"ВСЕГО НАЙДЕНО ТРОЕК: {len(all_triplets)}")
    
    if len(all_triplets) == 0:
        print("❌ Нет данных для обучения!")
        exit()
    
    # ФИЛЬТРАЦИЯ ПО ТИПУ ИЗОБРАЖЕНИЙ
    print(f"\n{'='*60}")
    print("ФИЛЬТРАЦИЯ ИЗОБРАЖЕНИЙ")
    print(f"{'='*60}")
    
    napkin_triplets = filter_triplets_by_number(all_triplets, min_number=17, mode='ge')
    print(f"С салфеткой (number >= 17): {len(napkin_triplets)} изображений")
    
    no_napkin_triplets = filter_triplets_by_number(all_triplets, min_number=17, mode='lt')
    print(f"Без салфетки (number < 17): {len(no_napkin_triplets)} изображений")
    
    active_triplets = no_napkin_triplets
    
    print(f"\nОбучаемся на {len(active_triplets)} изображениях")
    print(f"{'='*60}")
    
    # Инициализируем K-Fold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Для хранения результатов
    fold_results = {
        'dice': [], 
        'iou': [], 
        'precision': [], 
        'recall': [], 
        'specificity': []
    }
    fold_history = []
    best_models = []  # Для сохранения лучших моделей для визуализации
    
    # Цикл по фолдам
    for fold, (train_idx, val_idx) in enumerate(kfold.split(active_triplets)):
        print(f"\n{'='*60}")
        print(f"ФОЛД {fold + 1}/{K_FOLDS}")
        print(f"{'='*60}")
        
        # Разделяем данные
        train_triplets = [active_triplets[i] for i in train_idx]
        val_triplets = [active_triplets[i] for i in val_idx]
        
        # Датасеты и загрузчики
        train_transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(degrees=10),
                ToTensor()
        ])
        val_transforms = Compose([ToTensor()])
        train_dataset = SegmentDataset(train_triplets, transform=train_transforms)
        val_dataset = SegmentDataset(val_triplets, transform=val_transforms)
        train_loader = data.DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Модель и оптимизация
        model = UNetWithPretrainedEncoder(in_channels=3, num_classes=1, pretrained=True)
        model = model.to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=5, 
                                                         verbose=True, min_lr=1e-6)
        
        loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = SoftDiceLoss()
        
        # Для хранения истории эпох (ОБНОВЛЕНО: добавили новые метрики)
        history = {
            'epoch': [],
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': [], 'val_iou': [],
            'val_precision': [], 'val_recall': [], 'val_specificity': []
        }
        
        best_val_dice = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None 
        
        for epoch in range(EPOCHS):
            # ===== TRAIN =====
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            
            train_tqdm = tqdm(train_loader, desc=f"Фолд {fold+1}/{K_FOLDS} Эпоха {epoch+1}/{EPOCHS} [Train]")
            for x_train, y_train_vessel, _ in train_tqdm:
                x_train = x_train.to(device)
                y_train_vessel = y_train_vessel.to(device)
                
                optimizer.zero_grad()
                predict = model(x_train)
                loss = loss_fn(predict, y_train_vessel) + dice_loss_fn(predict, y_train_vessel)
                loss.backward()
                optimizer.step()
                
                probs = torch.sigmoid(predict)
                dice = dice_coefficient(probs, y_train_vessel)
                
                train_loss += loss.item()
                train_dice += dice.item()
                train_tqdm.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
            
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            # ===== VALIDATION (ОБНОВЛЕНО: добавили новые метрики) =====
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_specificity = 0.0
            
            with torch.no_grad():
                for x_val, y_val_vessel, _ in val_loader:
                    x_val = x_val.to(device)
                    y_val_vessel = y_val_vessel.to(device)
                    
                    predict = model(x_val)
                    loss = loss_fn(predict, y_val_vessel) + dice_loss_fn(predict, y_val_vessel)
                    
                    probs = torch.sigmoid(predict)
                    dice = dice_coefficient(probs, y_val_vessel)
                    iou = iou_score(probs, y_val_vessel)
                    precision = precision_score(probs, y_val_vessel)
                    recall = recall_score(probs, y_val_vessel)
                    specificity = specificity_score(probs, y_val_vessel)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_iou += iou.item()
                    val_precision += precision.item()
                    val_recall += recall.item()
                    val_specificity += specificity.item()
            
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            val_iou /= len(val_loader)
            val_precision /= len(val_loader)
            val_recall /= len(val_loader)
            val_specificity /= len(val_loader)
            
            # Сохраняем историю (ОБНОВЛЕНО)
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_specificity'].append(val_specificity)
            
            print(f"\n  Фолд {fold+1}/{K_FOLDS} | Эпоха {epoch+1}/{EPOCHS}")
            print(f"    Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"    Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"            Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, Spec: {val_specificity:.4f}")
            
            # Обновляем learning rate scheduler
            scheduler.step(val_dice)
            
            # Early stopping (ОБНОВЛЕНО: сохраняем все лучшие метрики)
            if val_dice > best_val_dice + MIN_DELTA:
                best_val_dice = val_dice
                best_val_iou = val_iou
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_val_specificity = val_specificity
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict()
                
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), f'models/model_fold{fold+1}_pretrained_{RANDOM_STATE}.pth')
                print(f"    ✅ Сохранена лучшая модель (Dice: {best_val_dice:.4f})")
            else:
                patience_counter += 1
                print(f"    ⏳ Patience: {patience_counter}/{PATIENCE}")
                
                if patience_counter >= PATIENCE:
                    print(f"    🛑 Early stopping на эпохе {epoch+1}")
                    break
        
        # Сохраняем лучшую модель с timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        torch.save(best_model_state, f'models/model_fold{fold+1}_pretrained_{RANDOM_STATE}_{timestamp}.pth')
        fold_history.append(history)
        best_models.append((model, best_model_state, val_loader))  # Сохраняем для визуализации
        
        # Сохраняем лучшие результаты фолда (ОБНОВЛЕНО)
        fold_results['dice'].append(best_val_dice)
        fold_results['iou'].append(best_val_iou)
        fold_results['precision'].append(best_val_precision)
        fold_results['recall'].append(best_val_recall)
        fold_results['specificity'].append(best_val_specificity)
        
        # Рисуем графики для текущего фолда
        plot_training_history(history, fold+1)
    
    # Финальные результаты и сводный график
    plot_fold_comparison(fold_results, K_FOLDS)
    
    # ================== ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ==================
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ЛУЧШИХ МОДЕЛЕЙ")
    print(f"{'='*60}")
    
    # Для каждого фолда покажем предсказания лучшей модели
    for fold, (model, best_state, val_loader) in enumerate(best_models):
        print(f"\nФолд {fold+1}:")
        model.load_state_dict(best_state)
        model.eval()
        visualize_predictions_with_errors(
            model, 
            val_loader, 
            device=device, 
            num_samples=2,  # Покажем по 2 примера на фолд
            save_path=f'predictions_fold{fold+1}.png'
        )
