import os
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image
from skimage import color 

from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tfs_v2


class SegmentDataset(data.Dataset):
    def __init__(self, path, path_img='prepared_images', path_masks='prepared_masks', transform=None):
        """
        transform: функция, которая принимает (img, mask) и возвращает трансформированные (img, mask)
        """
        self.path = path
        self.transform = transform

        path = os.path.join(self.path, path_img)
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))

        path = os.path.join(self.path, path_masks)
        list_files = os.listdir(path)
        self.masks = list(map(lambda _x: os.path.join(path, _x), list_files))

    def __getitem__(self, item):
        '''
        Открывает картинки по известному пути, конвертирует, 
        применяет трансформации к изображению и маске,
        бинаризует маску
        '''
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')

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
            img, mask = self.transform(img, mask)
        
        # Бинаризация маски
        mask = (mask > 0.5).float()  # если маска уже тензор
        
        return img, mask

    def __len__(self):
        return self.length


class Compose:
    """Кастомный Compose для пар (img, mask)"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontalFlip:
    """Синхронный горизонтальный флип"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, mask):
        if random.random() < self.p:
            return tfs_v2.functional.horizontal_flip(img), tfs_v2.functional.horizontal_flip(mask)
        return img, mask


class RandomVerticalFlip:
    """Синхронный вертикальный флип"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, mask):
        if random.random() < self.p:
            return tfs_v2.functional.vertical_flip(img), tfs_v2.functional.vertical_flip(mask)
        return img, mask


class RandomRotation:
    """Синхронный поворот"""
    def __init__(self, degrees=15):
        self.degrees = degrees
    
    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        # Интерполяция: для изображения - билинейная, для маски - ближайшая (чтобы не смазывать)
        img = tfs_v2.functional.rotate(img, angle, interpolation=tfs_v2.InterpolationMode.BILINEAR)
        mask = tfs_v2.functional.rotate(mask, angle, interpolation=tfs_v2.InterpolationMode.NEAREST)
        return img, mask


class ToTensor:
    """Конвертирует PIL в тензор"""
    def __call__(self, img, mask):
        img = tfs_v2.functional.to_image(img)
        img = tfs_v2.functional.to_dtype(img, torch.float32, scale=True)
        
        mask = tfs_v2.functional.to_image(mask)
        mask = tfs_v2.functional.to_dtype(mask, torch.float32, scale=False)  # не масштабируем
        return img, mask


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

def create_file_pairs(path, img_folder='roi_images_prepared', mask_folder='roi_masks_prepared'):
    """Создаёт список пар (изображение, маска)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, path)
    
    img_path = os.path.join(dataset_path, img_folder)
    mask_path = os.path.join(dataset_path, mask_folder)
    
    # Получаем все файлы изображений
    img_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    file_pairs = []
    for img_file in img_files:
        # Предполагаем, что маска называется так же, но с _roi.png или просто в папке масок
        base_name = os.path.splitext(img_file)[0]
        
        # Возможные имена масок
        possible_masks = [
            f"{base_name}_roi.png",
            f"{base_name}.png",
            img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        ]
        
        mask_file = None
        for pm in possible_masks:
            if os.path.exists(os.path.join(mask_path, pm)):
                mask_file = pm
                break
        
        if mask_file:
            file_pairs.append((
                os.path.join(img_path, img_file),
                os.path.join(mask_path, mask_file)
            ))
        else:
            print(f"⚠️ Предупреждение: для {img_file} не найдена маска")
    
    return file_pairs

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


if __name__ == "__main__":   
    # Создаём синхронные трансформации
    train_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        ToTensor()
    ])

    # Для валидации/теста только тензоры (без аугментаций)
    val_transforms = Compose([
        ToTensor()
    ])

    # Получаем путь к папке, где находится скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset_seg")
    
    # Создаём датасет с синхронными трансформациями
    d_train = SegmentDataset(
        dataset_path, 
        path_img='roi_images_prepared', 
        path_masks='roi_masks_prepared', 
        transform=train_transforms
    )
    
    # Проверяем трансформации
    print("Проверка синхронных трансформаций:")
    visualize_augmentations(d_train, num_samples=3)
    
    # DataLoader
    train_data = data.DataLoader(d_train, batch_size=2, shuffle=True)
    
    # Модель и обучение
    model = UNetModel()
    optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
    loss_1 = nn.BCEWithLogitsLoss()
    loss_2 = SoftDiceLoss()

    epochs = 8
    model.train()

    for epoch in range(epochs):
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_1(predict, y_train) + loss_2(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1/lm_count*loss.item() + (1 - 1/lm_count)*loss_mean
            train_tqdm.set_description(f"Epoch [{epoch + 1}/{epochs}], loss_mean {loss_mean:.3f}")

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'model_unet_LAB.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в: {os.path.abspath(model_path)}")