import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import color 

from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tfs_v2


class SegmentDataset(data.Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        path = os.path.join(self.path, 'prepared_images')
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))

        path = os.path.join(self.path, 'prepared_masks')
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

        # Нормализуем для нейросети (обычно приводят к [0,1] или [-1,1])
        img_lab_normalized = np.zeros_like(img_lab)
        img_lab_normalized[..., 0] = img_lab[..., 0] / 100.0  # L в [0,1]
        img_lab_normalized[..., 1] = (img_lab[..., 1] + 128) / 255.0  # a в [0,1]
        img_lab_normalized[..., 2] = (img_lab[..., 2] + 128) / 255.0  # b в [0,1]

        # Конвертируем обратно в PIL.Image (ToImage ожидает PIL или numpy)
        img = Image.fromarray((img_lab_normalized * 255).astype(np.uint8))

        if self.transform_img:
            img = self.transform_img(img)
        
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask[mask < 250] = 0
            mask[mask >= 250] = 1
        
        return img, mask

    def __len__(self):
        return self.length



class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):  # in_channels - число каналов входного тензора, out_channels - число каналов в свёрточных слоях
            super().__init__()  # Эта строка вызывает конструктор родительского класса (nn.Module)
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), # отключаем bias, так как используем BatchNorm
                nn.ReLU(inplace=True),  # функция активации
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        
        def forward(self, x):   # Пропуск тензора x через эту модель
            return self.model(x)
    
    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2) # свёртка с выбором максимума, передавая один параметр, передаём одновременно kernel_size и stride

        def forward(self, x):
            x = self.block(x)   # x пойдёт по сквозной связи
            y = self.max_pool(x) # y пойдёт на слой ниже
            return y, x
    
    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # размерность тензора уменьшится в два раза
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)    # in_channels, т. к. после конкатенации размерность опять вдвое увеличится
            
        def forward(self, x, y):    # описываем логику обработки тензоров x и y (y приходит по сквозной связи)
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
        # logits - пиксели-выходы nn, targets - целевая маска
        num = targets.size(0)   # размер батча
        probs = nn.functional.sigmoid(logits)   # выход последнего свёрточного слоя не подвергается преобразованиям, а мы хотим интерпретировать пиксели как вероятности
        m1 = probs.view(num, -1)    # сплющиваем в одномерный вектор каждый элемент батча
        m2 = targets.view(num, -1)
        intersection = (m1*m2)  # поэлементное произведение?

        #score = (2 * intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        #score = 1 - score.sum() / num
        dice_coeff = (2 * intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        dice_loss = 1 - dice_coeff.mean()
        return dice_loss
        #return score

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


if __name__ == "__main__":   
    #tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
    #tr_mask = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32)])

    # Аугментации для изображений
    tr_img = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.RandomHorizontalFlip(p=0.5),
        tfs_v2.RandomVerticalFlip(p=0.5),
        tfs_v2.RandomRotation(degrees=15),
        tfs_v2.ToDtype(torch.float32, scale=True)
    ])

    # Для масок делаем те же геометрические трансформации (чтобы изображение и маска совпадали)
    tr_mask = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.RandomHorizontalFlip(p=0.5),
        tfs_v2.RandomVerticalFlip(p=0.5),
        tfs_v2.RandomRotation(degrees=15),
        tfs_v2.ToDtype(torch.float32)
    ])

    # Получаем путь к папке, где находится скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset_seg")
    d_train = SegmentDataset(dataset_path, transform_img=tr_img, transform_mask=tr_mask)
    train_data = data.DataLoader(d_train, batch_size=2, shuffle=True)

    # Вызываем функцию после создания датасета
    print("Проверка трансформаций на случайных примерах:")
    visualize_augmentations(d_train, num_samples=3)
    
    model = UNetModel()

    optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
    loss_1 = nn.BCEWithLogitsLoss()
    loss_2 = SoftDiceLoss()

    epochs = 20
    model.train()

    for _e in range(epochs):
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(train_data, leave=True)
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_1(predict, y_train) + loss_2(predict, y_train)

            optimizer.zero_grad()   # обнуление градиентов, чтобы веса корректировались на основании текущего батча
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1/lm_count*loss.item() + (1 - 1/lm_count)*loss_mean
            train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean {loss_mean:.3f}")

    st = model.state_dict()
    os.makedirs('models', exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join('models', 'model_unet_LAB.tr')
    torch.save(st, model_path)
    print("Модель сохраняется в:", os.path.abspath('model_unet_LAB.tr'))
