import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Константы с путями до изображения и весов модели
IMAGE_PATH = "sample.png"
WEIGHTS_PATH = "mnist_cnn.pt"


class Net(nn.Module):
    """Простая CNN как в main.py для классификации MNIST."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)  # первый сверточный слой
        self.c2 = nn.Conv2d(32, 64, 3)  # второй сверточный слой
        self.p = nn.MaxPool2d(2)  # слой подвыборки
        self.d = nn.Dropout(0.25)  # dropout для регуляризации
        self.f1 = nn.Linear(64 * 12 * 12, 128)  # полносвязный слой
        self.f2 = nn.Linear(128, 10)  # выходной слой на 10 классов

    def forward(self, x):
        x = F.relu(self.c1(x))  # активация после первого слоя
        x = F.relu(self.c2(x))  # активация после второго слоя
        x = self.p(x)  # понижение размерности
        x = self.d(x)  # применение dropout
        x = x.view(x.size(0), -1)  # выпрямление тензора
        x = F.relu(self.f1(x))  # активация перед выходом
        return self.f2(x)  # необработанные логиты классов


def load_image(path: str) -> torch.Tensor:
    """Загружает изображение 28x28 и преобразует в тензор."""
    img = Image.open(path).convert("L")  # открываем и приводим к оттенкам серого
    if img.size != (28, 28):
        raise ValueError("Image must be 28x28 pixels")  # проверка размера
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )  # преобразование и нормализация
    return tfm(img).unsqueeze(0)  # добавляем размер батча


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # выбор устройства
net = Net().to(device)  # создаем модель и переносим на устройство
net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))  # загружаем веса
net.eval()  # переключаем в режим инференса

x = load_image(IMAGE_PATH).to(device)  # загружаем и подготавливаем изображение
with torch.no_grad():  # выключаем вычисление градиентов для инференса
    logits = net(x)
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)  # получаем наибольшую вероятность и класс

result = {"prediction": int(pred.item()), "confidence": float(conf.item())}
print(json.dumps(result, ensure_ascii=False))  # выводим результат в формате JSON

