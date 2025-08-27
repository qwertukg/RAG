"""Простой пример обучения CNN на датасете MNIST."""

# базовые модули PyTorch
import torch, torch.nn as nn, torch.nn.functional as F
# инструменты для загрузки данных
from torch.utils.data import DataLoader
# готовые датасеты и преобразования изображений
from torchvision import datasets, transforms

# определение простой сверточной сети
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # первая свертка: 1 канал -> 32 карты признаков
        self.c1 = nn.Conv2d(1, 32, 3)     # 28->26
        # вторая свертка: 32 -> 64 карт
        self.c2 = nn.Conv2d(32, 64, 3)    # 26->24
        # слой подвыборки
        self.p  = nn.MaxPool2d(2)         # 24->12
        # регуляризация Dropout
        self.d  = nn.Dropout(0.25)
        # полносвязные слои
        self.f1 = nn.Linear(64*12*12, 128)
        self.f2 = nn.Linear(128, 10)
    def forward(self, x):
        # применение сверток и функций активации
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        # уменьшение пространственных размеров
        x = self.p(x)
        # отключение некоторых нейронов
        x = self.d(x)
        # выравнивание в вектор
        x = x.view(x.size(0), -1)
        # полносвязный слой с ReLU
        x = F.relu(self.f1(x))
        # окончательный слой предсказаний
        return self.f2(x)

if __name__ == "__main__":
    # гиперпараметры обучения
    BATCH, EPOCHS, LR = 128, 3, 1e-3
    # выбираем GPU при наличии, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # преобразование: в тензор и нормализация по среднему/стандартному отклонению
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # загрузка обучающего и тестового набора
    train = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    # обёртывание наборов в DataLoader для итерирования по мини-батчам
    tr = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    te = DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    # создание модели и оптимизатора
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # функция подсчёта точности на заданном наборе данных
    def accuracy(loader):
        net.eval()  # режим оценки отключает Dropout
        correct = total = 0
        with torch.no_grad():  # без вычисления градиентов
            for x, y in loader:  # проходим по батчам
                x, y = x.to(device), y.to(device)  # перенос на устройство
                logits = net(x)  # прямой проход
                pred = logits.argmax(1)  # предсказанный класс
                correct += (pred == y).sum().item()  # накопление правильных ответов
                total += y.size(0)  # общее количество примеров
        return correct / total

    # основной цикл обучения
    for e in range(1, EPOCHS + 1):
        net.train()  # режим обучения
        for x, y in tr:  # перебор батчей
            x, y = x.to(device), y.to(device)  # перенос данных
            opt.zero_grad()  # обнуление градиентов
            loss = F.cross_entropy(net(x), y)  # вычисление функции потерь
            loss.backward()  # обратное распространение
            opt.step()  # обновление параметров
        # выводим точность на тесте после эпохи
        print(f"Эпоха {e}: точность={accuracy(te):.4f}")

    # пример инференса для одного изображения из теста:
    x, y = test[0]  # берём первый тестовый пример
    with torch.no_grad():  # отключаем градиенты
        p = net(x.unsqueeze(0).to(device)).argmax(1).item()  # получаем предсказание модели
    print("Предсказано:", p, "Правильный ответ:", y)
    # сохраняем веса модели на диск
    torch.save(net.state_dict(), "mnist_cnn.pt")
