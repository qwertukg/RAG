# mnist_cnn.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH, EPOCHS, LR = 128, 3, 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST(root="./data", train=True,  transform=tfm, download=True)
test  = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
tr = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
te = DataLoader(test,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)     # 28->26
        self.c2 = nn.Conv2d(32, 64, 3)    # 26->24
        self.p  = nn.MaxPool2d(2)         # 24->12
        self.d  = nn.Dropout(0.25)
        self.f1 = nn.Linear(64*12*12, 128)
        self.f2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.p(x)
        x = self.d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        return self.f2(x)

net = Net().to(device)
opt = torch.optim.Adam(net.parameters(), lr=LR)

def accuracy(loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = net(x)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
    return correct/total

for e in range(1, EPOCHS+1):
    net.train()
    for x,y in tr:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    print(f"Эпоха {e}: точность={accuracy(te):.4f}")

# пример инференса для одного изображения из теста:
x,y = test[0]
with torch.no_grad():
    p = net(x.unsqueeze(0).to(device)).argmax(1).item()
print("Предсказано:", p, "Правильный ответ:", y)
torch.save(net.state_dict(), "mnist_cnn.pt")