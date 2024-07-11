import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VAE_MODIFIED import ConvVAE
from config import config

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

input_channels = 3  # CIFAR-10 images have 3 color channels
num_classes = 10
model = ConvVAE(input_channels=input_channels, num_classes=num_classes)
model.cuda()

epochs = 10
logs = {
    'train_loss': [],
    'test_loss': [],
    'train_accuracy': [],
    'test_accuracy': []
}

for epoch in range(1, epochs + 1):
    model.train_vae(trainloader, testset, num_classes, config, logs, name='conv_vae')