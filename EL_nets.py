import torch
import torch.nn as nn
from torchvision import models

def CreateNet(name, freeze_weights=False):
    net = None
    im_input_size = 0
    torch.manual_seed(0)
    
    if name == 'AkramNet-4a':
        net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        im_input_size = 100
    
    elif name == 'AkramNet-4b':
        net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(96),
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
        im_input_size = 100
        
    elif name == 'AkramNet-4c':
        net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 144, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(144),
            nn.Flatten(),
            nn.Linear(144 * 7 * 7, 768), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(768, 5)
        )
        im_input_size = 100
    
    elif name == 'AkramNet-5a':
        net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Flatten(),
            nn.Linear(48 * 5 * 5, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        im_input_size = 160
        
    elif name == 'AkramNet-5b':
        net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(96),
            nn.Flatten(),
            nn.Linear(96 * 5 * 5, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
        im_input_size = 160
        
    elif name == 'AkramNet-5c':
        net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(72),
            nn.Conv2d(72, 72, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(72),
            nn.Conv2d(72, 144, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(144),
            nn.Flatten(),
            nn.Linear(144 * 5 * 5, 768), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(768, 5)
        )
        im_input_size = 160
        
    elif name == 'AkramNet-6a':
        net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        im_input_size = 256
        
    elif name == 'AkramNet-6b':
        net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(96),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
        im_input_size = 256
        
    elif name == 'AkramNet-6c':
        net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(72),
            nn.Conv2d(72, 72, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(72),
            nn.Conv2d(72, 144, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), nn.BatchNorm2d(144),
            nn.Flatten(),
            nn.Linear(144 * 4 * 4, 768), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(768, 5)
        )
        im_input_size = 256
        
    elif name == 'SqueezeNet':
        net = models.squeezenet1_0(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1))
        im_input_size = 224
        
    elif name == 'SqueezeNet-1.1':
        net = models.squeezenet1_1(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1))
        im_input_size = 224
        
    elif name == 'VGG-11':
        net = models.vgg11_bn(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.classifier[6] = nn.Linear(4096, 5)
        im_input_size = 224
    
    elif name == 'VGG-13':
        net = models.vgg13_bn(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        if freeze_weights:
            for param in net.features.parameters():
                param.requires_grad = False
        net.classifier[6] = nn.Linear(4096, 5)
        im_input_size = 224
    
    elif name == 'VGG-16':
        net = models.vgg16_bn(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.classifier[6] = nn.Linear(4096, 5)
        im_input_size = 224
        
    elif name == 'VGG-19':
        net = models.vgg19_bn(pretrained=True)
        w = net.features[0].weight
        net.features[0].in_channels = 1
        net.features[0].weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.classifier[6] = nn.Linear(4096, 5)
        im_input_size = 224
    
    elif name == 'ResNet-18':
        net = models.resnet18(pretrained=True)
        w = net.conv1.weight
        net.conv1.in_channels = 1
        net.conv1.weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.fc = nn.Linear(512, 5)
        im_input_size = 224
        
    elif name == 'ResNet-34':
        net = models.resnet34(pretrained=True)
        w = net.conv1.weight
        net.conv1.in_channels = 1
        net.conv1.weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.fc = nn.Linear(512, 5)
        im_input_size = 224
        
    elif name == 'ResNet-50':
        net = models.resnet50(pretrained=True)
        w = net.conv1.weight
        net.conv1.in_channels = 1
        net.conv1.weight = torch.nn.Parameter(w.sum(dim=1, keepdim=True))
        net.fc = nn.Linear(2048, 5)
        im_input_size = 224
    
    return (net, im_input_size)
