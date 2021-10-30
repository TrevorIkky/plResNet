import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer


learning_rate = 3e-4
max_epochs = 10
batch_size = 32

class residual_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_block=None, stride=1):
        super(residual_block, self).__init__()
        self.intermediate_channels = intermediate_channels
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * 4)
        self.identity_block = identity_block
    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_block is not None:
            x += self.identity_block(y)
        return self.relu(x)


class ResNet(pl.LightningModule):
    def __init__(self, in_dim=3, classes=3,  layers=[3, 4, 6, 3]):
        """Args:
        in_dim = number of channels in input image,
        classes = number of classes to predict,
        layers = number of residual blocks in each layer"""
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.in_dim = in_dim
        self.layers = layers
        self.classes = classes
        self.conv1 = nn.Conv2d(in_dim, self.in_channels, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(layers[0], 64,  stride=1)
        self.layer2 = self._make_layer(layers[1], 128, stride=2)
        self.layer3 = self._make_layer(layers[2], 256 ,stride=2)
        self.layer4 = self._make_layer(layers[3], 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.output(x)
        return x

    def _make_layer(self, residual_blocks, channels, stride):
        """
        Args:
        channels = intermediate_channels
        """
        blocks = []
        identity_block = None

        if stride != 1 or self.in_channels != channels:
            identity_block = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels * 4)
            )
        """Downsampling performed by block_3_1, block_4_1, block_5_1 with a stride of 2"""

        blocks.append(residual_block(self.in_channels,  channels, identity_block, stride))
        self.in_channels = channels * 4
        for _ in range(residual_blocks - 1):
            blocks.append(residual_block(self.in_channels, channels))
        return nn.Sequential(*blocks)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return { "loss" : loss }

    def train_dataloader(self):
        train_ds = torchvision.datasets.CIFAR10(
            './dataset', True, transform=torchvision.transforms.ToTensor(), download=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        return train_dl

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return { "val_loss" : loss }

    def val_dataloader(self):
        val_ds = torchvision.datasets.CIFAR10(
            './dataset', False, transform=torchvision.transforms.ToTensor(), download=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
        return val_dl


if  __name__ == "__main__":
    model = ResNet(3, 10)
    trainer =  Trainer(fast_dev_run=False)
    trainer.fit(model)
