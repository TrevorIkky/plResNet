import torch
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F
import pytorch_lightning as pl

import argparse

from typing import Any, Optional, Dict
from torch.optim import Adam, SGD
from pytorch_lightning.trainer import Trainer
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from cifar10_datamodule import Cifar10DataModule

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


momentum = 0.9
max_epochs = 30
batch_size = 32

class residual_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_block=None, stride=1):
        super(residual_block, self).__init__()
        self.intermediate_channels = intermediate_channels
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1, bias=False)
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
            y = self.identity_block(y)
        x += y
        return self.relu(x)


class ResNet(pl.LightningModule):
    def __init__(self, block, in_dim=3, classes=3,  layers=[3, 4, 6, 3], config=None):
        """Args:
        in_dim = number of channels in input image,
        classes = number of classes to predict,
        layers = number of residual blocks in each layer"""
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.in_dim = in_dim
        self.layers = layers
        self.classes = classes
        #uncomment for other datasets
        self.conv1 = nn.Conv2d(in_dim, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(in_dim, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64,  stride=1)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256 ,stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.Linear(2048, classes)

        self.val_accuracy = tm.Accuracy()
        self.test_accuracy = tm.Accuracy()
        self.train_accuracy = tm.Accuracy()

        self.config = config

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.output(x)
        return x

    def _make_layer(self, block, residual_blocks, channels, stride):
        """
        Args:
        channels = intermediate_channels
        """
        blocks = []
        identity_block = None

        if stride != 1 or self.in_channels != channels * 4:
            identity_block = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )
        """Downsampling performed by block_3_1, block_4_1, block_5_1 with a stride of 2"""

        blocks.append(block(self.in_channels,  channels, identity_block, stride))
        self.in_channels = channels * 4
        for _ in range(residual_blocks - 1):
            blocks.append(block(self.in_channels, channels))
        return nn.Sequential(*blocks)

    def configure_optimizers(self):
        learning_rate = 0.1
        weight_decay = 5e-4

        if self.config is not None:
            learning_rate = self.config['lr']
            weight_decay = self.config['weight_decay']

        optimizer= SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return { "optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor":  "val_accuracy"}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.train_accuracy(outputs, labels)
        return { "loss" : loss }


    def training_epoch_end(self, outputs):
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.val_accuracy(outputs, labels)
        return { "val_loss" : loss }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.test_accuracy(outputs, labels)
        return { "val_loss" : loss }

    def test_epoch_end(self, outputs):
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)


def train_resnet(config, num_epochs=10, num_gpus=1):

    layers = config['layers']
    batch_size = config['batch_size']

    cfm = Cifar10DataModule(batch_size=batch_size)
    model = ResNet(
        residual_block, in_dim=cfm.image_dims[0],
        classes=cfm.num_classes, layers=layers, config=config
    )

    ### Log metric progression
    logger = TensorBoardLogger('tuner_logs', name='regnet_logs')


    #Tune callback
    tune_report = TuneReportCallback({ "loss": "val_loss", "val_accuracy": "val_accuracy"}, on="validation_end")
    tune_report_ckpt = TuneReportCheckpointCallback(
        metrics={ "val_loss": "val_loss", "val_accuracy": "val_accuracy"},
        filename="tune_last_ckpt", on="validation_end"
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs, gpus= num_gpus, logger=logger,
        callbacks=[ tune_report, tune_report_ckpt ]
    )
    trainer.fit(model, cfm)


#======================================= Tuning Functions ============================================

def TuneAsha(train_fn, model:str, num_samples:int=10, num_epochs:int=10, cpus_per_trial:int=1, gpus_per_trial:int=1, data_dir='./tuner'):
    config = {
        "layers": [3, 4, 3, 2],
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-4, 1e-5),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "weight_decay"],
        metric_columns=["val_loss", "val_accuracy", "training_iteration"]
    )

    analysis = tune.run(
        tune.with_parameters(
            train_fn, num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_loss", mode="min", config=config,
        num_samples=num_samples, scheduler=scheduler, progress_reporter=reporter, name=f"tune_{model}_asha"
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    exit(0)

def TunePBT(train_fn, model:str, num_samples:int=10, num_epochs:int=10, cpus_per_trial:int=1, gpus_per_trial:int=1, data_dir='./tuner'):
    config = {
        "layers": [3, 4, 3, 2],
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-5),
        "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-5),
            "batch_size": [32, 64, 128]
        }
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "weight_decay"],
        metric_columns=["val_loss", "val_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_fn, num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_loss", mode="min", config=config,
        num_samples=num_samples, scheduler=scheduler, progress_reporter=reporter, name=f"{model}_asha"
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    exit(0)

#====================================== End Tuning Functions =====================================



if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', help='Find hyperparameter values', action='store_true')
    args = parser.parse_args()
    if args.tune:
        TuneAsha(
            train_resnet, 'resnet4', num_samples=1,
            num_epochs=1, cpus_per_trial=8, gpus_per_trial=0
        )

    cfm = Cifar10DataModule(batch_size=batch_size)
    model = ResNet(residual_block, in_dim=3, classes=10)

    ### Log metric progression
    logger = TensorBoardLogger('logs', name='regnet_logs')

    ### Callbacks
    stop_early = EarlyStopping(monitor='val_accuracy', patience=3)
    last_chkpt_path = 'checkpoints/last.ckpt'
    checkpoint = ModelCheckpoint(
        dirpath= last_chkpt_path, monitor='val_accuracy',
        filename='{epoch}-{val_accuracy:.2f}', verbose=True, save_top_k=1
    )

    trainer = Trainer(
        gpus=0, fast_dev_run=True, logger=logger,
        max_epochs=max_epochs, callbacks=[checkpoint],
    )

    trainer.fit(model, cfm)
