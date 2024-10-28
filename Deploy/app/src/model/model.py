"""

This file is used to define the model

Author: Kyrillos Botros
Date: Jul 26, 2023

"""


from monai.networks.nets import DenseNet201
from monai.losses import FocalLoss
import torch
from torch import optim
import torchmetrics
import pytorch_lightning as pl


class HemorrhageModel(pl.LightningModule):
    """
    This class is used to define the model
    """

    def __init__(self,
                 num_classes=6,
                 learning_rate=1e-3,
                 spatial_dims=2,
                 in_channels=1
                 ):
        """

        This class is used to define the model
        :param num_classes: the number of classes
        :param learning_rate: the learning rate for optimizer
        :param spatial_dims: the spatial dimensions of the input
        :param in_channels: the number of input channels
        """

        super().__init__()

        self.engine = DenseNet201(spatial_dims=spatial_dims,
                                  in_channels=in_channels,
                                  out_channels=num_classes,
                                  pretrained=True
                                  )
        self.loss_fn = FocalLoss(
            weight=torch.Tensor([1.4142, 8.2848, 2.4447, 2.8701, 2.4599, 2.1393])
        )
        self.optimizer = optim.Adam(self.engine.parameters(), lr=learning_rate)

        self.train_f1 = torchmetrics.F1Score(
            task='binary', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(
            task='binary', num_classes=num_classes)

    def forward(self, data):
        pred = self.engine(data)
        return pred

    def configure_optimizers(self):
        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        """
        This function is used to define the training step
        :param batch: the batch of data
        :param batch_idx: the batch index
        :return: the loss
        """
        images, labels = batch
        labels = labels.float()
        pred = self(images)
        loss = self.loss_fn(pred, labels)
        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True
                 )
        self.log("train_f1",
                 self.train_f1(pred, labels.int()),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True
                 )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        This function is used to define the validation step
        :param batch: the batch of data
        :param batch_idx: the batch index
        :return: the loss

        """
        images, labels = batch
        labels = labels.float()
        pred = self(images)
        loss = self.loss_fn(pred, labels)
        self.log("val_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True
                 )
        self.log("val_f1",
                 self.val_f1(pred, labels.int()),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True
                 )
