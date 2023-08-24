import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torch.nn.init as init


class NameClassificationModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, device):
        super(NameClassificationModel, self).__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_size * 2, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.init_weights()

    def forward(self, x):
        hidden = self.init_hidden(x.size(0))
        lstm_out, hidden = self.lstm(x, self.init_hidden(x.size(0)))
        lstm_out = self.fc(lstm_out[-1])
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.fc2(lstm_out)
        return lstm_out

    def init_hidden(self, batch):
        return (torch.zeros(2, batch, self.hparams.hidden_size).to(self.hparams.device),
                torch.zeros(2, batch, self.hparams.hidden_size).to(self.hparams.device))

    def training_step(self, batch, batch_idx):
        tensored_country, tensored_name = batch
        y_pred = self.forward(tensored_name.squeeze(dim=0))
        loss = F.cross_entropy(y_pred, tensored_country[0])
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        tensored_country, tensored_name = batch
        y_pred = self.forward(tensored_name.squeeze(dim=0))
        loss = F.cross_entropy(y_pred, tensored_country[0])
        self.log("val_loss", loss.item())
        return loss

    def test(self, linetotensor, all_categories):
        self.to(self.hparams.device)
        while True:
            test_name = input("Enter name to get origin (type 'quit' to exit): ")
            tensored_name = linetotensor(test_name).to(self.hparams.device)
            target_prediction = self.forward(tensored_name)
            _, target_prediction_idx = torch.max(target_prediction, dim=1)
            print(all_categories[target_prediction_idx])
            if test_name == "quit":
                break

    def init_weights(self):
        for layer in [self.fc, self.fc1]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        clip_value = 0.8
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'amp_backend': 'native',
            "amp_level": '02',
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss'
            # }
        }

