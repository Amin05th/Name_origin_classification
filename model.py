import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class NameClassificationModel(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, device, input_size, hidden_size, embedding_dim, num_layers, num_classes):
        super(NameClassificationModel, self).__init__()
        self.save_hyperparameters()
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        print(x)
        exit()
        x = self.embed(x.long())  # Convert data_ids_tensor to Long type
        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = self.fc(x[:, 1])
        return x

    def init_hidden(self, batch):
        h0 = torch.zeros(self.hparams.num_layers * 2, batch, self.hparams.hidden_size).to(self.hparams.device)
        c0 = torch.zeros(self.hparams.num_layers * 2, batch, self.hparams.hidden_size).to(self.hparams.device)
        return h0, c0

    def training_step(self, batch, batch_idx):
        data_ids_tensor, language_label_encoded = batch
        y_pred = self.forward(data_ids_tensor)
        print(y_pred)
        exit()

    def train_dataloader(self):
        return DataLoader(self.hparams.train_dataset, batch_size=64, num_workers=12, collate_fn=self.collate_fn)

    def validation_step(self, batch, batch_idx):
        pass

    def val_dataloader(self):
        return DataLoader(self.hparams.val_dataset, batch_size=64, num_workers=12, collate_fn=self.collate_fn)

    def test(self):
        pass

    def collate_fn(self, batch):
        data_ids_tensor, language_label_encoded = zip(*batch)
        data_ids_padded = pad_sequence(data_ids_tensor, batch_first=True)
        language_label_encoded = torch.cat(language_label_encoded)
        return data_ids_padded, language_label_encoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.30, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }