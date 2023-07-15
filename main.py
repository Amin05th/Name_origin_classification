import os
import string
import unicodedata
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, random_split
from model import NameClassificationModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# initialize cuda
device = "cuda" if torch.cuda.is_available() else "cpu"


# load data
def load_data():
    names_country_dict = {}
    names_directory = "data/names"
    languages_file_list = os.listdir(names_directory)
    for file in languages_file_list:
        language = file.split(".")[0]
        with open(f"{names_directory}/{file}", "r") as f:
            names_country_dict[language] = f.read().splitlines()
    return names_country_dict


# preprocess data
def preprocess_data():
    row_data = load_data()
    preprocessed_data = {}
    letters = string.ascii_letters + ".,:'"

    for country, data in row_data.items():
        one_hot_encoded_data = []

        for name in data:
            unidecode_name = [letter for letter in name if unicodedata.normalize("NFD", letter) in letters]
            one_hot_encoded_name = []

            for letter in unidecode_name:
                one_hot_encoded_letter = np.zeros(len(letters))
                indexed_letter = letters.index(letter)
                one_hot_encoded_letter[indexed_letter] = 1
                one_hot_encoded_name.append(one_hot_encoded_letter)

            one_hot_encoded_data.append(one_hot_encoded_name)

        preprocessed_data[country] = one_hot_encoded_data

    return preprocessed_data


class NameOriginDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.languages = list(self.data.keys())
        self.encoder = OneHotEncoder(categories=[self.languages], sparse_output=False)
        self.total_length = self.calculate_total_length()

    def calculate_total_length(self):
        total_length = 0
        for language in self.languages:
            total_length += len(self.data[language])
        return total_length

    def __getitem__(self, idx):
        language_index = 0
        sample_index = idx
        while sample_index >= len(self.data[self.languages[language_index]]):
            sample_index -= len(self.data[self.languages[language_index]])
            language_index += 1
        language_data = self.data[self.languages[language_index]]
        data_ids = np.array(language_data[sample_index])
        data_ids_tensor = torch.tensor(data_ids)
        language_label = self.languages[language_index]
        language_label_encoded = torch.Tensor(self.encoder.fit_transform([[language_label]]))

        return data_ids_tensor, language_label_encoded

    def __len__(self):
        return self.total_length


# created dataset and split dataset
dataset = NameOriginDataset(preprocess_data())
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# create model
model = NameClassificationModel(train_dataset, val_dataset, device, len(dataset), 3000, 100, 2, 18)

# set up Tensorboard
tb_logger = TensorBoardLogger(save_dir="./log", version="0")

# set up trainer for training
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1,
                     logger=tb_logger, precision="16-mixed")

trainer.fit(model)
model.test()
