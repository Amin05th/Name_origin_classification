from io import open
import glob
import os
import torch
import unicodedata
import string
import random
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from model import NameClassificationModel

seed_everything(42)

def findFiles(path): return glob.glob(path)



all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
fixed_sequence_length = 12


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category_tensor, line_tensor


training_set = []
val_set = []

for _ in range(1000):
    training_example = randomTrainingExample()
    training_set.append(training_example)

for _ in range(1000):
    val_example = randomTrainingExample()
    val_set.append(val_example)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_hidden = 512
num_layers = 2
tb_logger = TensorBoardLogger(save_dir="./log", version="0")
test_lstm = NameClassificationModel(n_letters, n_hidden, n_categories, device=device)
trainer = Trainer(max_epochs=100, logger=tb_logger, default_root_dir="./checkpoints", accumulate_grad_batches=20)
trainer.fit(test_lstm, train_dataloaders=DataLoader(training_set, 1, num_workers=12),
            val_dataloaders=DataLoader(val_set, 1, num_workers=12))
trainer.save_checkpoint("./model.ckpt")
loaded_model = NameClassificationModel.load_from_checkpoint("./model.ckpt")
test_lstm.test(lineToTensor, all_categories)
