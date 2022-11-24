import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import models


import numpy as np
import wandb
import pandas as pd
import cv2
from tqdm import tqdm
import argparse
from PIL import Image

from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model_VGG(nn.Module):
    def __init__(self, model_type):
        super(Model_VGG, self).__init__()

        if model_type == "vgg_16":
            self.model_backbone = models.vgg16(pretrained=True)
        elif model_type == "vgg_19":
            self.model_backbone = models.vgg19(pretrained=True)
        self.model_backbone.classifier[6] = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.model_backbone(x)
        return x

class Digit_Dataset(Dataset):
    def __init__(self, df_data, transforms=None):
        self.data = df_data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8)
        # 3 channels
        img = np.stack((img,) * 3, axis=-1)
        label = self.data.iloc[idx, 0]
        if self.transforms:
            img = self.transforms(img)
        return img, label


class Training_Model:
    def __init__(self, model, optimizer, criterion, dict_dataloader):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = dict_dataloader["train_loader"]
        self.val_loader = dict_dataloader["val_loader"]
        self.best_f1_score = 0

    def train(self, epochs):
        print("============TRAINING START {}============".format(epochs))
        self.model.train()
        total_loss = 0
        preds, targets = [], []
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data = data.to(device)
            target = target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            preds += predicted.tolist()
            targets += target.tolist()

        total_loss = round(total_loss / len(self.train_loader), 4)
        precision_scr = round(
            precision_score(targets, preds, average="macro"),
            4,
        )
        accuracy_scr = round(
            accuracy_score(targets, preds),
            4,
        )
        train_f1_scr = round(
            f1_score(targets, preds, average="macro"),
            4,
        )
        print(
            "train_loss: {}, train_precision: {}, train_accuracy: {}, train_f1_score: {}".format(
                total_loss, precision_scr, accuracy_scr, train_f1_scr
            )
        )
        wandb.log(
            {
                "train_loss": total_loss,
                "train_precision_score": precision_scr,
                "train_accuracy_score": accuracy_scr,
                "train_f1_score": train_f1_scr,
            }
        )

    def validation(self, epochs):
        print("============VAL START {}============".format(epochs))
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        for batch_idx, (data, target) in enumerate(tqdm(self.val_loader)):
            data = data.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                preds += predicted.tolist()
                targets += target.tolist()
        total_loss = round(total_loss / len(self.val_loader), 4)
        precision_scr = round(precision_score(targets, preds, average="macro"), 4)
        accuracy_scr = round(accuracy_score(targets, preds), 4)
        val_f1_scr = round(f1_score(targets, preds, average="macro"), 4)
        print(
            "val_loss: {}, val_precision: {}, val_accuracy: {}, val_f1_score: {}".format(
                total_loss, precision_scr, accuracy_scr, val_f1_scr
            )
        )
        wandb.log(
            {
                "val_loss": total_loss,
                "val_precision_score": precision_scr,
                "val_accuracy_score": accuracy_scr,
                "val_f1_score": val_f1_scr,
            }
        )
        if val_f1_scr > self.best_f1_score and val_f1_scr > 0.9:
            self.best_f1_score = val_f1_scr
            torch.save(
                self.model.state_dict(),
                "../../../weights/{}_f1_{}.pt".format(args.model_type, val_f1_scr),
            )
            torch.jit.save(
                torch.jit.script(self.model),
                self.model.state_dict(),
                "../../../weights/{}_jit_f1_{}.pt".format(args.model_type, val_f1_scr),
            )


def get_loader(csv_file, transforms, batch_size=64):
    dataset = Digit_Dataset(csv_file, transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vgg_16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    args = parser.parse_args()
    return args


def config_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = get_input()
    config_seed(args.seed)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay

    data = pd.read_csv("../../../DATA/train.csv")

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=10)

    transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    train_loader = get_loader(train_data, transforms=transforms, batch_size=batch_size)
    val_loader = get_loader(val_data, transforms=transforms, batch_size=batch_size)

    dict_dataloader = {"train_loader": train_loader, "val_loader": val_loader}

    model = Model_VGG(args.model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    wandb.init(
        project="Digit_Recognizer",
        group="VGG",
        name="{}_{}_{}_{}".format(args.model_type, batch_size, epochs, lr),
    )
    wandb.config.update(args)
    wandb.watch(model)

    training_model = Training_Model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dict_dataloader=dict_dataloader,
    )

    for epoch in range(epochs):
        training_model.train(epoch)
        training_model.validation(epoch)
