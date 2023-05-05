# Name: Martin Jimenez
# Date: 05/05/2023 (last updated)

import pathlib

import torch
import torchvision.io
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import requests
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List
import os

import random
from timeit import default_timer as timer

from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

"""
Test data downloaded from: https://www.kaggle.com/datasets/lantian773030/pokemonclassification?resource=download
Train data downloaded from: https://www.kaggle.com/datasets/thedagger/pokemon-generation-one
"""

# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set up data folders
data_path = Path('data/')
image_path = data_path / 'pokemon_images'

train_dir = image_path / 'train'
test_dir = image_path / 'test'

# set up transform
data_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor()])

# set up data
train_data = datasets.ImageFolder(root=str(train_dir),
                                  transform=data_transform)
test_data = datasets.ImageFolder(root=str(test_dir),
                                 transform=data_transform)

class_names = train_data.classes

# set up dataloaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=8,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=8,
                             shuffle=False)

# visualize images
image_path_list = list(image_path.glob('*/*/*.jpg'))

response = 'NULL'
while response != '':
    random_image_path = random.choice(image_path_list)

    img = Image.open(random_image_path)

    img.show()

    response = input('Would you like to delete this image? (Y/N)\n')

    if response.lower() == 'y':
        print(f'removed {random_image_path}\n')
        os.remove(random_image_path)
    else:
        print(f'did not remove {random_image_path}\n')


# make the model
class PokemonIdentifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=input_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2))
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=hidden_size*53*53,
                                                  out_features=output_size))

    def forward(self, x):
        x = self.conv_block_1(x)    # 1, 3, 224, 224 -> 1, 10, 110, 110
        x = self.conv_block_2(x)    # 1, 10, 110, 110 -> 1, 10, 53, 53
        x = self.classifier(x)      # 1, 10, 53, 53 -> 1, 9
        return x


# make a train step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device = device):
    # put model into train mode
    model.train()

    # declare loss and accuracy
    train_loss, train_acc = 0, 0

    # make prediction
    for batch, (X, y) in enumerate(dataloader):  # counter, (img, label)
        X, y = X.to(device), y.to(device)

        # make prediction
        pred = model(X)

        # calculate the loss
        loss = loss_fn(pred, y)  # (prediction, truth/label)
        train_loss += loss.item()

        # zero grad
        optimizer.zero_grad()

        # loss backwards
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate the accuracy
        pred_class = torch.argmax(pred)
        train_acc += (pred_class == y).sum().item() / len(pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


# make a test step function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device = device):
    # put model into eval mode
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            test_pred_class = torch.argmax(test_pred)
            test_acc += (test_pred_class == y).sum().item() / len(test_pred)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                device = device):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | '
              f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

    return results


def make_predictions(model: torch.nn.Module,
                     class_names: List[str],
                     dataloader: torch.utils.data.DataLoader,
                     device = device):

    model.to(device)
    # make prediction
    model.eval()
    with torch.inference_mode():
        fig = plt.figure(figsize=(9, 9))

        rows, cols = 4, 4  # grab 16 images

        for i in range(1, rows * cols + 1):
            random_idx = torch.randint(0, len(dataloader.dataset), size=[1]).item()  # random images

            image = dataloader.dataset[random_idx][0]

            pred_label = model(image.unsqueeze(0)).argmax(dim=1)
            truth_label = dataloader.dataset[random_idx][1]

            fig.add_subplot(rows, cols, i)

            plt.imshow(image.permute(1, 2, 0))

            if pred_label == truth_label:
                plt.title(f'M:{class_names[pred_label]} | T:{class_names[truth_label]}', fontsize=10, c='green')
            else:
                plt.title(f'M:{class_names[pred_label]} | T:{class_names[truth_label]}', fontsize=10, c='red')

            plt.axis(False)

    plt.show()


def make_single_prediction(model: torch.nn.Module,
                           transform,
                           img_path: str,
                           class_names: List[str],
                           device = device):
    image = torchvision.io.read_image(str(img_path)).type(torch.float32)

    image /= 255

    # transform
    image = transform(image)

    model.to(device)
    # make prediction
    model.eval()
    with torch.inference_mode():
        image = image.unsqueeze(dim=0)

        image_pred = model(image.to(device))

    image_pred_label = torch.argmax(image_pred)

    pred_class = class_names[image_pred_label]

    # plot
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(pred_class)
    plt.axis(False)
    plt.show()


def save_model(model: torch.nn.Module,
               model_name: str):
    # saving the model
    model_path = Path('model_saves')
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    print(f'Saving model to: {model_save_path}')
    torch.save(model.state_dict(), model_save_path)


def load_model(model_name: str):
    # load the model
    model_path = Path('model_saves')
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    loaded_model = PokemonIdentifier(input_size=3,
                                     hidden_size=10,
                                     output_size=len(class_names))

    loaded_model.load_state_dict(torch.load(model_save_path))

    return loaded_model


if __name__ == '__main__':
    # model_1 = PokemonIdentifier(input_size=3,
    #                             hidden_size=10,
    #                             output_size=len(class_names)).to(device)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params=model_1.parameters(),
    #                              lr=0.001)
    #
    # model_1_results = train_model(model=model_1,
    #                               train_dataloader=train_dataloader,
    #                               test_dataloader=test_dataloader,
    #                               loss_fn=loss_function,
    #                               optimizer=optimizer,
    #                               epochs=10,
    #                               device=device)
    #
    # save_model(model=model_1,
    #            model_name='model_1.pth')

    model_1 = load_model(model_name='model_1.pth')

    transform = transforms.Compose([transforms.Resize(size=(224, 224))])

    make_predictions(model=model_1,
                     class_names=class_names,
                     dataloader=test_dataloader,
                     device=device)

    make_single_prediction(model=model_1,
                           transform=transform,
                           img_path=r'data\pokemon_images\test\Venusaur\2f46340d214f481cbac464821ad4d069.jpg',
                           class_names=class_names,
                           device=device)





    # make predictions with trained model
    pred_labels = []
    model_1.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            pred = model_1(X)

            pred_label = torch.softmax(pred.squeeze(), dim=0).argmax(dim=1)
            pred_labels.append(pred_label)

    # concatenate list of predictions into a tensor
    pred_labels_tensor = torch.cat(pred_labels)

    # setup confusion matrix
    confmat = ConfusionMatrix(num_classes=len(class_names),
                              task='multiclass')
    confmat_tensor = confmat(preds=pred_labels_tensor,
                             target=torch.from_numpy(np.array(test_data.targets)))

    # plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10, 7))
    plt.show()
