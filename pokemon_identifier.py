# Name: Martin Jimenez
# Date: 05/16/2023 (last updated)

import torch
import torchvision.io
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
from typing import List, Dict
import os

import random
import time

from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

from tqdm.auto import tqdm

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

# set up transforms
train_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                      transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                      transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor()])

# set up data
train_data = datasets.ImageFolder(root=str(train_dir),
                                  transform=train_transform)
test_data = datasets.ImageFolder(root=str(test_dir),
                                 transform=test_transform)

# img, label = list(iter(train_data))[2000]
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

class_names = train_data.classes

# set up dataloaders
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
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
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2))
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_size,
                                                    out_channels=hidden_size,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=hidden_size*56*56,
                                                  out_features=output_size))

    def forward(self, x):
        x = self.conv_block_1(x)    # 32, 3, 224, 224 -> 32, 10, 112, 112
        x = self.conv_block_2(x)    # 32, 10, 112, 112 -> 32, 10, 56, 56
        x = self.classifier(x)      # 32, 10, 56, 56 -> 32, 9
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
    with tqdm(total=len(dataloader), desc='Train Step', leave=False, ncols=75, disable=False) as train_bar:
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
            pred_class = torch.argmax(pred, dim=1)
            train_acc += (pred_class == y).sum().item() / len(pred)

            train_bar.update()

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

    with tqdm(total=len(dataloader), desc='Test Step', leave=False, ncols=75, disable=False) as test_bar:
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                test_pred = model(X)

                loss = loss_fn(test_pred, y)
                test_loss += loss.item()

                test_pred_class = torch.argmax(test_pred, dim=1)
                test_acc += (test_pred_class == y).sum().item() / len(test_pred)

                test_bar.update()

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
    """Runs train and test loops for a given number of epochs"""
    print(f'Training model...\n')

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

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%')

    return results


# plot loss curves
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary"""
    # get the loss values of the results dictionary
    loss = results['train_loss']
    test_loss = results['test_loss']

    # get accuracy
    acc = results['train_acc']
    test_acc = results['test_acc']

    epochs = range(len(results['train_loss']))

    # set up the plot
    plt.figure(figsize=(15, 7))

    # plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='train_acc')
    plt.plot(epochs, test_acc, label='test_acc')
    plt.title('Acc')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()


def make_predictions(model: torch.nn.Module,
                     class_names: List[str],
                     dataloader: torch.utils.data.DataLoader,
                     device = device):
    """Makes 16 random predictions on a trained model"""
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
                           device = device,
                           output_num: int = 5):
    """Makes a prediction on a trained model given the image file path"""
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

    pred_percent = image_pred.softmax(dim=1)

    if output_num:
        print_predictions(prediction=pred_percent,
                          class_names=class_names,
                          pred_output=output_num)

    image_pred_label = torch.argmax(image_pred)

    pred_class = class_names[image_pred_label]

    # plot
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(pred_class)
    plt.axis(False)
    plt.show()


def print_predictions(prediction: torch.tensor,
                      class_names: List[str],
                      pred_output: int):
    pred_percent_dict = {}
    for i in range(len(class_names)):
        pred_percent_dict.update({class_names[i]: prediction[0][i].item() * 100})

    print(f'Top {pred_output} percentages | ', end='')
    for i in range(pred_output):
        percent = sorted(pred_percent_dict.values(), reverse=True)[i]
        pokemon = list(pred_percent_dict.keys())[list(pred_percent_dict.values()).index(percent)]

        end = '\n' if i == pred_output - 1 else ', '

        print(f'{percent:.4f}% {pokemon}', end=end)


def set_confusion_matrix(model: torch.nn.Module,
                         class_names: List[str],
                         dataloader: torch.utils.data.DataLoader):
    """Plots a confusion matrix on a trained model"""
    # make predictions with trained model
    pred_labels = []

    acc = 0

    model.eval()
    with tqdm(total=len(dataloader), desc='Confusion Matrix', leave=False, ncols=75, disable=False) as confusion_bar:
        with torch.inference_mode():
            for X, y in dataloader:
                pred = model(X)

                pred_label = torch.argmax(pred, dim=1)
                pred_labels.append(pred_label)

                acc += (pred_label == y).sum().item() / len(pred)

                confusion_bar.update()

    acc /= len(dataloader)

    # concatenate list of predictions into a tensor
    pred_labels_tensor = torch.cat(pred_labels)

    # setup confusion matrix
    confmat = ConfusionMatrix(num_classes=len(class_names),
                              task='multiclass')
    confmat_tensor = confmat(preds=pred_labels_tensor,
                             target=torch.from_numpy(np.array(dataloader.dataset.targets)))

    # plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10, 7))

    mode = str(dataloader.dataset.root).split('pokemon_images\\')[1]
    print(f'Model Accuracy for {mode} data is {acc*100:.2f}%')

    plt.title(f'{mode} data')
    plt.show()


def save_model(model: torch.nn.Module,
               model_name: str):
    """Saves a model to a 'model_saves' folder"""
    # saving the model
    model_path = Path('model_saves')
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    print(f'Saving model to: {model_save_path}')
    torch.save(model.state_dict(), model_save_path)


def load_model(model_name: str):
    """Loads a saved model from the 'model_saves' folder"""
    # load the model
    model_path = Path('model_saves')
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    loaded_model = PokemonIdentifier(input_size=3,
                                     hidden_size=64,
                                     output_size=len(class_names))

    loaded_model.load_state_dict(torch.load(model_save_path))

    return loaded_model


if __name__ == '__main__':
    # current_model = load_model(model_name='model_11_1-26.pth')
    current_model = PokemonIdentifier(input_size=3,
                                      hidden_size=64,
                                      output_size=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=current_model.parameters(),
                                 lr=0.001)

    start_time = time.time()

    model_results = train_model(model=current_model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=loss_function,
                                optimizer=optimizer,
                                epochs=10,
                                device=device)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Took {total_time:.2f}s to train the model')

    save_model(model=current_model,
               model_name='model_11_1-36.pth')

    plot_loss_curves(results=model_results)

    simple_transform = transforms.Compose([transforms.Resize(size=(224, 224), antialias=True)])

    set_confusion_matrix(model=current_model,
                         class_names=class_names,
                         dataloader=test_dataloader)

    make_predictions(model=current_model,
                     class_names=class_names,
                     dataloader=test_dataloader,
                     device=device)

    make_single_prediction(model=current_model,
                           transform=simple_transform,
                           img_path=str(random.choice(image_path_list)),
                           class_names=class_names,
                           device=device,
                           output_num=5)

    # USE IN CASE THERE IS TOO MUCH DATA; MOVES DATA TO EXTRA FOLDER
    '''
    import shutil
    pokemon = 'Pikachu'
    folder = 'train'
    pokemon_path_list = list(image_path.glob(f'{folder}/{pokemon}/*.jpg'))
    for i in range(80):
        source_file = random.choice(pokemon_path_list)
        destination_file = str(source_file).split('train\\')[1]
        shutil.move(source_file, destination_file)

        pokemon_path_list = list(image_path.glob(f'{folder}/{pokemon}/*.jpg'))
        print(f'Moved {source_file} to {destination_file}')
    '''

    # USE IN CASE ALL IMAGES AREN'T IN JPG FILE FORMAT:
    '''
    def convert_file_to_jpg(img_file, output_dir):
        # Open the PNG image
        image = Image.open(img_file)

        # Convert the image to RGB format (if it's not already)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Generate the output file path and filename
        filename = os.path.basename(img_file)
        jpg_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')

        # Save the image as a JPG file
        image.save(jpg_file, 'JPEG')

        print(f"Converted {img_file} to {jpg_file}")


    for pokemon in class_names:
        for folder in ['train', 'test']:
            # Path to the directory containing PNG files
            input_dir = f'data/pokemon_images/{folder}/{pokemon}'

            # Path to the directory where JPG files will be saved
            output_dir = f'data/pokemon_images/{folder}/{pokemon}'

            # Iterate over PNG files in the input directory
            base_file_type = '.jpg'
            for filename in os.listdir(input_dir):
                if not filename.endswith(base_file_type):
                    img_file = os.path.join(input_dir, filename)
                    convert_file_to_jpg(img_file, output_dir)
                    os.remove(img_file)
    '''

    # USE TO COUNT DATA IN FOLDERS
    '''
    total_train = 0
    total_test = 0
    for pokemon in class_names:
        for folder in ['test', 'train']:
            pokemon_path_list = list(image_path.glob(f'{folder}/{pokemon}/*.jpg'))
            print(f'there are {len(pokemon_path_list)} images of {pokemon} in the {folder} folder')

            if folder == 'test':
                total_test += len(pokemon_path_list)
            else:
                total_train += len(pokemon_path_list)

        print('\n')
    print(f'there are {total_test} images in the test data')
    print(f'there are {total_train} images in the train data')
    '''
